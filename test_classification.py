import torch
import numpy as np
import os
import glob
import shutil
from PIL import Image
from tqdm import tqdm

from Config import Config, set_seed
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- Helper Functions ---
def calculate_histogram(image_tensor, bins=32):
    """計算 RGB 圖像的顏色直方圖"""
    image_tensor = image_tensor * 255
    r_hist = torch.histc(image_tensor[0], bins=bins, min=0, max=255)
    g_hist = torch.histc(image_tensor[1], bins=bins, min=0, max=255)
    b_hist = torch.histc(image_tensor[2], bins=bins, min=0, max=255)
    hist = torch.cat((r_hist, g_hist, b_hist))
    hist = hist / torch.sum(hist)
    return hist

def chi_squared_distance(hist1, hist2):
    """計算兩個直方圖之間的卡方距離"""
    epsilon = 1e-10
    return torch.sum((hist1 - hist2)**2 / (hist1 + hist2 + epsilon))

def test_hybrid_classification():
    """
    使用兩階段混合策略進行分類：
    1. 先用像素級最近鄰進行快速初步分類。
    2. 如果初步分類結果為紋理類別，則再用顏色直方圖法在紋理類別中進行二次精準分類。
    """
    set_seed(Config.SEED)
    device = Config.DEVICE
    BINS = 32

    # 1. 準備轉換
    # 像素比較法對解析度敏感，我們使用之前討論過較大的尺寸
    transform = A.Compose([
        A.Resize(height=Config.PIXEL_CLASSIFY_IMG_SIZE, width=Config.PIXEL_CLASSIFY_IMG_SIZE),
        ToTensorV2(),
    ])

    # 2. 建立代表圖庫 (同時準備像素和直方圖)
    print("建立代表圖（Exemplars）的像素及直方圖庫...")
    exemplar_tensors = {}
    exemplar_histograms = {}
    categories = list(Config.CLASS_FILENAME_MAPPING.keys())
    TEXTURE_CLASSES = ['地毯', '皮革', '木板', '磁磚', '鐵網']

    for category in tqdm(categories, desc="Loading Exemplars"):
        try:
            exemplar_id = Config.CLASS_FILENAME_MAPPING[category][0]
            exemplar_path = os.path.join(Config.TRAIN_DIR, f"{exemplar_id}.png")
            if os.path.exists(exemplar_path):
                image = np.array(Image.open(exemplar_path).convert("RGB"))
                image_tensor = transform(image=image)['image'].to(device)
                exemplar_tensors[category] = image_tensor
                # 只為紋理類別預先計算直方圖
                if category in TEXTURE_CLASSES:
                    exemplar_histograms[category] = calculate_histogram(image_tensor, bins=BINS)
            else:
                print(f"警告：找不到類別 '{category}' 的代表圖 '{exemplar_path}'")
        except IndexError:
            print(f"警告：類別 '{category}' 在 mapping 中為空。")

    print(f"成功建立 {len(exemplar_tensors)} 個像素代表圖和 {len(exemplar_histograms)} 個直方圖代表。")

    # 3. 準備輸出資料夾
    output_root_dir = 'classification_test_results_hybrid'
    if os.path.exists(output_root_dir):
        shutil.rmtree(output_root_dir)
    print(f"建立輸出資料夾: {output_root_dir}")
    for category in categories:
        os.makedirs(os.path.join(output_root_dir, category), exist_ok=True)

    # 4. 準備測試圖片
    test_image_paths = glob.glob(os.path.join(Config.TEST_DIR, '*.png'))
    print(f"\n開始對 {len(test_image_paths)} 張測試圖片進行混合策略分類...")
    
    # 5. 主迴圈
    for image_path in tqdm(test_image_paths, desc="Classifying Test Images"):
        image = np.array(Image.open(image_path).convert("RGB"))
        image_tensor = transform(image=image)['image'].to(device)

        # --- STAGE 1: Pixel-wise Classification ---
        min_pixel_diff = float('inf')
        initial_prediction = None
        for category, exemplar_tensor in exemplar_tensors.items():
            diff = torch.mean((image_tensor.float() - exemplar_tensor.float())**2).item()
            if diff < min_pixel_diff:
                min_pixel_diff = diff
                initial_prediction = category
        
        final_prediction = initial_prediction

        # --- STAGE 2: Histogram Refinement (if needed) ---
        if initial_prediction in TEXTURE_CLASSES:
            min_hist_dist = float('inf')
            test_hist = calculate_histogram(image_tensor, bins=BINS)
            
            # 只在紋理類別中，用直方圖法重新比較
            for category, exemplar_hist in exemplar_histograms.items():
                dist = chi_squared_distance(test_hist, exemplar_hist).item()
                if dist < min_hist_dist:
                    min_hist_dist = dist
                    final_prediction = category # 精煉預測結果

        # 6. 複製檔案
        if final_prediction:
            dest_folder = os.path.join(output_root_dir, final_prediction)
            shutil.copy(image_path, dest_folder)

    print(f"\n分類完成！所有測試圖片已複製到 '{output_root_dir}' 的對應子資料夾中。")

if __name__ == "__main__":
    test_hybrid_classification()