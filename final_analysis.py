import torch
import numpy as np
import os
import glob
import shutil
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans

from Config import Config, set_seed
from UNet_Autoencoder_Model import UNetAutoencoder
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- Helper Functions ---
def calculate_histogram(image_tensor, bins=32):
    """計算 RGB 圖像的顏色直方圖"""
    image_tensor = image_tensor * 255
    hists = [torch.histc(channel, bins=bins, min=0, max=255) for channel in image_tensor]
    hist = torch.cat(hists)
    hist = hist / torch.sum(hist)
    return hist

def chi_squared_distance(hist1, hist2):
    """計算兩個直方圖之間的卡方距離"""
    epsilon = 1e-10
    return torch.sum((hist1 - hist2)**2 / (hist1 + hist2 + epsilon))

def final_analysis_pipeline():
    """
    執行最終的、分層的分析流程：
    1. 混合分類法：使用像素+直方圖法對測試集進行初步分類。
    2. 模型計分：對每個分類後的組，使用對應的類別模型計算異常分數。
    3. 動態閾值：在每個組內，根據頭部樣本的平均跳躍幅度，動態尋找閾值。
    4. 結果歸檔：將圖片複製到對應的 normal/abnormal 資料夾。
    """
    set_seed(Config.SEED)
    device = Config.DEVICE
    
    # --- Constants ---
    BINS = 32
    JUMP_SENSITIVITY_MULTIPLIER = 5.0 # 一次性跳躍多少倍的前幾名平均的MIN_SAMPLES_FOR_THRESHOLD才開始算做異常
    MIN_SAMPLES_FOR_THRESHOLD = 21 # 至少需要21個樣本才能計算前20個的跳躍
    CLASSIFY_IMG_SIZE = Config.PIXEL_CLASSIFY_IMG_SIZE
    SCORING_IMG_SIZE = Config.IMG_SIZE

    # --- Part 1: Load all assets ---
    print("--- Part 1: Loading all models and feature libraries ---")
    try:
        all_feature_data = torch.load('feature_libraries_all.pt', map_location=torch.device('cpu'))
    except FileNotFoundError:
        print("錯誤：找不到 'feature_libraries_all.pt'。請先執行 create_feature_library.py。")
        return

    categories = list(all_feature_data.keys())
    models = {}
    for category in tqdm(categories, desc="Loading Models"):
        model_path = Config.MODEL_SAVE_PATH.format(TARGET_CLASS=category) if category != '地毯' else os.path.join(Config.CHECKPOINT_DIR, 'model_地毯_epoch_15.pth')
        if not os.path.exists(model_path):
            print(f"警告：找不到類別 '{category}' 的模型，計分時將無法處理此類別。")
            continue
            
        model = UNetAutoencoder().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models[category] = model
    
    classify_transform = A.Compose([A.Resize(height=CLASSIFY_IMG_SIZE, width=CLASSIFY_IMG_SIZE), A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)), ToTensorV2()])
    
    exemplar_tensors = {}
    exemplar_histograms = {}
    TEXTURE_CLASSES = ['地毯', '皮革', '木板', '磁磚', '鐵網']
    for category in tqdm(categories, desc="Loading Exemplars"):
        try:
            exemplar_id = Config.CLASS_FILENAME_MAPPING[category][0]
            exemplar_path = os.path.join(Config.TRAIN_DIR, f"{exemplar_id}.png")
            if os.path.exists(exemplar_path):
                image = np.array(Image.open(exemplar_path).convert("RGB"))
                image_tensor = classify_transform(image=image)['image'].to(device)
                exemplar_tensors[category] = image_tensor
                if category in TEXTURE_CLASSES:
                    exemplar_histograms[category] = calculate_histogram(image_tensor.cpu(), bins=BINS)
        except IndexError:
            print(f"警告：類別 '{category}' 在 mapping 中為空。")

    print(f"成功載入 {len(models)} 個模型和 {len(exemplar_tensors)} 個代表圖。")

    # --- Part 2: Initial Classification ---
    print("\n--- Part 2: Classifying all test images ---")
    output_root_dir = 'final_classification_results'
    if os.path.exists(output_root_dir): shutil.rmtree(output_root_dir)
    
    classified_images = {category: [] for category in categories}
    test_image_paths = glob.glob(os.path.join(Config.TEST_DIR, '*.png'))

    for image_path in tqdm(test_image_paths, desc="Stage 1 Classification"):
        image = np.array(Image.open(image_path).convert("RGB"))
        image_tensor = classify_transform(image=image)['image'].to(device)

        min_pixel_diff = float('inf')
        initial_prediction = None
        for category, exemplar_tensor in exemplar_tensors.items():
            diff = torch.mean((image_tensor.float() - exemplar_tensor.float())**2).item()
            if diff < min_pixel_diff:
                min_pixel_diff = diff
                initial_prediction = category
        
        final_prediction = initial_prediction
        if initial_prediction in TEXTURE_CLASSES:
            min_hist_dist = float('inf')
            test_hist = calculate_histogram(image_tensor.cpu(), bins=BINS)
            for category, exemplar_hist in exemplar_histograms.items():
                dist = chi_squared_distance(test_hist, exemplar_hist).item()
                if dist < min_hist_dist:
                    min_hist_dist = dist
                    final_prediction = category
        
        if final_prediction:
            classified_images[final_prediction].append(image_path)

    # --- Part 3 & 4: Scoring, Dynamic Thresholding, and Placement ---
    print("\n--- Part 3 & 4: Scoring, Thresholding, and Archiving ---")
    for category, image_paths in classified_images.items():
        print(f"\nProcessing classified group: [{category}] ({len(image_paths)} images)")
        normal_dir = os.path.join(output_root_dir, category, 'normal')
        abnormal_dir = os.path.join(output_root_dir, category, 'abnormal')
        os.makedirs(normal_dir, exist_ok=True)
        os.makedirs(abnormal_dir, exist_ok=True)

        if not image_paths or category not in models:
            print("  -> 跳過 (無圖片或無模型)")
            continue

        model = models[category]
        feature_lib = all_feature_data[category]['features'].to(device)
        
        scores_for_category = []
        scoring_transform = A.Compose([A.Resize(height=SCORING_IMG_SIZE, width=SCORING_IMG_SIZE), A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)), ToTensorV2()])
        
        for image_path in tqdm(image_paths, desc=f"Scoring {category}", leave=False):
            image = np.array(Image.open(image_path).convert("RGB"))
            image_tensor = scoring_transform(image=image)['image'].unsqueeze(0).to(device)
            with torch.no_grad():
                test_feature = model.encode(image_tensor)
                distances = torch.sum((test_feature - feature_lib)**2, dim=[1, 2, 3])
                score = torch.min(distances).item()
                scores_for_category.append({'path': image_path, 'score': score})

        if len(scores_for_category) < 2:
            print("  -> 樣本數不足，全部歸為正常。")
            for item in scores_for_category:
                shutil.copy(item['path'], normal_dir)
            continue

        # --- K-Means Clustering ---
        scores_array = np.array([item['score'] for item in scores_for_category]).reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=Config.SEED, n_init='auto').fit(scores_array)
        
        # 找出中心點分數較高的那一群作為異常群
        abnormal_cluster_label = np.argmax(kmeans.cluster_centers_)
        print(f"  -> K-Means 完成。異常群中心: {kmeans.cluster_centers_[abnormal_cluster_label][0]:.6e}, 正常群中心: {kmeans.cluster_centers_[1 - abnormal_cluster_label][0]:.6e}")

        predicted_labels = kmeans.labels_

        for i, item in enumerate(scores_for_category):
            if predicted_labels[i] == abnormal_cluster_label:
                shutil.copy(item['path'], abnormal_dir)
            else:
                shutil.copy(item['path'], normal_dir)
    
    print("\n\n分析完成！所有圖片已根據預測分類及動態閾值放入對應資料夾。")

if __name__ == "__main__":
    final_analysis_pipeline()