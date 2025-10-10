import torch
from PIL import Image
import numpy as np
import os
import pandas as pd

from Config import Config, set_seed
from UNet_Autoencoder_Model import UNetAutoencoder
import albumentations as A
from albumentations.pytorch import ToTensorV2

def evaluate_with_feature_library():
    """
    使用儲存的特徵庫來評估特定測試圖片的異常分數。
    """
    set_seed(Config.SEED)
    device = Config.DEVICE
    category = Config.TARGET_CLASS

    # 1. 載入指定的最佳模型和特徵庫
    model_name = f"best_model_{category}.pth"
    model_path = Config.MODEL_SAVE_PATH.format(TARGET_CLASS=category)
    feature_library_path = f"{category}_features.pt"

    if not os.path.exists(model_path) or not os.path.exists(feature_library_path):
        print(f"錯誤：找不到必要的檔案。")
        print(f"  - 模型路徑: {model_path} (是否存在: {os.path.exists(model_path)})")
        print(f"  - 特徵庫路徑: {feature_library_path} (是否存在: {os.path.exists(feature_library_path)})")
        return

    print(f"正在載入模型: {model_path}")
    model = UNetAutoencoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"正在載入特徵庫: {feature_library_path}")
    feature_library = torch.load(feature_library_path).to(device)
    print(f"特徵庫載入成功，維度: {feature_library.shape}")

    # 2. 準備影像轉換和待測圖片
    transform = A.Compose([
        A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2(),
    ])

    test_images = {
        '634': 'Normal', '379': 'Normal', '391': 'Normal',
        '851': 'Anomalous', '423': 'Anomalous', '462': 'Anomalous'
    }
    image_ids = list(test_images.keys())
    test_dir = Config.TEST_DIR
    
    results = []
    print("\n開始計算指定圖片的異常分數...")

    # 3. 對每張圖片計算異常分數
    with torch.no_grad():
        for img_id in image_ids:
            img_path = os.path.join(test_dir, f"{img_id}.png")
            if not os.path.exists(img_path):
                print(f"警告：找不到測試圖片 {img_path}，已跳過。")
                continue

            image = np.array(Image.open(img_path).convert("RGB"))
            tensor_image = transform(image=image)['image'].unsqueeze(0).to(device)

            # a. 提取測試圖片的特徵
            test_feature = model.encode(tensor_image)

            # b. 計算與特徵庫中所有向量的距離 (MSE)
            distances = torch.sum((test_feature - feature_library)**2, dim=[1, 2, 3])
            
            # c. 找到最小距離作為異常分數
            min_distance = torch.min(distances).item()

            results.append({
                "Image": f"{img_id}.png",
                "True Label": test_images[img_id],
                "Anomaly Score": min_distance
            })

    # 4. 顯示結果
    if results:
        df = pd.DataFrame(results)
        df_sorted = df.sort_values(by="Anomaly Score", ascending=False)
        print("\n--- 評估結果 ---")
        print(df_sorted.to_string(index=False))

if __name__ == "__main__":
    evaluate_with_feature_library()
