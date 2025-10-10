import torch
from PIL import Image
import numpy as np
import os

from Config import Config, set_seed
from UNet_Autoencoder_Model import UNetAutoencoder
from Evaluation import CombinedLoss
import albumentations as A
from albumentations.pytorch import ToTensorV2

def run_inference_on_specific_images():
    """對指定的測試圖片進行推論，並輸出異常分數。"""
    set_seed(Config.SEED)
    device = Config.DEVICE
    
    # 1. 載入模型
    model = UNetAutoencoder().to(device)
    # 從 Config.py 讀取模型路徑，確保與訓練時一致
    model_path = Config.MODEL_SAVE_PATH 
    if not os.path.exists(model_path):
        print(f"錯誤：在 '{model_path}' 找不到模型。")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"模型已從 '{model_path}' 載入。 সন")

    # 2. 定義影像轉換 (必須與驗證時的轉換一致)
    transform = A.Compose([
        A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2(),
    ])

    # 3. 準備要測試的圖片列表
    image_ids = ['1512', '1144', '832', '640', '1700', '97']
    test_dir = Config.TEST_DIR
    image_paths = [os.path.join(test_dir, f"{id}.png") for id in image_ids]

    # 4. 定義損失函數 (用於計算異常分數)
    criterion = CombinedLoss(ssim_weight=Config.SSIM_WEIGHT).to(device)

    # 5. 執行推論並計算分數
    print("\n開始對指定圖片進行推論...")
    results = {}
    with torch.no_grad():
        for img_path in image_paths:
            if not os.path.exists(img_path):
                print(f"警告：找不到圖片 '{img_path}'，已跳過。 সন")
                continue

            image = np.array(Image.open(img_path).convert("RGB"))
            tensor_image = transform(image=image)['image'].unsqueeze(0).to(device)
            
            # 模型重建
            recon_image = model(tensor_image)
            
            # 計算異常分數 (重建損失)
            loss = criterion(recon_image, tensor_image, reduction='none').detach().cpu().item()
            
            image_name = os.path.basename(img_path)
            results[image_name] = loss
            print(f"  - 圖片: {image_name:<10} | 異常分數: {loss:.6f}")
            
    return results

if __name__ == "__main__":
    run_inference_on_specific_images()
