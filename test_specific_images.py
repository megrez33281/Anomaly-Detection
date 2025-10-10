import torch
from PIL import Image
import numpy as np
import os
import glob
import pandas as pd

from Config import Config, set_seed
from UNet_Autoencoder_Model import UNetAutoencoder
from Evaluation import CombinedLoss
import albumentations as A
from albumentations.pytorch import ToTensorV2

def test_all_epochs():
    """遍歷所有 epoch 的權重，計算指定測試圖片的異常分數，並生成總結報告。"""
    set_seed(Config.SEED)
    device = Config.DEVICE

    # 1. 找到所有 epoch 權重檔案
    checkpoint_dir = Config.CHECKPOINT_DIR
    checkpoints = sorted(
        glob.glob(os.path.join(checkpoint_dir, "*.pth")),
        key=lambda x: int(os.path.splitext(os.path.basename(x).split('_')[-1])[0])
    )

    if not checkpoints:
        print(f"錯誤：在 '{checkpoint_dir}' 中找不到任何權重檔案。")
        return

    # 2. 準備影像轉換和圖片路徑
    transform = A.Compose([
        A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2(),
    ])
    image_ids = ['1512', '1144', '832', '640', '1700', '97']
    test_dir = Config.TEST_DIR
    image_paths = [os.path.join(test_dir, f"{id}.png") for id in image_ids]

    all_results = []
    print("開始遍歷所有 Epoch 權重並計算【特徵空間】異常分數...")

    # 3. 遍歷每一個權重檔案
    for model_path in checkpoints:
        try:
            epoch_num = int(os.path.splitext(os.path.basename(model_path).split('_')[-1])[0])
        except (IndexError, ValueError):
            print(f"警告：無法從檔名 '{model_path}' 中解析 epoch 序號，已跳過。")
            continue

        print(f"--- 正在測試 Epoch {epoch_num} ---")
        model = UNetAutoencoder().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        with torch.no_grad():
            for img_path in image_paths:
                image_name = os.path.basename(img_path)
                image = np.array(Image.open(img_path).convert("RGB"))
                tensor_image = transform(image=image)['image'].unsqueeze(0).to(device)
                
                # 1. 重建圖片
                recon_image = model(tensor_image)
                
                # 2. 提取原始圖與重建圖的特徵
                features_orig = model.encode(tensor_image)
                features_recon = model.encode(recon_image)
                
                # 3. 計算特徵空間的 MSE 作為新的異常分數
                loss = torch.nn.functional.mse_loss(features_orig, features_recon).item()
                
                all_results.append({
                    "epoch": epoch_num,
                    "image": image_name,
                    "score": loss
                })
    
    # 4. 產出總結報告
    if all_results:
        df = pd.DataFrame(all_results)
        summary_df = df.pivot(index='epoch', columns='image', values='score')
        # 重新排列欄位順序，方便觀察
        ordered_columns = [f"{img_id}.png" for img_id in image_ids]
        summary_df = summary_df[ordered_columns]
        print("\n--- 所有 Epoch 分數總結 ---")
        print(summary_df)

if __name__ == "__main__":
    test_all_epochs()
