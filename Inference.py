import torch
import os
import glob
import pandas as pd
from tqdm import tqdm
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

# ============================================================ 
# 匯入自定義模組
# ============================================================ 
from Config import Config
from UNet_Autoencoder_Model import UNetAutoencoder

# ============================================================
# 全局設定
# ============================================================
# 手動指定最佳模型的類別與 epoch
BEST_MODEL_CLASS = "地毯"
BEST_MODEL_EPOCH = 15
# ============================================================

# ============================================================
# 推論專用資料集 (Test Dataset)
# ============================================================
class TestDataset(Dataset):
    """專為推論設計的資料集，僅載入影像並記錄檔名"""
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        image = np.array(Image.open(img_path).convert("RGB"))
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image, img_id

# ============================================================
# 主推論程式
# ============================================================
if __name__ == "__main__":
    print(f"Using device: {Config.DEVICE}")

    # --- 1. 準備模型 ---
    model = UNetAutoencoder().to(Config.DEVICE)
    model_path = os.path.join(Config.CHECKPOINT_DIR, f"model_{BEST_MODEL_CLASS}_epoch_{BEST_MODEL_EPOCH}.pth")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"指定的最佳模型權重不存在：{model_path}")

    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    model.eval()
    print(f"成功載入最佳模型：'{model_path}'")

    # --- 2. 準備測試資料 ---
    test_image_paths = sorted(glob.glob(os.path.join(Config.TEST_DIR, "*.png")), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    print(f"找到 {len(test_image_paths)} 張測試圖片。")

    test_transform = A.Compose([
        A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2(),
    ])

    test_dataset = TestDataset(image_paths=test_image_paths, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)

    # --- 3. 執行推論並使用「特徵空間比較」計算異常分數 ---
    results = []
    print("開始對所有測試圖片進行推論，並計算特徵空間異常分數...")

    with torch.no_grad():
        for images, img_ids in tqdm(test_loader, desc="Inferencing on test set"):
            inputs = images.to(Config.DEVICE)
            
            # 重建圖片
            outputs = model(inputs)
            
            # 提取原始與重建圖片的特徵
            features_orig = model.encode(inputs)
            features_recon = model.encode(outputs)

            # 計算每個樣本在特徵空間的 MSE
            batch_scores = torch.mean((features_orig - features_recon) ** 2, dim=[1, 2, 3]).cpu().numpy()

            for img_id, score in zip(img_ids, batch_scores):
                results.append({"id": int(img_id), "score": score})

    # --- 4. 決定閾值並生成預測 ---
    results_df = pd.DataFrame(results).sort_values(by="score", ascending=False)
    num_anomalies = int(len(results_df) * 0.05)
    print(f"將分數最高的 {num_anomalies} 個樣本 (5%) 標記為異常。")

    results_df["prediction"] = 0
    results_df.iloc[:num_anomalies, results_df.columns.get_loc('prediction')] = 1

    # --- 5. 生成提交檔案 ---
    submission_df = results_df[["id", "prediction"]]
    submission_df = submission_df.sort_values(by="id").reset_index(drop=True)
    
    submission_path = os.path.join(Config.ROOT_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    print(f"\n提交檔案已成功生成： {submission_path}")
    print(f"總預測筆數： {len(submission_df)}")
    print(f"預測為異常 (1) 的數量： {submission_df['prediction'].sum()}")
