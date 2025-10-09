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
from Evaluation import CombinedLoss

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
        # 提取檔名作為 ID
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        # 讀取影像並轉為Numpy
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            # 套用轉換
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
    # 載入訓練好的權重
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE))
    # 開啟評估模式
    model.eval()
    print(f"Model loaded from {Config.MODEL_SAVE_PATH}")

    # --- 2. 準備測試資料 ---
    test_image_paths = sorted(glob.glob(os.path.join(Config.TEST_DIR, "*.png")), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    print(f"Found {len(test_image_paths)} images in the test set.")

    # 定義測試時的影像轉換
    test_transform = A.Compose([
        A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2(),
    ])

    # 建立測試資料集與資料載入器
    test_dataset = TestDataset(image_paths=test_image_paths, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)

    # --- 3. 執行推論並計算異常分數 ---
    criterion = CombinedLoss(ssim_weight=Config.SSIM_WEIGHT).to(Config.DEVICE)
    results = []

    with torch.no_grad(): # 關閉梯度計算
        for images, img_ids in tqdm(test_loader, desc="Inferencing on test set"):
            inputs = images.to(Config.DEVICE)
            
            # 模型重建
            outputs = model(inputs)
            
            # 計算異常分數
            batch_scores = criterion(outputs, inputs, reduction='none').cpu().numpy()
            
            # 記錄結果 (ID 和 分數)
            for img_id, score in zip(img_ids, batch_scores):
                results.append({"id": int(img_id), "score": score})

    # --- 4. 決定閾值並生成預測 ---
    # 將結果轉為 pandas DataFrame，並按分數從高到低排序
    results_df = pd.DataFrame(results).sort_values(by="score", ascending=False)

    # 計算要標記為異常的樣本數量 (前5%)
    num_anomalies = int(len(results_df) * 0.05)
    print(f"Marking the top {num_anomalies} samples (5%) as anomalies.")

    # 建立預測欄位，預設全為 0 (正常)
    results_df["prediction"] = 0
    # 將分數最高的前 num_anomalies 個樣本標記為 1 (異常)
    results_df.iloc[:num_anomalies, results_df.columns.get_loc('prediction')] = 1

    # --- 5. 生成提交檔案 ---
    # 根據ID重新排序，以符合提交格式要求
    submission_df = results_df[["id", "prediction"]]
    submission_df = submission_df.sort_values(by="id").reset_index(drop=True)
    
    submission_path = os.path.join(Config.ROOT_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    print(f"\nSubmission file created successfully at: {submission_path}")
    print(f"Total predictions: {len(submission_df)}")
    print(f"Anomalies predicted (1s): {submission_df['prediction'].sum()}")
