from PIL import Image
from torch.utils.data import Dataset, DataLoader
import glob
import os
from sklearn.model_selection import train_test_split
from Config import Config
from Evaluation import *
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


# -----------------------------
# 自定義 Dataset
# -----------------------------
class AnomalyDataset(Dataset):
    """自訂異常檢測資料集，可載入正常或異常影像"""
    def __init__(self, image_paths, transform=None, label_transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.label_transform = label_transform  # 若設定則對影像製造異常

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # PIL讀取影像，並轉換為Numpy array，以供albumentations使用
        image = np.array(Image.open(img_path).convert("RGB"))
        
        label = 0  # 預設為正常
        if self.label_transform:
            # 如果有label_transform，就會先用label_transform將其轉換成異常圖片
            # 此處的label_transform仍作用於PIL Image，因此需先轉回PIL
            pil_image = Image.fromarray(image)
            pil_image = self.label_transform(pil_image)
            image = np.array(pil_image)
            label = 1  # 標記為異常

        if self.transform:
            # 套用albumentations的轉換
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, label
    


def GenerateDataset():
    # 載入與分割資料（此處型態仍是路徑）
    all_image_paths = sorted(glob.glob(os.path.join(Config.TRAIN_DIR, "*.png")), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    train_paths, val_paths = train_test_split(all_image_paths, test_size=0.1, random_state=Config.SEED)
    print(f"Training set size: {len(train_paths)}")
    print(f"Validation set size: {len(val_paths)}")

    # --- 定義影像轉換（albumentations） ---
    # 訓練資料增強：使用 Cutout (CoarseDropout 的前身)
    train_transform = A.Compose([
        # 隨機裁剪一個區域，然後縮放到指定大小
        A.RandomResizedCrop(size=(Config.IMG_SIZE, Config.IMG_SIZE), scale=(0.8, 1.0)),
        # GridDropout：網格狀地挖掉一些區域，強迫模型學習紋理
        A.GridDropout(ratio=0.5, p=0.5),
        # 將像素值從[0, 255]映射到[0.0, 1.0]，並轉換為Tensor
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)), # 僅做歸一化
        ToTensorV2(),
    ])

    # 驗證資料轉換：僅縮放與轉換為Tensor
    val_transform = A.Compose([
        A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2(),
    ])

    # 生成異常樣本的轉換 (CutPaste)
    cutpaste_transform = CutPaste()  

    # --- 建立 Dataset 與 DataLoader ---
    train_dataset = AnomalyDataset(image_paths=train_paths, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)

    # 驗證集：一半正常、一半CutPaste製造的異常
    val_normal_dataset = AnomalyDataset(image_paths=val_paths, transform=val_transform)
    val_anomaly_dataset = AnomalyDataset(image_paths=val_paths, transform=val_transform, label_transform=cutpaste_transform)
    val_dataset = torch.utils.data.ConcatDataset([val_normal_dataset, val_anomaly_dataset])
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader