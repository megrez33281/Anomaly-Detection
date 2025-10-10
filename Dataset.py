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
    # --- 根據 Config.py 中的字典載入特定類別的圖片路徑 ---
    class_mapping = Config.CLASS_FILENAME_MAPPING
    target_class_files = class_mapping.get(Config.TARGET_CLASS)
    if not target_class_files:
        raise ValueError(f"Target class '{Config.TARGET_CLASS}' not found in mapping dictionary.")

    flat_train_dir = os.path.join(Config.DATA_DIR, "train")
    all_image_paths = [os.path.join(flat_train_dir, f"{file_id}.png") for file_id in target_class_files]

    # 載入與分割資料
    train_paths, val_paths = train_test_split(all_image_paths, test_size=0.2, random_state=Config.SEED)
    print(f"Target Class: {Config.TARGET_CLASS}")
    print(f"Training set size: {len(train_paths)}")
    print(f"Validation set size: {len(val_paths)}")

    # --- 定義影像轉換（albumentations） ---
    train_transform = A.Compose([
        A.RandomResizedCrop(size=(Config.IMG_SIZE, Config.IMG_SIZE), scale=(0.8, 1.0)),
        A.GridDropout(ratio=0.5, p=0.5),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2(),
    ])

    # --- 根據類別選擇驗證策略 ---
    TEXTURE_CLASSES = ['地毯', '皮革', '木板', '磁磚', '鐵網']
    OBJECT_CLASSES = ['拉鍊', '螺絲', '螺母', '瓶子', '藥片', '膠囊', '榛果', '電晶體', '電纜', '牙刷']

    if Config.TARGET_CLASS in TEXTURE_CLASSES:
        print("Validation Strategy: TextureDamage (for texture class)")
        anomaly_transform = TextureDamage()
    else:
        print("Validation Strategy: CutPaste (for object class)")
        anomaly_transform = CutPaste()

    # --- 建立 Dataset 與 DataLoader ---
    train_dataset = AnomalyDataset(image_paths=train_paths, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)

    val_normal_dataset = AnomalyDataset(image_paths=val_paths, transform=val_transform)
    val_anomaly_dataset = AnomalyDataset(image_paths=val_paths, transform=val_transform, label_transform=anomaly_transform)
    val_dataset = torch.utils.data.ConcatDataset([val_normal_dataset, val_anomaly_dataset])
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader