from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
import os
from sklearn.model_selection import train_test_split
from Config import Config
from Evaluation import *


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
        image = Image.open(img_path).convert("RGB")
        
        label = 0  # 預設為正常
        if self.label_transform:
            # 如果有label_transform，就會先用label_transform將其轉換成異常圖片
            image = self.label_transform(image)  # 套用異常變換
            label = 1  # 標記為異常

        if self.transform:
            image = self.transform(image)
            
        return image, label
    


def GenerateDataset():
    # 載入與分割資料（此處型態仍是路徑）
    all_image_paths = sorted(glob.glob(os.path.join(Config.TRAIN_DIR, "*.png")), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    train_paths, val_paths = train_test_split(all_image_paths, test_size=0.1, random_state=Config.SEED)
    print(f"Training set size: {len(train_paths)}")
    print(f"Validation set size: {len(val_paths)}")

    # --- 定義影像轉換（訓練或驗證資料的處理流程） ---
    train_transform = transforms.Compose([
        # 隨機裁剪一個區域，然後縮放到指定大小 Config.IMG_SIZE × Config.IMG_SIZE
        transforms.RandomResizedCrop(size=Config.IMG_SIZE, scale=(0.8, 1.0)),
        # 將像素值從[0, 255]映射到[0.0, 1.0]，shape會變成[channel, H, W]
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        # 直接縮放圖片至指定大小並轉換成Tensor格式
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
    ])

    # 生成異常樣本
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