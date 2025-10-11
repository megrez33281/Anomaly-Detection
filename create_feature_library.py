import torch
import numpy as np
import os
import glob
from tqdm import tqdm
from PIL import Image

from Config import Config, set_seed
from UNet_Autoencoder_Model import UNetAutoencoder
import albumentations as A
from albumentations.pytorch import ToTensorV2

def create_unified_feature_library(epoch):
    """
    載入指定 epoch 的統一模型，為所有正常訓練樣本建立一個
    統一的特徵庫，並將特徵與對應的圖片路徑儲存下來。
    """
    set_seed(Config.SEED)
    device = Config.DEVICE

    print(f"為 Epoch {epoch} 建立統一特徵庫...")
    print("======================================================")

    # 1. 載入指定 epoch 的模型
    model_path = os.path.join(Config.CHECKPOINT_DIR, f'model_epoch_{epoch}.pth')
    if not os.path.exists(model_path):
        print(f"錯誤：找不到模型 '{model_path}'，請確認權重檔案是否存在。")
        return

    print(f"正在載入模型: {model_path}")
    model = UNetAutoencoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. 準備影像轉換和所有訓練圖片的路徑
    transform = A.Compose([
        A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2(),
    ])
    
    # 使用全部的訓練資料來建立完整的特徵庫
    all_image_paths = sorted(glob.glob(os.path.join(Config.TRAIN_DIR, "*.png")), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    if not all_image_paths:
        print(f"警告：在 {Config.TRAIN_DIR} 中找不到任何圖片。")
        return

    print(f"找到 {len(all_image_paths)} 張訓練圖片，開始提取特徵...")

    # 3. 提取所有圖片的特徵
    all_features = []
    all_paths = []
    with torch.no_grad():
        for img_path in tqdm(all_image_paths, desc=f"Extracting features for Epoch {epoch}"):
            if not os.path.exists(img_path): continue
            
            image = np.array(Image.open(img_path).convert("RGB"))
            tensor_image = transform(image=image)['image'].unsqueeze(0).to(device)
            
            features = model.encode(tensor_image)
            
            all_features.append(features.squeeze().cpu())
            all_paths.append(img_path)
    
    # 4. 儲存整個特徵庫
    if all_features:
        stacked_features = torch.stack(all_features)
        
        feature_library = {
            'paths': all_paths,
            'features': stacked_features
        }
        
        output_filename = f"feature_library_epoch_{epoch}.pt"
        output_path = os.path.join(Config.ROOT_DIR, output_filename)
        torch.save(feature_library, output_path)
        
        print("\n======================================================")
        print(f"成功！已將 {len(all_paths)} 個特徵向量儲存至: {output_path}")
        print(f"特徵庫維度: {stacked_features.shape}")

if __name__ == "__main__":
    # 根據視覺化分析，只有 Epoch 1 的模型有學到東西
    TARGET_EPOCH = 1 
    create_unified_feature_library(TARGET_EPOCH)
