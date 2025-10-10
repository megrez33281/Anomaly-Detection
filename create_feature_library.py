import torch
import numpy as np
import os
from tqdm import tqdm
from PIL import Image

from Config import Config, set_seed
from UNet_Autoencoder_Model import UNetAutoencoder
import albumentations as A
from albumentations.pytorch import ToTensorV2

def create_all_feature_libraries_with_centers():
    """
    遍歷所有類別，為每個類別載入其最佳模型，並使用100%的正常樣本
    建立特徵庫。同時計算每個庫的中心點。最終將所有數據
    （特徵庫 + 中心點）儲存在一個字典檔案中。
    """
    set_seed(Config.SEED)
    device = Config.DEVICE
    all_data = {} # 改名以更清晰地表示儲存所有數據
    all_classes = list(Config.CLASS_FILENAME_MAPPING.keys())

    print(f"開始建立所有 {len(all_classes)} 個類別的特徵庫及中心點...")
    print("======================================================")

    for category in all_classes:
        print(f"\nProcessing Category: [{category}]")
        print("----------------------------------------------------")

        # 1. 載入該類別的最佳模型
        if category == '地毯': # 地毯是手動挑選的特例
            model_path = os.path.join(Config.CHECKPOINT_DIR, 'model_地毯_epoch_15.pth')
        else:
            model_path = Config.MODEL_SAVE_PATH.format(TARGET_CLASS=category)

        if not os.path.exists(model_path):
            print(f"警告：找不到模型 '{model_path}'，已跳過此類別。")
            continue

        print(f"正在載入模型: {model_path}")
        model = UNetAutoencoder().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # 2. 準備影像轉換和該類別的全部圖片路徑
        transform = A.Compose([
            A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
            ToTensorV2(),
        ])
        
        file_ids = Config.CLASS_FILENAME_MAPPING.get(category, [])
        image_paths = [os.path.join(Config.TRAIN_DIR, f"{fid}.png") for fid in file_ids]
        
        if not image_paths:
            print(f"警告：找不到 '{category}' 的任何圖片，已跳過。")
            continue

        print(f"找到 {len(image_paths)} 張 '{category}' 類別的圖片，開始提取特徵...")

        # 3. 提取所有圖片的特徵
        category_features = []
        with torch.no_grad():
            for img_path in tqdm(image_paths, desc=f"Extracting {category}"):
                if not os.path.exists(img_path): continue
                image = np.array(Image.open(img_path).convert("RGB"))
                tensor_image = transform(image=image)['image'].unsqueeze(0).to(device)
                features = model.encode(tensor_image)
                category_features.append(features.squeeze().cpu())
        
        # 4. 建立特徵庫並計算中心點
        if category_features:
            stacked_features = torch.stack(category_features)
            center_feature = torch.mean(stacked_features, dim=0)
            
            all_data[category] = {
                'features': stacked_features,
                'center': center_feature
            }
            print(f"'{category}' 特徵庫建立完成，維度: {stacked_features.shape}")
            print(f"'{category}' 中心點計算完成，維度: {center_feature.shape}")

    # 5. 儲存整個字典
    if all_data:
        output_path = "feature_libraries_all.pt"
        torch.save(all_data, output_path)
        print("\n======================================================")
        print(f"成功！已將 {len(all_data)} 個類別的特徵庫及中心點儲存至: {output_path}")

if __name__ == "__main__":
    create_all_feature_libraries_with_centers()