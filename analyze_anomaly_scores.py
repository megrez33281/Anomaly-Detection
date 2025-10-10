import torch
import numpy as np
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from Config import Config, set_seed
from UNet_Autoencoder_Model import UNetAutoencoder
from Evaluation import TextureDamage, CutPaste
import albumentations as A
from albumentations.pytorch import ToTensorV2

def analyze_scores_manually():
    """
    手動載入資料，不透過 DataLoader，以分析三組樣本的分數。
    """
    set_seed(Config.SEED)
    device = Config.DEVICE
    category = Config.TARGET_CLASS

    print(f"--- 手動分析類別: {category} ---")

    # 1. 載入模型和特徵庫
    model_path = Config.MODEL_SAVE_PATH.format(TARGET_CLASS=category)
    if category == '地毯': # 地毯是特例，使用 epoch 15
        model_path = os.path.join(Config.CHECKPOINT_DIR, 'model_地毯_epoch_15.pth')
    
    feature_library_path = f"{category}_features.pt"

    if not os.path.exists(model_path) or not os.path.exists(feature_library_path):
        print(f"錯誤：找不到模型或特徵庫.\n模型: {model_path}\n特徵庫: {feature_library_path}")
        return

    print(f"正在載入模型: {model_path}")
    model = UNetAutoencoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"正在載入特徵庫: {feature_library_path}")
    feature_library = torch.load(feature_library_path).to(device)

    # 2. 手動分割資料集，取得驗證集路徑
    class_mapping = Config.CLASS_FILENAME_MAPPING
    target_class_files = class_mapping.get(category)
    flat_train_dir = os.path.join(Config.DATA_DIR, "train")
    all_image_paths = [os.path.join(flat_train_dir, f"{file_id}.png") for file_id in target_class_files]
    _, val_paths = train_test_split(all_image_paths, test_size=0.2, random_state=Config.SEED)

    # 3. 定義轉換
    transform = A.Compose([
        A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2(),
    ])

    TEXTURE_CLASSES = ['地毯', '皮革', '木板', '磁磚', '鐵網']
    anomaly_transform = TextureDamage() if category in TEXTURE_CLASSES else CutPaste()

    # 4. 計算分數
    val_normal_scores = []
    val_pseudo_anomaly_scores = []
    real_anomaly_scores = []

    def get_score(image_tensor):
        feature = model.encode(image_tensor)
        distances = torch.sum((feature - feature_library)**2, dim=[1, 2, 3])
        return torch.min(distances).item()

    print("\n計算驗證集分數 (正常 vs 偽異常)...")
    with torch.no_grad():
        for img_path in tqdm(val_paths, desc="Processing Validation Set"):
            # 正常樣本
            original_image = np.array(Image.open(img_path).convert("RGB"))
            normal_tensor = transform(image=original_image)['image'].unsqueeze(0).to(device)
            val_normal_scores.append(get_score(normal_tensor))

            # 偽異常樣本
            pseudo_anomaly_pil = anomaly_transform(Image.fromarray(original_image))
            pseudo_anomaly_image = np.array(pseudo_anomaly_pil)
            pseudo_anomaly_tensor = transform(image=pseudo_anomaly_image)['image'].unsqueeze(0).to(device)
            val_pseudo_anomaly_scores.append(get_score(pseudo_anomaly_tensor))

    # 真實異常樣本
    if category == '地毯':
        real_anomaly_ids = ['640', '97', '1144']
    elif category == '牙刷':
        real_anomaly_ids = ['851', '423', '462']
    else:
        real_anomaly_ids = []

    if real_anomaly_ids:
        print("\n計算真實異常樣本分數...")
        with torch.no_grad():
            for img_id in tqdm(real_anomaly_ids, desc="Processing Real Anomalies"):
                img_path = os.path.join(Config.TEST_DIR, f"{img_id}.png")
                if not os.path.exists(img_path): continue
                image = np.array(Image.open(img_path).convert("RGB"))
                tensor_image = transform(image=image)['image'].unsqueeze(0).to(device)
                real_anomaly_scores.append(get_score(tensor_image))

    # 5. 產出統計報告
    def get_stats(scores, name):
        if not scores: return pd.Series([0, np.nan, np.nan, np.nan, np.nan], index=['count', 'mean', 'std', 'min', 'max'], name=name)
        s = pd.Series(scores)
        return s.describe()

    stats_df = pd.concat([
        get_stats(val_normal_scores, 'Normal (Validation Set)'),
        get_stats(val_pseudo_anomaly_scores, f'Pseudo Anomaly ({anomaly_transform.__class__.__name__})'),
        get_stats(real_anomaly_scores, 'Real Anomaly (Test Set)')
    ], axis=1)

    print("\n--- 分數統計分析 ---")
    print(stats_df.to_string())

if __name__ == "__main__":
    analyze_scores_manually()