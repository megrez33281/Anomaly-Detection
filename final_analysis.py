import torch
import numpy as np
import os
import glob
from tqdm import tqdm
from PIL import Image
from sklearn.cluster import KMeans
import joblib
import argparse

from Config import Config, set_seed
from UNet_Autoencoder_Model import UNetAutoencoder
import albumentations as A
from albumentations.pytorch import ToTensorV2

def extract_features(epoch, model, device):
    """
    使用給定的模型為所有訓練樣本提取特徵。
    返回特徵和對應的路徑，全部保留在記憶體中。
    """
    print(f"--- 步驟 1: 為 Epoch {epoch} 提取特徵 ---")
    
    # 修正標準化參數以匹配模型訓練時的設定
    transform = A.Compose([
        A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])
    
    all_image_paths = sorted(glob.glob(os.path.join(Config.TRAIN_DIR, "*.png")), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    if not all_image_paths:
        print(f"警告：在 {Config.TRAIN_DIR} 中找不到任何圖片。")
        return None, None

    print(f"找到 {len(all_image_paths)} 張訓練圖片，開始提取...")

    all_features = []
    with torch.no_grad():
        for img_path in tqdm(all_image_paths, desc=f"Extracting features for Epoch {epoch}"):
            image = np.array(Image.open(img_path).convert("RGB"))
            tensor_image = transform(image=image)['image'].unsqueeze(0).to(device)
            features = model.encode(tensor_image)
            all_features.append(features.squeeze().cpu())
    
    stacked_features = torch.stack(all_features)
    print(f"特徵提取完成。 特徵庫維度: {stacked_features.shape}")
    return stacked_features, all_image_paths

def cluster_features(features, paths, epoch):
    """
    對記憶體中的特徵進行 K-Means 分群並儲存結果。
    """
    print("\n--- 步驟 2: 進行 K-Means 分群 (K=15) ---")
    
    num_features = features.shape[0]
    features_flat = features.view(num_features, -1).numpy()
    
    print(f"特徵已扁平化，維度: {features_flat.shape}")
    
    kmeans = KMeans(n_clusters=15, random_state=Config.SEED, n_init='auto')
    kmeans.fit(features_flat)
    
    print("K-Means 分群完成！")
    
    clustering_results = {
        'image_paths': paths,
        'cluster_labels': kmeans.labels_,
        'cluster_centers': kmeans.cluster_centers_,
    }
    
    output_filename = f"clustering_results_epoch_{epoch}.pt"
    output_path = os.path.join(Config.ROOT_DIR, output_filename)
    torch.save(clustering_results, output_path)
    
    kmeans_model_path = os.path.join(Config.ROOT_DIR, f"kmeans_model_epoch_{epoch}.joblib")
    joblib.dump(kmeans, kmeans_model_path)

    print("\n======================================================")
    print(f"成功！已將分群結果儲存至: {output_path}")
    print(f"K-Means 模型已儲存至: {kmeans_model_path}")
    print(f"- 共 {len(paths)} 個樣本")
    print(f"- 分成 {len(np.unique(kmeans.labels_))} 個群組")

def main():
    """主執行流程"""
    parser = argparse.ArgumentParser(description="Extract features and run clustering for a given epoch.")
    parser.add_argument("epoch", type=int, help="The epoch number of the model to use.")
    args = parser.parse_args()

    set_seed(Config.SEED)
    TARGET_EPOCH = args.epoch
    device = Config.DEVICE

    # 載入模型
    model_path = os.path.join(Config.CHECKPOINT_DIR, f'model_epoch_{TARGET_EPOCH}.pth')
    if not os.path.exists(model_path):
        print(f"錯誤：找不到模型 '{model_path}'。")
        return

    print(f"正在載入模型: {model_path}")
    # 腳本會從 UNet_Autoencoder_Model.py 導入當前的模型架構
    # 我們已確認該檔案是正確的 2-layer 版本
    model = UNetAutoencoder().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"載入模型權重時發生錯誤: {e}")
        print("請確認 UNet_Autoencoder_Model.py 中的架構與儲存的權重相符。 সন")
        return
        
    model.eval()

    # 步驟 1: 提取特徵
    features, paths = extract_features(TARGET_EPOCH, model, device)

    # 步驟 2: 分群
    if features is not None and paths is not None:
        cluster_features(features, paths, TARGET_EPOCH)

if __name__ == "__main__":
    main()