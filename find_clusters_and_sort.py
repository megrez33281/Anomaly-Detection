"""用手肘法找到最佳分群數，並"""
import torch
import numpy as np
import os
import glob
import shutil
from tqdm import tqdm
from PIL import Image
from sklearn.cluster import KMeans
import joblib
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from kneed import KneeLocator

from Config import Config, set_seed
from UNet_Autoencoder_Model import UNetAutoencoder

def find_optimal_clusters_and_sort(epoch):
    """
    載入模型，提取特徵，使用手肘法尋找最佳 K 值，
    然後進行分群並排序圖片。
    """
    set_seed(Config.SEED)
    device = Config.DEVICE

    print(f"為 Epoch {epoch} 尋找最佳群數並分群...")
    print("======================================================")

    # --- 1. 載入模型 ---
    model_path = os.path.join(Config.CHECKPOINT_DIR, f'model_epoch_{epoch}.pth')
    if not os.path.exists(model_path):
        print(f"錯誤：找不到模型 '{model_path}'。")
        return

    print(f"正在載入模型: {model_path}")
    model = UNetAutoencoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- 2. 準備影像轉換和路徑 ---
    transform = A.Compose([
        A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2(),
    ])
    
    all_image_paths = sorted(glob.glob(os.path.join(Config.TRAIN_DIR, "*.png")), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    if not all_image_paths:
        print(f"警告：在 {Config.TRAIN_DIR} 中找不到任何圖片。")
        return

    print(f"找到 {len(all_image_paths)} 張訓練圖片，開始提取特徵...")

    # --- 3. 提取特徵 ---
    all_features = []
    with torch.no_grad():
        for img_path in tqdm(all_image_paths, desc=f"Extracting features for Epoch {epoch}"):
            if not os.path.exists(img_path): continue
            image = np.array(Image.open(img_path).convert("RGB"))
            tensor_image = transform(image=image)['image'].unsqueeze(0).to(device)
            features = model.encode(tensor_image)
            all_features.append(features.squeeze().cpu())
    
    if not all_features:
        print("錯誤：未能提取任何特徵。")
        return
        
    stacked_features = torch.stack(all_features)
    num_features = stacked_features.shape[0]
    features_flat = stacked_features.view(num_features, -1).numpy()
    print(f"特徵提取與扁平化完成，維度: {features_flat.shape}")

    # --- 4. 使用手肘法尋找最佳 K 值 ---
    print("\n開始使用手肘法尋找最佳 K 值...")
    inertia_values = []
    K_range = range(12, 16)
    for k in tqdm(K_range, desc="Elbow Method"):
        kmeans = KMeans(n_clusters=k, random_state=Config.SEED, n_init='auto')
        kmeans.fit(features_flat)
        inertia_values.append(kmeans.inertia_)

    # 使用 kneed 找到拐點
    kn = KneeLocator(list(K_range), inertia_values, curve='convex', direction='decreasing')
    optimal_k = kn.knee
    if optimal_k is None:
        print("警告：未能自動找到最佳 K 值，將預設為 15。")
        optimal_k = 15
    else:
        print(f"手肘法找到的最佳 K 值為: {optimal_k}")

    # 繪製手肘法圖表
    plt.figure(figsize=(10, 6))
    plt.plot(list(K_range), inertia_values, 'bx-')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal K')
    plt.vlines(optimal_k, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', colors='r')
    elbow_plot_path = os.path.join(Config.ROOT_DIR, 'elbow_method.png')
    plt.savefig(elbow_plot_path)
    print(f"手肘法圖表已儲存至: {elbow_plot_path}")

    # --- 5. 使用最佳 K 值進行最終分群 ---
    print(f"\n使用 K={optimal_k} 進行最終分群...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=Config.SEED, n_init='auto')
    kmeans.fit(features_flat)
    print("最終分群完成！")

    # --- 6. 儲存分群結果 ---
    clustering_results = {
        'image_paths': all_image_paths,
        'cluster_labels': kmeans.labels_,
    }
    results_filename = f"clustering_results_epoch_{epoch}_k{optimal_k}.pt"
    results_path = os.path.join(Config.ROOT_DIR, results_filename)
    torch.save(clustering_results, results_path)
    print(f"分群結果已儲存至: {results_path}")

    # --- 7. 根據分群結果排序圖片 ---
    output_base_dir = os.path.join(Config.ROOT_DIR, 'classified_train')
    if os.path.exists(output_base_dir):
        print(f"警告：輸出資料夾 '{output_base_dir}' 已存在，將會被覆蓋。")
        shutil.rmtree(output_base_dir)
    os.makedirs(output_base_dir, exist_ok=True)
    print(f"已創建根輸出資料夾: {output_base_dir}")

    print("\n開始根據分群結果複製圖片...")
    for img_path, cluster_id in tqdm(zip(all_image_paths, kmeans.labels_), total=len(all_image_paths), desc="Sorting Images"):
        dest_folder_name = f"cluster_{cluster_id}"
        dest_dir = os.path.join(output_base_dir, dest_folder_name)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy(img_path, dest_dir)

    print("\n======================================================")
    print(f"成功！所有圖片已根據 {optimal_k} 個群組分類完畢")
    print(f"請查看 '{output_base_dir}' 資料夾")

if __name__ == "__main__":
    TARGET_EPOCH = 5
    find_optimal_clusters_and_sort(TARGET_EPOCH)
