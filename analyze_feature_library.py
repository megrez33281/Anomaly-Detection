import torch
import numpy as np
import os
from sklearn.cluster import KMeans
import joblib

from Config import Config, set_seed

def analyze_features(epoch):
    """
    載入特徵庫，使用 K-Means 進行分群，並儲存分群結果。
    """
    set_seed(Config.SEED)
    
    # 1. 載入特徵庫
    feature_library_path = os.path.join(Config.ROOT_DIR, f'feature_library_epoch_{epoch}.pt')
    if not os.path.exists(feature_library_path):
        print(f"錯誤：找不到特徵庫檔案 '{feature_library_path}' ويعمل.")
        return

    print(f"正在載入特徵庫: {feature_library_path}")
    feature_data = torch.load(feature_library_path)
    features = feature_data['features']
    paths = feature_data['paths']
    
    # 2. 準備特徵以進行分群
    # 將 (N, C, H, W) 的特徵圖扁平化為 (N, C*H*W) 的向量
    num_features = features.shape[0]
    features_flat = features.view(num_features, -1).numpy()
    
    print(f"特徵已載入並扁平化，維度: {features_flat.shape}")
    print("開始進行 K-Means 分群 (K=15)...")
    
    # 3. 執行 K-Means
    # n_init='auto' 是 scikit-learn 1.4.0 之後的建議值
    kmeans = KMeans(n_clusters=15, random_state=Config.SEED, n_init='auto')
    kmeans.fit(features_flat)
    
    print("K-Means 分群完成！")
    
    # 4. 準備並儲存結果
    clustering_results = {
        'image_paths': paths,
        'cluster_labels': kmeans.labels_, # 每個圖片對應的群組標籤
        'cluster_centers': kmeans.cluster_centers_, # 15個群組的中心點
        'kmeans_model': kmeans # 儲存整個模型以備不時之需
    }
    
    output_filename = f"clustering_results_epoch_{epoch}.pt"
    output_path = os.path.join(Config.ROOT_DIR, output_filename)
    torch.save(clustering_results, output_path)
    
    # 為了方便，也用 joblib 儲存一份 kmeans 模型
    kmeans_model_path = os.path.join(Config.ROOT_DIR, f"kmeans_model_epoch_{epoch}.joblib")
    joblib.dump(kmeans, kmeans_model_path)

    print("\n======================================================")
    print(f"成功！已將分群結果儲存至: {output_path}")
    print(f"K-Means 模型已儲存至: {kmeans_model_path}")
    print(f"- 共 {len(paths)} 個樣本")
    print(f"- 分成 {len(np.unique(kmeans.labels_))} 個群組")

if __name__ == "__main__":
    TARGET_EPOCH = 1
    analyze_features(TARGET_EPOCH)
