import torch
import os
import numpy as np
import pandas as pd
from collections import Counter
import argparse

from Config import Config

# 建立一個反向映射，從檔名數字 -> 類別名稱
FILENAME_TO_CLASS_MAPPING = {}
for class_name, filenames in Config.CLASS_FILENAME_MAPPING.items():
    for filename_num in filenames:
        FILENAME_TO_CLASS_MAPPING[str(filename_num)] = class_name

def get_true_class_from_path(path):
    """從影像路徑中提取真實的類別名稱。"""
    # 例如: C:\...\Dataset\train\1512.png -> 1512
    base_name = os.path.basename(path)
    file_num_str = os.path.splitext(base_name)[0]
    return FILENAME_TO_CLASS_MAPPING.get(file_num_str, "Unknown") # 從反向映射中查找

def generate_cluster_mappings(epoch):
    """
    分析 K-Means 的分群結果，驗證其純度，並生成
    「群組 -> 主要類別」的對應關係。
    """
    results_path = os.path.join(Config.ROOT_DIR, f'clustering_results_epoch_{epoch}.pt')
    
    if not os.path.exists(results_path):
        print(f"錯誤：找不到分群結果檔案 '{results_path}'。")
        return

    print(f"正在載入分群結果: {results_path}")
    cluster_data = torch.load(results_path, weights_only=False)
    image_paths = cluster_data['image_paths']
    cluster_labels = cluster_data['cluster_labels']

    # 建立一個 DataFrame 以便分析
    df = pd.DataFrame({
        'path': image_paths,
        'true_class': [get_true_class_from_path(p) for p in image_paths],
        'cluster': cluster_labels
    })

    print("\n--- 各群組組成分析 ---")
    cluster_to_class_mapping = {}
    all_cluster_compositions = {}

    for i in range(15):
        cluster_df = df[df['cluster'] == i]
        if cluster_df.empty:
            print(f"\nCluster {i}: 是空的")
            continue

        # 計算這個群組中，各真實類別的數量
        composition = Counter(cluster_df['true_class'])
        all_cluster_compositions[i] = composition
        
        # 找到數量最多的那個類別作為這個群組的代表
        most_common_class, count = composition.most_common(1)[0]
        total_in_cluster = len(cluster_df)
        purity = (count / total_in_cluster) * 100
        
        cluster_to_class_mapping[i] = most_common_class
        
        print(f"\nCluster {i} (共 {total_in_cluster} 個樣本) -> 主要類別: '{most_common_class}'")
        print(f"  - 純度: {purity:.2f}% ({count}/{total_in_cluster}) ")
        print("    - 詳細組成:")
        for cls, num in sorted(composition.items(), key=lambda item: item[1], reverse=True):
            print(f"        - {cls}: {num}")

    # 儲存這個對應關係
    mapping_path = os.path.join(Config.ROOT_DIR, 'cluster_to_class_mapping.pt')
    torch.save(cluster_to_class_mapping, mapping_path)
    print("\n======================================================")
    print(f"成功！已將 '群組->類別' 的對應關係儲存至: {mapping_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze K-Means clustering results for a given epoch.")
    parser.add_argument("epoch", type=int, help="The epoch number of the clustering results to analyze.")
    args = parser.parse_args()
    generate_cluster_mappings(args.epoch)