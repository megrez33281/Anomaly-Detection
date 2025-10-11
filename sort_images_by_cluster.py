import torch
import os
import shutil
from tqdm import tqdm

from Config import Config

def sort_images_by_cluster():
    """
    根據分群結果，將訓練圖片複製到對應 cluster ID 的資料夾中。
    """
    TARGET_EPOCH = 5 # 我們使用第5個epoch的結果
    
    results_path = os.path.join(Config.ROOT_DIR, f'clustering_results_epoch_{TARGET_EPOCH}.pt')
    output_base_dir = os.path.join(Config.ROOT_DIR, 'classified_train')

    # --- 檢查檔案是否存在 ---
    if not os.path.exists(results_path):
        print(f"錯誤：找不到分群結果檔案 '{results_path}'。")
        return

    print("正在載入檔案...")
    cluster_data = torch.load(results_path, weights_only=False)

    image_paths = cluster_data['image_paths']
    cluster_labels = cluster_data['cluster_labels']

    # --- 創建輸出資料夾 ---
    if os.path.exists(output_base_dir):
        print(f"警告：輸出資料夾 '{output_base_dir}' 已存在，將會被覆蓋。")
        shutil.rmtree(output_base_dir)
    os.makedirs(output_base_dir, exist_ok=True)
    print(f"已創建根輸出資料夾: {output_base_dir}")

    # --- 複製檔案 ---
    print("\n開始根據分群結果複製圖片...")
    for img_path, cluster_id in tqdm(zip(image_paths, cluster_labels), total=len(image_paths), desc="Sorting Images"):
        # 使用 cluster ID 作為資料夾名稱
        dest_folder_name = f"cluster_{cluster_id}"
        dest_dir = os.path.join(output_base_dir, dest_folder_name)
        os.makedirs(dest_dir, exist_ok=True)
        
        # 執行複製
        shutil.copy(img_path, dest_dir)

    print("\n======================================================")
    print("成功！所有圖片已根據分群結果分類完畢")
    print(f"請查看 '{output_base_dir}' 資料夾")

if __name__ == "__main__":
    sort_images_by_cluster()
