import torch
import os
import shutil
from tqdm import tqdm

from Config import Config

def sort_images_by_cluster():
    """
    根據分群結果，將訓練圖片複製到對應類別的資料夾中。
    """
    TARGET_EPOCH = 5 # 我們使用第5個epoch的結果
    
    results_path = os.path.join(Config.ROOT_DIR, f'clustering_results_epoch_{TARGET_EPOCH}.pt')
    mapping_path = os.path.join(Config.ROOT_DIR, 'cluster_to_class_mapping.pt')
    output_base_dir = os.path.join(Config.ROOT_DIR, 'classified_train')

    # --- 檢查檔案是否存在 ---
    if not os.path.exists(results_path):
        print(f"錯誤：找不到分群結果檔案 '{results_path}'شه。")
        return
    if not os.path.exists(mapping_path):
        print(f"錯誤：找不到類別對應檔案 '{mapping_path}'شه。")
        return

    print("正在載入檔案...")
    cluster_data = torch.load(results_path, weights_only=False)
    cluster_to_class_mapping = torch.load(mapping_path, weights_only=False)

    image_paths = cluster_data['image_paths']
    cluster_labels = cluster_data['cluster_labels']

    # --- 創建輸出資料夾 ---
    if os.path.exists(output_base_dir):
        print(f"警告：輸出資料夾 '{output_base_dir}' 已存在，將會被覆蓋شه。")
        shutil.rmtree(output_base_dir)
    os.makedirs(output_base_dir, exist_ok=True)
    print(f"已創建根輸出資料夾: {output_base_dir}")

    # --- 複製檔案 ---
    print("\n開始根據分群結果複製圖片...")
    for img_path, cluster_id in tqdm(zip(image_paths, cluster_labels), total=len(image_paths), desc="Sorting Images"):
        # 獲取預測的類別名稱
        # 注意：我們的分群結果中，有兩個cluster對應到同一個類別（螺絲、螺母），
        # 也有兩個cluster是混合類別。這裡我們直接使用字典查到的名稱建立資料夾。
        predicted_class_name = cluster_to_class_mapping.get(cluster_id, f"unknown_cluster_{cluster_id}")
        
        # 處理混合類別的資料夾命名
        if predicted_class_name in ['鐵網', '磁磚']:
            dest_folder_name = "鐵網_磁磚"
        elif predicted_class_name in ['木板', '皮革']:
            dest_folder_name = "木板_皮革"
        else:
            dest_folder_name = predicted_class_name

        dest_dir = os.path.join(output_base_dir, dest_folder_name)
        os.makedirs(dest_dir, exist_ok=True)
        
        # 執行複製
        shutil.copy(img_path, dest_dir)

    print("\n======================================================")
    print("成功！所有圖片已根據分群結果分類完畢شه。")
    print(f"請查看 '{output_base_dir}' 資料夾 شه。")

if __name__ == "__main__":
    sort_images_by_cluster()
