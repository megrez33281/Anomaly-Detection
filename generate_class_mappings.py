import os
import json

def create_class_to_filenames_mapping():
    """
    掃描訓練資料夾，識別類別子資料夾，
    並建立一個將每個類別映射到其圖像檔名（整數形式）列表的字典。
    """
    # 定義好 train 資料夾的絕對路徑
    base_dir = "C:\\齊齊\\交大\\課程\\資料探勘\\作業\\作業二：Anomaly Detection"
    train_dir = os.path.join(base_dir, 'Dataset', 'Dataset', 'train')
    class_mapping = {}

    # 檢查 train 資料夾是否存在
    if not os.path.isdir(train_dir):
        print(f"錯誤：在 '{train_dir}' 找不到訓練資料夾")
        return None

    # 遍歷 train 資料夾中的每個項目
    for class_name in os.listdir(train_dir):
        class_dir = os.path.join(train_dir, class_name)
        
        # 檢查是否為資料夾
        if os.path.isdir(class_dir):
            filenames_int = []
            # 遍歷類別資料夾中的每個檔案
            for filename in os.listdir(class_dir):
                if filename.endswith('.png'):
                    # 從檔名中提取數字
                    try:
                        file_id = int(os.path.splitext(filename)[0])
                        filenames_int.append(file_id)
                    except ValueError:
                        # 處理檔名不是純數字的情況
                        print(f"警告：無法將 '{class_name}' 中的檔名 '{filename}' 解析為整數。已跳過。")
            
            # 對檔名ID列表進行排序以保持一致性
            filenames_int.sort()
            class_mapping[class_name] = filenames_int

    return class_mapping

if __name__ == "__main__":
    mapping = create_class_to_filenames_mapping()
    if mapping:
        output_filename = 'class_filename_mapping.json'
        # 將字典寫入 JSON 檔案，ensure_ascii=False 確保中文能正確顯示
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=4, ensure_ascii=False)
        
        print(f"成功生成對應關係，並已儲存至 '{output_filename}'")
        # 為了方便預覽，同時在終端機印出結果
        # print("\n預覽：")
        # print(json.dumps(mapping, indent=4, ensure_ascii=False))
