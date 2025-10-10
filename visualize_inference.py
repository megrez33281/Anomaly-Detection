import torch
from PIL import Image
import numpy as np
import os

from Config import Config, set_seed
from UNet_Autoencoder_Model import UNetAutoencoder
import albumentations as A
from albumentations.pytorch import ToTensorV2

def visualize_specific_images():
    """對指定的測試圖片進行推論，並生成原圖與重建圖的對比影像。"""
    set_seed(Config.SEED)
    device = Config.DEVICE
    
    # 1. 建立輸出資料夾
    output_dir = "inference_results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"對比圖片將儲存於 '{output_dir}' 資料夾")

    # 2. 載入模型
    model = UNetAutoencoder().to(device)
    model_path = Config.MODEL_SAVE_PATH
    if not os.path.exists(model_path):
        print(f"錯誤：在 '{model_path}' 找不到模型。")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"模型已從 '{model_path}' 載入。")

    # 3. 定義影像轉換
    transform = A.Compose([
        A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2(),
    ])

    # 4. 準備要測試的圖片列表
    image_ids = ['1512', '1144', '832', '640', '1700', '97']
    test_dir = Config.TEST_DIR
    image_paths = [os.path.join(test_dir, f"{id}.png") for id in image_ids]

    # 5. 執行推論並生成視覺化結果
    print("\n開始生成對比圖片...")
    with torch.no_grad():
        for img_path in image_paths:
            image_name = os.path.basename(img_path)
            if not os.path.exists(img_path):
                print(f"警告：找不到圖片 '{img_path}'，已跳過。")
                continue

            # 載入原始 PIL 圖片用於視覺化
            original_pil = Image.open(img_path).convert("RGB").resize((Config.IMG_SIZE, Config.IMG_SIZE))
            
            # 準備輸入模型的 Tensor
            image_np = np.array(original_pil)
            tensor_image = transform(image=image_np)['image'].unsqueeze(0).to(device)
            
            # 模型重建
            recon_tensor = model(tensor_image).squeeze(0)
            
            # 將輸出的 Tensor 轉回 PIL 圖片以便儲存
            recon_np = recon_tensor.cpu().numpy()
            recon_np = np.transpose(recon_np, (1, 2, 0)) # 從 (C, H, W) 轉為 (H, W, C)
            # 反正規化並轉換型態
            recon_np = np.clip(recon_np, 0, 1)
            recon_np = (recon_np * 255).astype(np.uint8)
            recon_pil = Image.fromarray(recon_np)

            # 建立並排的對比圖
            comparison_img = Image.new('RGB', (Config.IMG_SIZE * 2, Config.IMG_SIZE))
            comparison_img.paste(original_pil, (0, 0))
            comparison_img.paste(recon_pil, (Config.IMG_SIZE, 0))
            
            # 儲存對比圖
            save_path = os.path.join(output_dir, f"compare_{os.path.splitext(image_name)[0]}.png")
            comparison_img.save(save_path)
            print(f"  - 已儲存 {image_name} 的對比圖至 {save_path}")

if __name__ == "__main__":
    visualize_specific_images()
