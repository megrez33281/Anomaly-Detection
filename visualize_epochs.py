import torch
from PIL import Image
import numpy as np
import os
import glob

from Config import Config, set_seed
from UNet_Autoencoder_Model import UNetAutoencoder
import albumentations as A
from albumentations.pytorch import ToTensorV2

def visualize_epoch_reconstruction():
    """
    遍歷所有儲存的 epoch 權重，對指定的測試圖片進行推論，
    並將每個 epoch 的重建結果儲存到對應的資料夾中。
    """
    set_seed(Config.SEED)
    device = Config.DEVICE
    
    # 1. 找到所有 epoch 權重檔案
    checkpoint_dir = Config.CHECKPOINT_DIR
    # 排序權重檔案，確保按 epoch 1, 2, 3... 的順序處理
    checkpoints = sorted(
        glob.glob(os.path.join(checkpoint_dir, "*.pth")), 
        key=lambda x: int(os.path.splitext(os.path.basename(x).split('_')[-1])[0])
    )
    
    if not checkpoints:
        print(f"錯誤：在 '{checkpoint_dir}' 中找不到任何權重檔案。")
        return

    # 2. 準備影像轉換和圖片路徑
    transform = A.Compose([
        A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2(),
    ])
    image_ids = ['998', '432', '1540']
    test_dir = Config.TEST_DIR
    image_paths = [os.path.join(test_dir, f"{id}.png") for id in image_ids]

    print("開始遍歷所有 Epoch 權重並生成重建對比圖...")
    # 3. 遍歷每一個權重檔案
    for model_path in checkpoints:
        try:
            epoch_num = int(os.path.splitext(os.path.basename(model_path).split('_')[-1])[0])
        except (IndexError, ValueError):
            print(f"警告：無法從檔名 '{model_path}' 中解析 epoch 序號，已跳過。")
            continue

        print(f"--- 正在處理 Epoch {epoch_num} ---")

        # 為每個 epoch 建立獨立的輸出資料夾
        output_dir = os.path.join("inference_results", f"epoch_{epoch_num}")
        os.makedirs(output_dir, exist_ok=True)

        # 載入當前 epoch 的模型
        model = UNetAutoencoder().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # 對指定的6張圖片進行推論並儲存結果
        with torch.no_grad():
            for img_path in image_paths:
                image_name = os.path.basename(img_path)
                original_pil = Image.open(img_path).convert("RGB").resize((Config.IMG_SIZE, Config.IMG_SIZE))
                image_np = np.array(original_pil)
                tensor_image = transform(image=image_np)['image'].unsqueeze(0).to(device)
                
                recon_tensor = model(tensor_image).squeeze(0)
                
                recon_np = recon_tensor.cpu().numpy()
                recon_np = np.transpose(recon_np, (1, 2, 0))
                recon_np = np.clip(recon_np, 0, 1)
                recon_np = (recon_np * 255).astype(np.uint8)
                recon_pil = Image.fromarray(recon_np)

                comparison_img = Image.new('RGB', (Config.IMG_SIZE * 2, Config.IMG_SIZE))
                comparison_img.paste(original_pil, (0, 0))
                comparison_img.paste(recon_pil, (Config.IMG_SIZE, 0))
                
                save_path = os.path.join(output_dir, f"compare_{os.path.splitext(image_name)[0]}.png")
                comparison_img.save(save_path)
        print(f"  -> Epoch {epoch_num} 的所有對比圖已儲存完畢。")

if __name__ == "__main__":
    visualize_epoch_reconstruction()