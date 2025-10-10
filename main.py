
import torch
import torch.optim as optim
from tqdm import tqdm
from Config import set_seed, Config
from UNet_Autoencoder_Model import UNetAutoencoder
from Evaluation import *
from sklearn.metrics import roc_auc_score
from Dataset import GenerateDataset
import numpy as np

import os

# 設定整體隨機種子
set_seed(Config.SEED)

# ============================================================
# 主程式執行區域
# ============================================================
if __name__ == "__main__":
    # 獲取所有類別
    all_classes = list(Config.CLASS_FILENAME_MAPPING.keys())
    
    print(f"開始自動化訓練流程，共 {len(all_classes)} 個類別。")
    print("====================================================")

    # 遍歷所有類別並進行訓練
    for category in all_classes[::-1]:
        print(f"\nProcessing Category: [{category}]")
        print("----------------------------------------------------")
        
        # 動態設定當前要處理的類別
        Config.TARGET_CLASS = category

        # --- 以下為原始的單一類別訓練邏輯 ---
        
        print(f"Using device: {Config.DEVICE}")

        # 生成train dataset、validation dataset
        train_loader, val_loader = GenerateDataset()

        # --- 模型、優化器與損失函數 ---
        model = UNetAutoencoder().to(Config.DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
        criterion = CombinedLoss(ssim_weight=Config.SSIM_WEIGHT).to(Config.DEVICE)
        
        # --- 訓練與驗證流程 ---
        if Config.DEBUG_MODE:
            os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
            print(f"Debug mode enabled. Checkpoints will be saved to: {Config.CHECKPOINT_DIR}")
        else:
            best_val_loss = float('inf')
            save_path = Config.MODEL_SAVE_PATH.format(TARGET_CLASS=Config.TARGET_CLASS)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            print(f"Normal mode enabled. Best model will be saved to: {save_path}")

        print("Starting training...")

        for epoch in range(Config.EPOCHS):
            # --- 訓練階段 ---
            model.train()
            train_loss = 0.0
            for images, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Train]", leave=False):
                inputs = images.to(Config.DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, inputs, reduction='mean')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)

            # --- 驗證階段 ---
            model.eval()
            val_loss = 0.0
            val_scores = []
            val_labels = []
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Val]", leave=False):
                    inputs = images.to(Config.DEVICE)
                    outputs = model(inputs)
                    
                    batch_loss = criterion(outputs, inputs, reduction='mean')
                    val_loss += batch_loss.item()

                    loss_per_image = criterion(outputs, inputs, reduction='none').detach().cpu().numpy()
                    val_scores.extend(loss_per_image)
                    val_labels.extend(labels.numpy())
            
            avg_val_loss = val_loss / len(val_loader)
            val_auroc = roc_auc_score(np.asarray(val_labels), np.asarray(val_scores))
            
            print(f"Epoch [{epoch+1}/{Config.EPOCHS}], Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}, Val AUROC: {val_auroc:.4f}")

            # --- 根據模式儲存模型 ---
            if Config.DEBUG_MODE:
                checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"model_{Config.TARGET_CLASS}_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), checkpoint_path)
            else:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), save_path)
                    print(f"  -> New best validation loss: {best_val_loss:.4f}. Model saved to {save_path}")

        print(f"\nTraining complete for category: [{category}]")
        if not Config.DEBUG_MODE:
            print(f"Best validation loss achieved: {best_val_loss:.4f}")
        print("====================================================")

    print("All categories have been processed.")
