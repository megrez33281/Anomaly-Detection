
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
        best_auroc = 0.0
        print(f"Normal mode enabled. Best model will be saved to: {Config.MODEL_SAVE_PATH}")

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        # --- 訓練階段 ---
        model.train()
        train_loss = 0.0
        for images, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Train]"):
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
        val_scores = []
        val_labels = []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Val]"):
                inputs = images.to(Config.DEVICE)
                outputs = model(inputs)
                loss_per_image = criterion(outputs, inputs, reduction='none').detach().cpu().numpy()
                val_scores.extend(loss_per_image)
                val_labels.extend(labels.numpy())
        
        val_auroc = roc_auc_score(np.asarray(val_labels), np.asarray(val_scores))
        print(f"Epoch [{epoch+1}/{Config.EPOCHS}], Avg Train Loss: {avg_train_loss:.4f}, Validation AUROC: {val_auroc:.4f}")

        # --- 根據模式儲存模型 ---
        if Config.DEBUG_MODE:
            # 偵錯模式：儲存所有 epoch 的權重
            checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"model_{Config.TARGET_CLASS}_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> [Debug Mode] Checkpoint saved to {checkpoint_path}")
        else:
            # 正常模式：只儲存最佳 AUROC 的權重
            if val_auroc > best_auroc:
                best_auroc = val_auroc
                torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
                print(f"  -> New best AUROC: {best_auroc:.4f}. Model saved to {Config.MODEL_SAVE_PATH}")

    print("\nTraining complete.")
    if not Config.DEBUG_MODE:
        print(f"Best validation AUROC achieved: {best_auroc:.4f}")
