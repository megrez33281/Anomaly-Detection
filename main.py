
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
    model.init_weights()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    criterion = CombinedLoss(ssim_weight=Config.SSIM_WEIGHT).to(Config.DEVICE)


    # --- 訓練與驗證流程 ---
    if Config.DEBUG_MODE:
      os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
      print(f"Debug mode enabled. Checkpoints will be saved to: {Config.CHECKPOINT_DIR}")
    else:
      best_val_loss = float('inf')
      save_path = Config.MODEL_SAVE_PATH
      os.makedirs(os.path.dirname(save_path), exist_ok=True)
      print(f"Normal mode enabled. Best model will be saved to: {save_path}")
    # --- 訓練與驗證流程 ---
    print("Starting full training and validation pipeline...")
    best_auroc = 0.0

    for epoch in range(Config.EPOCHS):
        # --- 訓練階段 ---
        # 開啟訓練模式
        model.train()
        train_loss = 0.0
        # 遍歷訓練資料train_loader，每次取得一個batch(images)
        for images, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Train]"):
            inputs = images.to(Config.DEVICE) # 把影像搬到指定的Device（GPU或CPU）

            # 利用模型重建圖像
            outputs = model(inputs)
            # 呼叫定義的Loss Function計算Loss
            loss = criterion(outputs, inputs, reduction='mean')
            # 清空之前梯度
            optimizer.zero_grad()
            # 計算梯度
            loss.backward()
            # 更新權重
            optimizer.step()
            # 累加每個batch的loss
            train_loss += loss.item()
        # 計算epoch平均loss，用於觀察每個Epoch的表現
        avg_train_loss = train_loss / len(train_loader)

        # --- 驗證階段 ---
        model.eval() # 開啟驗證模式
        val_loss = 0.0
        val_scores = []
        val_labels = []

        with torch.no_grad(): # 不計算梯度，節省記憶體
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Val]"): # 遍歷驗證集val_loader
                inputs = images.to(Config.DEVICE) # 把影像搬到指定的Device（GPU或CPU）
                labels = labels.numpy()

                # 重建影像
                outputs = model(inputs)

                # 用設定好的Loss Function作為anomaly 分數
                # Change reduction to 'none' to get per-image scores
                batch_scores = criterion(outputs, inputs, reduction='none').cpu().numpy()
                val_loss += batch_scores.mean() # Use mean here only for tracking average batch loss

                # 如果還是 Tensor，就在這裡聚合成 per-image scalar
                if isinstance(batch_scores, torch.Tensor):
                    if batch_scores.ndim > 1:
                        batch_scores = batch_scores.view(batch_scores.size(0), -1).mean(dim=1)
                    batch_scores_np = batch_scores.detach().cpu().numpy()
                else:
                    # 已經是 NumPy 陣列的情況（理论上不该发生，但保险）
                    batch_scores_np = batch_scores

                val_scores.extend(batch_scores_np)
                val_labels.extend(labels)

        # --- 計算 AUROC ---
        # ----------------- DEBUG BLOCK START -----------------
        # 根據建議.txt，檢查 val_labels / val_scores 結構與統計
        val_scores_arr = np.asarray(val_scores).ravel() # 攤平成一維
        val_labels_arr = np.asarray(val_labels).ravel()

        print("\n----- DEBUG INFO FOR EPOCH {} -----".format(epoch + 1))
        print(f"DEBUG: len(scores)={len(val_scores_arr)}, len(labels)={len(val_labels_arr)}")
        print(f"DEBUG: labels unique and counts: {np.unique(val_labels_arr, return_counts=True)}")
        print(f"DEBUG: scores min/mean/median/max: {val_scores_arr.min():.4f}/{val_scores_arr.mean():.4f}/{np.median(val_scores_arr):.4f}/{val_scores_arr.max():.4f}")

        # 分別計算 class 0 和 class 1 的平均分數
        for lbl in np.unique(val_labels_arr):
            mask = (val_labels_arr == lbl)
            if mask.sum() > 0:
                print(f"  -> class {lbl}: count={mask.sum()}, mean_score={val_scores_arr[mask].mean():.6f}, std={val_scores_arr[mask].std():.6f}")

        # 檢查分數是否可能反轉 (正常樣本分數反而更高)
        val_auroc = roc_auc_score(val_labels_arr, val_scores_arr)
        avg_val_loss = val_loss / len(val_loader)
        print(f"ROC AUC (as-is): {val_auroc:.4f}")
        if val_auroc < 0.5:
            au_inv = roc_auc_score(val_labels_arr, -val_scores_arr)
            print(f"ROC AUC (inverted scores): {au_inv:.4f}")
            if au_inv > val_auroc:
                print("NOTE: Scores might be inverted. Higher score may mean more normal.")
        print("-------------------------------------\n")
        # ----------------- DEBUG BLOCK END -----------------

        print(f"Epoch [{epoch+1}/{Config.EPOCHS}], Avg Train Loss: {avg_train_loss:.4f}, Validation AUROC: {val_auroc:.4f}")

        # --- 根據模式儲存模型 ---
        if Config.DEBUG_MODE:
          checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"model_epoch_{epoch+1}.pth")
          torch.save(model.state_dict(), checkpoint_path)
        else:
          if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best validation loss: {best_val_loss:.4f}. Model saved to {save_path}")

    print("\nTraining and validation complete.")
    print(f"Best validation AUROC achieved: {best_auroc:.4f}")