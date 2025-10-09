
import torch
import torch.optim as optim
from tqdm import tqdm
from Config import set_seed, Config
from UNet_Autoencoder_Model import UNetAutoencoder
from Evaluation import *
from sklearn.metrics import roc_auc_score
from Dataset import GenerateDataset

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
        val_scores = []
        val_labels = []

        with torch.no_grad(): # 不計算梯度，節省記憶體
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Val]"): # 遍歷驗證集val_loader
                inputs = images.to(Config.DEVICE) # 把影像搬到指定的Device（GPU或CPU）
                labels = labels.numpy()

                # 重建影像
                outputs = model(inputs)

                # 用設定好的Loss Function作為anomaly 分數
                batch_scores = criterion(outputs, inputs, reduction='none').cpu().numpy()
                
                # 把這一batch的每張圖片的anomaly 分數以及該圖片的真實類別val_scores, val_labels
                val_scores.extend(batch_scores)
                val_labels.extend(labels)

        # --- 計算 AUROC ---
        val_auroc = roc_auc_score(val_labels, val_scores) # 模型在不同threshold下的表現
        print(f"Epoch [{epoch+1}/{Config.EPOCHS}], Avg Train Loss: {avg_train_loss:.4f}, Validation AUROC: {val_auroc:.4f}")

        # 若AUROC提升（面積越大越好）則儲存模型
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  -> New best AUROC: {best_auroc:.4f}. Model saved to {Config.MODEL_SAVE_PATH}")

    print("\nTraining and validation complete.")
    print(f"Best validation AUROC achieved: {best_auroc:.4f}")
