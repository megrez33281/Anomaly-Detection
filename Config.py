import numpy as np
import os
import random
import torch


# ============================================================
# 設定與隨機性控制 (Configuration and Reproducibility)
# ============================================================
def set_seed(seed):
    """設定隨機種子，確保實驗可重現"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # 固定卷積結果
        torch.backends.cudnn.benchmark = False     # 禁用自動最佳化以保持一致性

class Config:
    """配置類別，儲存所有超參數與路徑"""
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 資料與模型儲存路徑
    ROOT_DIR = os.getcwd()
    DATA_DIR = os.path.join(ROOT_DIR, "Dataset", "Dataset")
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    TEST_DIR = os.path.join(DATA_DIR, "test")
    MODEL_SAVE_PATH = os.path.join(ROOT_DIR, "best_model_auroc.pth")
    
    # 訓練超參數
    IMG_SIZE = 256          # 圖片輸入大小
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    EPOCHS = 10             
    
    # 損失函數權重 (SSIM + MSE)
    SSIM_WEIGHT = 0.1