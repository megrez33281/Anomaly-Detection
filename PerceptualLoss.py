import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights

class VGGPerceptualLoss(nn.Module):
    """ 
    使用預訓練的VGG16模型計算感知損失。
    它計算輸入圖片和目標圖片在VGG網路中某幾個中間層的特徵圖之間的L1損失。
    """
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        # 加載在ImageNet上預訓練的VGG16模型
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        # 我們只需要VGG的特徵提取部分（卷積層）
        self.features = vgg.features
        self.features.eval()  # 設置為評估模式

        # 凍結VGG的所有參數，因為我們不訓練它
        for param in self.features.parameters():
            param.requires_grad = False

        # 定義用於計算損失的VGG層的索引
        # 這些層通常是每個卷積塊後的ReLU激活層
        self.loss_layers_indices = {
            '3': 'relu1_2',   # 第1個卷積塊的末尾
            '8': 'relu2_2',   # 第2個卷積塊的末尾
            '15': 'relu3_3',  # 第3個卷積塊的末尾
            '22': 'relu4_3'   # 第4個卷積塊的末尾
        }

        # VGG期望的輸入是標準化過的
        # 平均值和標準差是ImageNet數據集的統計值
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.resize = resize

    def forward(self, input, target):
        """
        計算感知損失。
        input: 重建圖像 (模型輸出)
        target: 原始圖像
        """
        # 如果輸入的範圍是[-1, 1]，先將其轉換回[0, 1]
        if input.min() < -0.1: # A simple check
            input = (input + 1) / 2
            target = (target + 1) / 2

        # 對輸入和目標進行VGG期望的標準化
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std

        # 如果需要，調整圖像大小以匹配VGG的最低輸入要求
        if self.resize:
            input = F.interpolate(input, size=(224, 224), mode='bilinear', align_corners=False)
            target = F.interpolate(target, size=(224, 224), mode='bilinear', align_corners=False)

        loss = 0.0
        x = input
        y = target

        # 遍歷VGG的每一層
        for name, layer in self.features._modules.items():
            x = layer(x)
            y = layer(y)
            # 如果當前層是我們定義的損失層之一，則計算L1損失
            if name in self.loss_layers_indices:
                loss += F.l1_loss(x, y)
        
        return loss
