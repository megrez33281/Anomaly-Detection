from PIL import Image
import random
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import albumentations as A
from PerceptualLoss import VGGPerceptualLoss

# -----------------------------
# CutPaste製作異常資料
# -----------------------------
class CutPaste(object):
    """實作 CutPaste 增強：隨機剪下一塊貼回其他區域，模擬瑕疵圖片"""
    def __init__(self, patch_size_ratio_range=(0.05, 0.15)):
        self.patch_size_ratio_range = patch_size_ratio_range

    def __call__(self, image):
        img_h, img_w = image.size
        ratio = random.uniform(*self.patch_size_ratio_range)
        patch_h, patch_w = int(img_h * ratio), int(img_w * ratio)
        src_x = random.randint(0, img_w - patch_w)
        src_y = random.randint(0, img_h - patch_h)
        patch = image.crop((src_x, src_y, src_x + patch_w, src_y + patch_h))
        dest_x = random.randint(0, img_w - patch_w)
        dest_y = random.randint(0, img_h - patch_h)
        image.paste(patch, (dest_x, dest_y))
        return image

class TextureDamage(object):
    """實作紋理破壞增強：使用 CoarseDropout 在圖像上挖洞，模擬紋理瑕疵"""
    def __init__(self):
        self.transform = A.Compose([
            A.CoarseDropout(
                num_holes_range=(10, 20),
                hole_height_range=(0.05, 0.1),
                hole_width_range=(0.05, 0.1),
                fill=0,
                p=1.0
            )
        ])

    def __call__(self, image):
        np_image = np.array(image)
        augmented_image = self.transform(image=np_image)['image']
        return Image.fromarray(augmented_image)

# ============================================================
# New Combined Loss: L1 + Perceptual
# ============================================================
class CombinedLoss(nn.Module):
    """結合L1損失和VGG感知損失"""
    def __init__(self, perceptual_weight=1.0):
        super().__init__()
        self.perceptual_weight = perceptual_weight
        # 將感知損失模組實例化，並移到對應的device
        self.perceptual_loss = VGGPerceptualLoss().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def forward(self, recon, target, reduction='mean'):
        # L1 Loss (Pixel-level)
        l1_loss = F.l1_loss(recon, target, reduction='none')

        # Perceptual Loss (Feature-level)
        p_loss = self.perceptual_loss(recon, target)
        
        # p_loss 的輸出維度是 [batch_size]，需要擴展以匹配 l1_loss 的 [batch, C, H, W]
        p_loss = p_loss.view(-1, 1, 1, 1)

        # Combine losses
        total_loss = l1_loss + self.perceptual_weight * p_loss

        if reduction == 'mean':
            return total_loss.mean()
        elif reduction == 'none':
            # 為驗證返回每個圖像的分數
            return total_loss.view(total_loss.size(0), -1).mean(dim=1)
        else:
            raise ValueError(f"Unsupported reduction type: {reduction}")