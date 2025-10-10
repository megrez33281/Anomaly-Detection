from PIL import Image
import random
import torch.nn.functional as F
import torch.nn as nn
import torch

# -----------------------------
# CutPaste製作異常資料
# -----------------------------
import numpy as np
import albumentations as A

class CutPaste(object):
    """實作 CutPaste 增強：隨機剪下一塊貼回其他區域，模擬瑕疵圖片"""
    def __init__(self, patch_size_ratio_range=(0.05, 0.15)):
        self.patch_size_ratio_range = patch_size_ratio_range

    def __call__(self, image):
        img_h, img_w = image.size
        
        # 隨機選取貼片比例
        ratio = random.uniform(*self.patch_size_ratio_range)
        patch_h, patch_w = int(img_h * ratio), int(img_w * ratio)
        
        # 決定貼片來源位置
        src_x = random.randint(0, img_w - patch_w)
        src_y = random.randint(0, img_h - patch_h)
        
        # 剪下貼片
        patch = image.crop((src_x, src_y, src_x + patch_w, src_y + patch_h))
        
        # 決定貼片目的地位置
        dest_x = random.randint(0, img_w - patch_w)
        dest_y = random.randint(0, img_h - patch_h)
        
        # 將貼片貼回原圖（造成異常）
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
        # 將 PIL Image 轉換為 numpy array 以便 albumentations 處理
        np_image = np.array(image)
        # 套用 CoarseDropout
        augmented_image = self.transform(image=np_image)['image']
        # 將處理後的 numpy array 轉回 PIL Image
        return Image.fromarray(augmented_image)

    
# ============================================================
# 結構相似度 SSIM + 損失函數組合
# ============================================================
def ssim(img1, img2, window, window_size, channel, size_average=True):
    """計算兩張影像間的SSIM(結構相似度)"""
    C1 = 0.01**2
    C2 = 0.03**2
    
    #F.conv2d 可以一次對多張圖做卷積運算
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel) # μ_x
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel) # μ_y

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq # σ_x^2
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq # σ_y^2
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2 # σ_xy

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))/((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        # 計算整批圖片的平均ssim
        return ssim_map.mean()
    else:
        # 每張圖各自的SSIM值
        return ssim_map.mean([1, 2, 3])

class CombinedLoss(nn.Module):
    def __init__(self, ssim_weight=0.5, window_size=11):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.ssim_weight = ssim_weight
        self.window_size = window_size
        self.window = None  # 之後在 forward 建立或搬到裝置

    def create_window(self, window_size, channel, device):
        """建立Gaussian權重窗口"""
        # 設置滑動窗口的權重
        _1D_window = torch.exp(-(torch.arange(window_size) - window_size // 2)**2 / (2 * 1.5**2)).float()
        _2D_window = _1D_window.unsqueeze(1) @ _1D_window.unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window.to(device)

    def forward(self, recon, target, reduction='mean'):
        # output：模型輸出（重建的圖像）
        # target：真實圖像
        # output和target的維度皆為[batch, channel, H, W]，代表的是一個batch的圖片
        mse_loss = self.mse(recon, target)

        # 建立window (第一次 forward 或裝置不同時)
        if (self.window is None) or (self.window.device != recon.device) or (self.window.size(0) != recon.size(1)):
            self.window = self.create_window(self.window_size, recon.size(1), recon.device)

        ssim_val = ssim(recon, target, window=self.window, window_size=self.window_size, channel=recon.size(1), size_average=False)
        # ssim_loss 越大代表越不相似
        ssim_loss = 1 - ssim_val

        # broadcast ssim_loss to match shape
        ssim_loss = ssim_loss.view(-1, 1, 1, 1)
        total_loss = (1 - self.ssim_weight) * mse_loss + self.ssim_weight * ssim_loss

        if reduction == 'mean':
            return total_loss.mean()
        elif reduction == 'none':
            # 每張圖一個 loss
            return total_loss.view(total_loss.size(0), -1).mean(dim=1)
        else:
            raise ValueError("Unsupported reduction type")
