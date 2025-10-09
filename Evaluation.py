
import random
import torch.nn.functional as F
import torch.nn as nn
import torch

# -----------------------------
# CutPaste製作異常資料
# -----------------------------
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
    """
    結合MSE與SSIM的損失函數
    ssim_weight控制兩者比例
    """
    def __init__(self, ssim_weight=0.84, window_size=11, channels=3):
        super(CombinedLoss, self).__init__()
        self.ssim_weight = ssim_weight
        self.mse_loss_avg = nn.MSELoss()
        self.mse_loss_none = nn.MSELoss(reduction='none')
        self.window_size = window_size
        self.channels = channels

        # 預先建立 SSIM window，避免重複生成
        window = self.create_window(window_size, channels)
        self.register_buffer('window', window)

    def create_window(self, window_size, channel):
        """建立Gaussian權重窗口"""
        # 設置滑動窗口的權重
        _1D_window = torch.exp(-(torch.arange(window_size) - window_size // 2)**2 / (2 * 1.5**2)).float()
        _2D_window = _1D_window.unsqueeze(1) @ _1D_window.unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, output, target, reduction='mean'):
        # output：模型輸出（重建的圖像）
        # target：真實圖像
        # output和target的維度皆為[batch, channel, H, W]，代表的是一個batch的圖片
        if reduction == 'mean':
            # 平均化損失
            mse = self.mse_loss_avg(output, target)
            #　此處ssim的設計允許一此處理一批圖片，在means時會選擇將最後的ssim結果取平均
            ssim_val = ssim(output, target, window=self.window, window_size=self.window_size, channel=self.channels, size_average=True)
            ssim_loss = 1 - ssim_val #這裡是一個純量
            # combined_loss = (1 - α) * MSE + α * (1 - SSIM)（以純量的形式）
            combined_loss = (1 - self.ssim_weight) * mse + self.ssim_weight * ssim_loss
            return combined_loss # 訓練時整體損失
        elif reduction == 'none':
            # 每張圖片分別計算
            mse_none = self.mse_loss_none(output, target).mean([1, 2, 3])
            ssim_none = ssim(output, target, window=self.window, window_size=self.window_size, channel=self.channels, size_average=False)
            # combined_loss = (1 - α) * MSE + α * (1 - SSIM)（以array的形式）
            ssim_loss_none = 1 - ssim_none # 這裡是一個向量
            return (1 - self.ssim_weight) * mse_none + self.ssim_weight * ssim_loss_none  # 每張圖的損失/異常分數
        else:
            raise ValueError(f"Invalid reduction type: {reduction}")