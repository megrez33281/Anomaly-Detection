from PIL import Image
import random
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import albumentations as A

# -----------------------------
# CutPaste Augmentation
# -----------------------------
class CutPaste(object):
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

# ============================================================
# SSIM + L1 Combined Loss
# ============================================================
def ssim(img1, img2, window, window_size, channel, size_average=True):
    C1 = 0.01**2
    C2 = 0.03**2
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean([1, 2, 3])

class CombinedLoss(nn.Module):
    def __init__(self, ssim_weight):
        super().__init__()
        self.ssim_weight = ssim_weight
        self.window_size = 11
        self.window = None

    def create_window(self, window_size, channel, device):
        _1D_window = torch.exp(-(torch.arange(window_size) - window_size // 2)**2 / (2 * 1.5**2)).float()
        _2D_window = _1D_window.unsqueeze(1) @ _1D_window.unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window.to(device)

    def forward(self, recon, target, reduction='mean'):
        l1_loss = F.l1_loss(recon, target, reduction='none')

        if self.ssim_weight > 0:
            if (self.window is None) or (self.window.device != recon.device) or (self.window.size(0) != recon.size(1)):
                self.window = self.create_window(self.window_size, recon.size(1), recon.device)
            
            ssim_val = ssim(recon, target, window=self.window, window_size=self.window_size, channel=recon.size(1), size_average=False)
            ssim_loss = 1 - ssim_val
            ssim_loss = ssim_loss.view(-1, 1, 1, 1)
            total_loss = (1 - self.ssim_weight) * l1_loss + self.ssim_weight * ssim_loss
        else:
            total_loss = l1_loss

        if reduction == 'mean':
            return total_loss.mean()
        elif reduction == 'none':
            return total_loss.view(total_loss.size(0), -1).mean(dim=1)
        else:
            raise ValueError(f"Unsupported reduction type: {reduction}")
