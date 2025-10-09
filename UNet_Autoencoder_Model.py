import torch
import torch.nn as nn

# ============================================================
# 模型架構：U-Net Autoencoder
# ============================================================
class UNetAutoencoder(nn.Module):
    """U-Net 自編碼器，用於影像重建"""
    def __init__(self):
        super(UNetAutoencoder, self).__init__()
        # Encoder 編碼部分
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Bottleneck 瓶頸層
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder 解碼部分（移除skip connection）
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(512, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(256, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(128, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(64, 64)
        
        # 最終輸出層，輸出重建圖
        self.out_conv = nn.Conv2d(64, 3, kernel_size=1)
        self.activation = nn.Sigmoid() 

    def conv_block(self, in_channels, out_channels):
        """基本卷積區塊 (Conv + ReLU)*2"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder：下採樣過程
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        e4 = self.enc4(p3)
        p4 = self.pool(e4)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Decoder：上採樣（無 skip connection）
        d4 = self.upconv4(b)
        # d4 = torch.cat((d4, e4), dim=1) # 移除跳接
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        # d3 = torch.cat((d3, e3), dim=1) # 移除跳接
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        # d2 = torch.cat((d2, e2), dim=1) # 移除跳接
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        # d1 = torch.cat((d1, e1), dim=1) # 移除跳接
        d1 = self.dec1(d1)
        
        # 輸出範圍 [0,1]
        output = torch.sigmoid(self.out_conv(d1))
        return output
