import torch
import torch.nn as nn

# ============================================================
# 模型架構：U-Net Autoencoder
# ============================================================
class UNetAutoencoder(nn.Module):
    """
    一個超淺的、帶有部分跳接的深度自編碼器 (Extra-Shallow Reduced U-Net)。
    下採樣2次，只保留最淺層的跳接。
    """
    def __init__(self):
        super(UNetAutoencoder, self).__init__()
        # Encoder
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = self.conv_block(128, 256)
        
        # Decoder
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(128, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)   # 輸入是 upconv1(64) + e1(64) = 128
        
        # Output
        self.out_conv = nn.Conv2d(64, 3, kernel_size=1)
        self.activation = nn.Tanh()

    def conv_block(self, in_channels, out_channels):
        """(Conv + ReLU)*2"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def init_weights(self):
        """Kaiming Normal initialization as recommended."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        
        # Bottleneck
        b = self.bottleneck(p2)
        
        # Decoder
        d2 = self.upconv2(b)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        # Only skip connection at the shallowest layer
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)
        
        # Output
        output = self.activation(self.out_conv(d1))
        return output

    def encode(self, x):
        """僅使用編碼器部分，提取影像特徵"""
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        b = self.bottleneck(p2)
        return b
