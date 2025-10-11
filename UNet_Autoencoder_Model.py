import torch
import torch.nn as nn

class UNetAutoencoder(nn.Module):
    """
    一個4層深度的Autoencoder，但只帶有最淺層的跳接 (Reduced U-Net)。
    """
    def __init__(self):
        super(UNetAutoencoder, self).__init__()
        # Encoder
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(512, 512) # No skip
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(256, 256) # No skip
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(128, 128) # No skip
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # 沒有跳接就填64
        self.dec1 = self.conv_block(64, 64)    # Skip connection: 64 (upconv) + 64 (e1)
        
        # Output
        self.out_conv = nn.Conv2d(64, 3, kernel_size=1)
        self.activation = nn.Tanh()

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
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
        
        # Decoder
        d4 = self.upconv4(b)
        d4 = self.dec4(d4) # No skip
        
        d3 = self.upconv3(d4)
        d3 = self.dec3(d3) # No skip
        
        d2 = self.upconv2(d3)
        d2 = self.dec2(d2) # No skip
        
        d1 = self.upconv1(d2)
        # d1 = torch.cat((d1, e1), dim=1) # The only skip connection
        d1 = self.dec1(d1)
        
        output = self.activation(self.out_conv(d1))
        return output

    def encode(self, x):
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        e4 = self.enc4(p3)
        p4 = self.pool(e4)
        b = self.bottleneck(p4)
        return b
