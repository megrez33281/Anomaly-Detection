import torch
import torch.nn as nn
import argparse
import os
from Config import Config

# ============================================================ 
# Model Architecture (Copied from UNet_Autoencoder_Model.py)
# This must match the architecture of the checkpoint being loaded.
# ============================================================ 
class UNetAutoencoder(nn.Module):
    """
    一個標準的深度自編碼器（移除了U-Net的跳接）。
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
        
        # Decoder (no skip connections)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(512, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(256, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(128, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(64, 64)
        
        # Output
        self.out_conv = nn.Conv2d(64, 3, kernel_size=1)
        self.activation = nn.Sigmoid()

    def conv_block(self, in_channels, out_channels):
        """(Conv + ReLU)*2"""
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
        d4 = self.dec4(d4)
        d3 = self.upconv3(d4)
        d3 = self.dec3(d3)
        d2 = self.upconv2(d3)
        d2 = self.dec2(d2)
        d1 = self.upconv1(d2)
        d1 = self.dec1(d1)
        
        # Output
        output = self.activation(self.out_conv(d1))
        return output

# ============================================================ 
# Analysis Function
# ============================================================ 
def analyze_model_weights(epoch):
    """Loads a model checkpoint and prints statistics for each parameter."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(Config.CHECKPOINT_DIR, f"model_epoch_{epoch}.pth")

    if not os.path.exists(model_path):
        print(f"Error: Checkpoint file not found at {model_path}")
        return

    print(f"--- Analyzing weights for {model_path} ---\n")

    # Instantiate the model and load the state dict
    model = UNetAutoencoder().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("Model loaded successfully.\n")
    except Exception as e:
        print(f"Error loading state_dict: {e}")
        print("Please ensure the model architecture in this script matches the checkpoint.")
        return

    # Iterate through parameters and print stats
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name}")
            print(f"  - Shape: {param.shape}")
            print(f"  - Mean:  {param.data.mean():.6f}")
            print(f"  - Std:   {param.data.std():.6f}")
            print(f"  - Min:   {param.data.min():.6f}")
            print(f"  - Max:   {param.data.max():.6f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze model weights for a given epoch.")
    parser.add_argument("epoch", type=int, help="The epoch number of the model to analyze.")
    args = parser.parse_args()

    analyze_model_weights(args.epoch)
