
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
import torch.nn.functional as F
from tqdm import tqdm

# 1. Configuration and Reproducibility
def set_seed(seed):
    """Fix random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class Config:
    """Configuration class for all hyperparameters and paths."""
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paths
    ROOT_DIR = "C:/齊齊/交大/課程/資料探勘/作業/作業二：Anomaly Detection"
    DATA_DIR = os.path.join(ROOT_DIR, "Dataset", "Dataset")
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    TEST_DIR = os.path.join(DATA_DIR, "test")
    
    # Training parameters from the plan
    IMG_SIZE = 256 # Start with a single size first
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    EPOCHS = 30 # Start with 1 epoch for debugging
    
    # Model and Loss
    SSIM_WEIGHT = 0.84

# Set seed for the entire script
set_seed(Config.SEED)

# 2. Dataset and DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 2. Dataset, Augmentations, and DataLoader

class CutPaste(object):
    """Applies the CutPaste augmentation."""
    def __init__(self, patch_size_ratio_range=(0.05, 0.15)):
        self.patch_size_ratio_range = patch_size_ratio_range

    def __call__(self, image):
        img_h, img_w = image.size
        
        # Define patch size
        ratio = random.uniform(*self.patch_size_ratio_range)
        patch_h, patch_w = int(img_h * ratio), int(img_w * ratio)
        
        # Define patch source coordinates
        src_x = random.randint(0, img_w - patch_w)
        src_y = random.randint(0, img_h - patch_h)
        
        # Cut the patch
        patch = image.crop((src_x, src_y, src_x + patch_w, src_y + patch_h))
        
        # Define paste destination coordinates
        dest_x = random.randint(0, img_w - patch_w)
        dest_y = random.randint(0, img_h - patch_h)
        
        # Paste the patch
        image.paste(patch, (dest_x, dest_y))
        
        return image

class AnomalyDataset(Dataset):
    """Custom Dataset for loading anomaly detection data."""
    def __init__(self, image_paths, transform=None, label_transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.label_transform = label_transform # For creating anomalous images

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        label = 0
        if self.label_transform:
            image = self.label_transform(image)
            label = 1

        if self.transform:
            image = self.transform(image)
            
        return image, label


# 3. Model Architecture (U-Net Autoencoder)
class UNetAutoencoder(nn.Module):
    """U-Net like Autoencoder for image reconstruction."""
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
        self.dec4 = self.conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # Output layer
        self.out_conv = nn.Conv2d(64, 3, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder path
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
        
        # Decoder path with skip connections
        d4 = self.upconv4(b)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)
        
        # Output
        output = torch.sigmoid(self.out_conv(d1)) # Use sigmoid for output in [0, 1] range
        return output

# 3.5. Loss Function
def ssim(img1, img2, window_size=11, size_average=True):
    """Computes the Structural Similarity Index (SSIM) between two images."""
    C1 = 0.01**2
    C2 = 0.03**2

    # Create a 2D Gaussian window
    window = torch.exp(-(torch.arange(window_size) - window_size // 2)**2 / (2 * 1.5**2)).unsqueeze(1)
    window = window.mm(window.t()).float().unsqueeze(0).unsqueeze(0)
    window = window.repeat(img1.size(1), 1, 1, 1)
    window = window.to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.size(1))

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=img1.size(1)) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class CombinedLoss(nn.Module):
    """Combines MSE loss and SSIM loss."""
    def __init__(self, ssim_weight=0.84):
        super(CombinedLoss, self).__init__()
        self.ssim_weight = ssim_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, output, target):
        mse = self.mse_loss(output, target)
        ssim_val = ssim(output, target)
        # SSIM is a similarity metric (higher is better), so for loss we use 1 - ssim
        ssim_loss = 1 - ssim_val
        
        combined_loss = (1 - self.ssim_weight) * mse + self.ssim_weight * ssim_loss
        return combined_loss

# 4. Main Execution Block
if __name__ == "__main__":
    # Update epochs for a real run
    Config.EPOCHS = 10

    print(f"Using device: {Config.DEVICE}")

    # --- 1. Data Loading and Splitting ---
    all_image_paths = sorted(glob.glob(os.path.join(Config.TRAIN_DIR, "*.png")), 
                               key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    train_paths, val_paths = train_test_split(all_image_paths, test_size=0.1, random_state=Config.SEED)

    print(f"Training set size: {len(train_paths)}")
    print(f"Validation set size: {len(val_paths)}")

    # --- 2. Transforms ---
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=Config.IMG_SIZE, scale=(0.8, 1.0)),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
    ])

    cutpaste_transform = CutPaste()

    # --- 3. Datasets and DataLoaders ---
    train_dataset = AnomalyDataset(image_paths=train_paths, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)

    # Create validation set: half normal, half with CutPaste
    val_normal_dataset = AnomalyDataset(image_paths=val_paths, transform=val_transform)
    val_anomaly_dataset = AnomalyDataset(image_paths=val_paths, transform=val_transform, label_transform=cutpaste_transform)
    val_dataset = torch.utils.data.ConcatDataset([val_normal_dataset, val_anomaly_dataset])
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)

    # --- 4. Model, Optimizer, Loss ---
    model = UNetAutoencoder().to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    criterion = CombinedLoss(ssim_weight=Config.SSIM_WEIGHT)
    
    # --- 5. Training & Validation Loop ---
    print("Starting full training and validation pipeline...")
    best_auroc = 0.0

    for epoch in range(Config.EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        for images, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Train]"):
            inputs = images.to(Config.DEVICE)
            
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_scores = []
        val_labels = []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Val]"):
                inputs = images.to(Config.DEVICE)
                labels = labels.numpy()

                outputs = model(inputs)
                # Anomaly score is the reconstruction loss
                batch_scores = [criterion(outputs[i].unsqueeze(0), inputs[i].unsqueeze(0)).item() for i in range(inputs.size(0))]
                
                val_scores.extend(batch_scores)
                val_labels.extend(labels)

        # --- AUROC Calculation ---
        val_auroc = roc_auc_score(val_labels, val_scores)
        print(f"Epoch [{epoch+1}/{Config.EPOCHS}], Avg Train Loss: {avg_train_loss:.4f}, Validation AUROC: {val_auroc:.4f}")

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            # Here you would save the best model
            # torch.save(model.state_dict(), 'best_model.pth')
            print(f"  -> New best AUROC: {best_auroc:.4f}. Model checkpoint would be saved here.")

    print("\nTraining and validation complete.")
    print(f"Best validation AUROC achieved: {best_auroc:.4f}")

