
import torch
import os
import glob
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import random

# Import custom modules
from Config import Config
from UNet_Autoencoder_Model import UNetAutoencoder

def tensor_to_pil(tensor):
    """Converts a PyTorch tensor to a PIL Image."""
    # Move tensor to CPU and detach from graph
    image = tensor.cpu().clone().detach()
    # Remove batch dimension
    image = image.squeeze(0)
    # Transpose from (C, H, W) to (H, W, C)
    image = image.permute(1, 2, 0)
    # Convert to numpy array
    image = image.numpy()
    # Denormalize from [0, 1] to [0, 255]
    image = (image * 255).astype(np.uint8)
    return Image.fromarray(image)

if __name__ == "__main__":
    print(f"Using device: {Config.DEVICE}")

    # --- 1. Prepare Model ---
    model = UNetAutoencoder().to(Config.DEVICE)
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE))
    model.eval()
    print(f"Model loaded from {Config.MODEL_SAVE_PATH}")

    # --- 2. Prepare Test Data ---
    test_image_paths = sorted(glob.glob(os.path.join(Config.TEST_DIR, "*.png")), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    
    # Randomly select 5 images to visualize
    if len(test_image_paths) > 5:
        selected_paths = random.sample(test_image_paths, 5)
    else:
        selected_paths = test_image_paths
        
    print(f"Selected {len(selected_paths)} images for visualization: {[os.path.basename(p) for p in selected_paths]}")

    # Define the same transformation as in inference
    transform = A.Compose([
        A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2(),
    ])

    # --- 3. Create Output Directory ---
    output_dir = "reconstruction_results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to '{output_dir}' directory.")

    # --- 4. Perform Inference and Save Images ---
    with torch.no_grad():
        for img_path in tqdm(selected_paths, desc="Generating reconstructions"):
            img_id = os.path.splitext(os.path.basename(img_path))[0]
            
            # Load and transform the original image
            original_pil = Image.open(img_path).convert("RGB")
            image_np = np.array(original_pil)
            augmented = transform(image=image_np)
            input_tensor = augmented['image'].unsqueeze(0).to(Config.DEVICE)

            # Get model reconstruction
            recon_tensor = model(input_tensor)

            # Convert tensors back to PIL images
            # For the input, we need to reverse the normalization to see the correct colors
            input_pil_resized = Image.fromarray(np.array(A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE)(image=image_np)['image']))
            recon_pil = tensor_to_pil(recon_tensor)

            # Create a side-by-side comparison image
            comparison_img = Image.new('RGB', (Config.IMG_SIZE * 2, Config.IMG_SIZE))
            comparison_img.paste(input_pil_resized, (0, 0))
            comparison_img.paste(recon_pil, (Config.IMG_SIZE, 0))

            # Save the comparison image
            comparison_img.save(os.path.join(output_dir, f"compare_{img_id}.png"))

    print("\nVisualization complete.")
