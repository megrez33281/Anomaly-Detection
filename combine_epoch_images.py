import os
from PIL import Image, ImageDraw, ImageFont
import argparse

def combine_images(image_id, total_epochs, output_filename):
    """
    Combines comparison images from multiple epochs for a single ID into one summary image.
    It arranges them in a grid and labels each with its epoch number.
    """
    base_path = 'inference_results'
    images_to_combine = []
    
    # --- Load images and add labels ---
    for i in range(1, total_epochs + 1):
        epoch_path = os.path.join(base_path, f'epoch_{i}')
        img_path = os.path.join(epoch_path, f'compare_{image_id}.png')
        
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGBA")
            
            # Add epoch label
            draw = ImageDraw.Draw(img)
            try:
                # Try to use a common font, fallback to default
                font = ImageFont.truetype("arial.ttf", 40)
            except IOError:
                font = ImageFont.load_default(size=40)
            
            text = f"Epoch {i}"
            text_position = (10, 10)
            
            # Draw a white outline for better visibility
            draw.text((text_position[0]-2, text_position[1]-2), text, font=font, fill="white")
            draw.text((text_position[0]+2, text_position[1]-2), text, font=font, fill="white")
            draw.text((text_position[0]-2, text_position[1]+2), text, font=font, fill="white")
            draw.text((text_position[0]+2, text_position[1]+2), text, font=font, fill="white")
            # Draw the main text
            draw.text(text_position, text, font=font, fill="black")

            images_to_combine.append(img)
        else:
            print(f"Warning: Image not found for epoch {i} at {img_path}")

    if not images_to_combine:
        print("Error: No images found to combine.")
        return

    # --- Create grid layout ---
    img_width, img_height = images_to_combine[0].size
    
    # Define grid size (e.g., 2 rows, 5 columns for 10 epochs)
    cols = 5
    rows = (len(images_to_combine) + cols - 1) // cols
    
    canvas_width = cols * img_width
    canvas_height = rows * img_height
    
    canvas = Image.new('RGBA', (canvas_width, canvas_height), (255, 255, 255, 255))
    
    # Paste images onto the canvas
    for i, img in enumerate(images_to_combine):
        row = i // cols
        col = i % cols
        x_offset = col * img_width
        y_offset = row * img_height
        canvas.paste(img, (x_offset, y_offset))

    # Save the final image
    canvas.save(output_filename)
    print(f"Successfully combined {len(images_to_combine)} images into {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine epoch comparison images into a single summary file.")
    parser.add_argument("image_id", type=str, help="The common image ID to combine (e.g., '432').")
    parser.add_argument("total_epochs", type=int, help="The total number of epochs to process.")
    parser.add_argument("-o", "--output", type=str, help="Output filename.")
    
    args = parser.parse_args()
    
    output_file = args.output if args.output else f"summary_{args.image_id}.png"
    
    combine_images(args.image_id, args.total_epochs, output_file)
