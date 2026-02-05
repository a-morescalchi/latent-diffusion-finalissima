import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import requests
from io import BytesIO
import os
from pathlib import Path
from tqdm import tqdm
import random

def download_and_process_parquet(parquet_url, output_dir="output"):
    """
    Download images from a parquet file and save them as 256x256 images.
    Also generate random masks for inpainting training.
    
    Args:
        parquet_url: URL to the parquet file
        output_dir: Base directory for output
    """
    # Create output directories
    image_dir = Path(output_dir) / "image"
    mask_dir = Path(output_dir) / "mask"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading parquet file from {parquet_url}...")
    df = pd.read_parquet(parquet_url)
    
    print(f"Found {len(df)} entries in parquet file")
    
    # Try to find the image column (common names)
    image_column = None
    for col in ['image', 'jpg', 'png', 'img', 'picture']:
        if col in df.columns:
            image_column = col
            break
    
    if image_column is None:
        print(f"Available columns: {df.columns.tolist()}")
        raise ValueError("Could not find image column. Please specify the column name.")
    
    print(f"Using column '{image_column}' for images")
    
    successful = 0
    failed = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        try:
            # Handle different image formats in parquet
            img_data = row[image_column]
            
            if isinstance(img_data, bytes):
                # Direct bytes
                img = Image.open(BytesIO(img_data))
            elif isinstance(img_data, str):
                # URL or base64
                if img_data.startswith('http'):
                    response = requests.get(img_data, timeout=10)
                    img = Image.open(BytesIO(response.content))
                else:
                    img = Image.open(BytesIO(img_data.encode()))
            elif isinstance(img_data, dict) and 'bytes' in img_data:
                # HuggingFace datasets format
                img = Image.open(BytesIO(img_data['bytes']))
            else:
                # Try direct PIL image
                img = img_data
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to 256x256
            img = img.resize((256, 256), Image.LANCZOS)
            
            # Save image
            img_path = image_dir / f"image_{idx:06d}.png"
            img.save(img_path)
            
            # Generate and save random mask
            mask = generate_random_mask(256, 256)
            mask_path = mask_dir / f"mask_{idx:06d}.png"
            mask.save(mask_path)
            
            successful += 1
            
        except Exception as e:
            failed += 1
            print(f"\nFailed to process index {idx}: {str(e)}")
            continue
    
    print(f"\n✓ Successfully processed: {successful}")
    print(f"✗ Failed: {failed}")
    print(f"\nImages saved to: {image_dir}")
    print(f"Masks saved to: {mask_dir}")

def generate_random_mask(width, height):
    """
    Generate a random mask for inpainting.
    White (255) = areas to inpaint
    Black (0) = areas to keep
    
    Args:
        width: Mask width
        height: Mask height
    
    Returns:
        PIL Image of the mask
    """
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    # Randomly choose mask type
    mask_type = random.choice(['rectangle', 'circle', 'free_form', 'multiple'])
    
    if mask_type == 'rectangle':
        # Random rectangle
        x1 = random.randint(0, width // 2)
        y1 = random.randint(0, height // 2)
        x2 = random.randint(width // 2, width)
        y2 = random.randint(height // 2, height)
        draw.rectangle([x1, y1, x2, y2], fill=255)
    
    elif mask_type == 'circle':
        # Random circle/ellipse
        x = random.randint(width // 4, 3 * width // 4)
        y = random.randint(height // 4, 3 * height // 4)
        r = random.randint(30, min(width, height) // 3)
        draw.ellipse([x-r, y-r, x+r, y+r], fill=255)
    
    elif mask_type == 'free_form':
        # Random free-form strokes
        num_strokes = random.randint(1, 5)
        for _ in range(num_strokes):
            num_points = random.randint(4, 12)
            points = []
            x, y = random.randint(0, width), random.randint(0, height)
            for _ in range(num_points):
                x += random.randint(-50, 50)
                y += random.randint(-50, 50)
                x = max(0, min(width, x))
                y = max(0, min(height, y))
                points.append((x, y))
            draw.line(points, fill=255, width=random.randint(10, 30))
    
    else:  # multiple
        # Multiple smaller shapes
        num_shapes = random.randint(2, 5)
        for _ in range(num_shapes):
            shape = random.choice(['rect', 'circle'])
            if shape == 'rect':
                x1 = random.randint(0, width - 50)
                y1 = random.randint(0, height - 50)
                w = random.randint(20, 80)
                h = random.randint(20, 80)
                draw.rectangle([x1, y1, x1+w, y1+h], fill=255)
            else:
                x = random.randint(0, width)
                y = random.randint(0, height)
                r = random.randint(15, 40)
                draw.ellipse([x-r, y-r, x+r, y+r], fill=255)
    
    return mask

# Example usage
if __name__ == "__main__":
    # Example parquet URL - replace with your actual URL
    PARQUET_URL = "hf://datasets/huggan/few-shot-obama/data/train-00000-of-00001.parquet"
    
    # Or if you have a local file:
    # PARQUET_URL = "path/to/your/file.parquet"
    
    download_and_process_parquet(
        parquet_url=PARQUET_URL,
        output_dir="loraimgs"
    )
