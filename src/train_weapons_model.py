#!/usr/bin/env python3
"""
Weapons Detection Model Trainer

This script downloads and processes the Kaggle weapons dataset, then fine-tunes
a YOLOv8 model to classify weapons into specific categories:
- rifle
- bazooka
- shotgun
- handgun
- knife
- grenade
- weapon (other weapons)

Usage:
  python src/train_weapons_model.py --epochs 50 --batch 16 --img-size 640
"""

import os
import sys
import argparse
import yaml
import shutil
import subprocess
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
import torch
from config import settings
from utils.logger import logger

# Define weapon classes
WEAPON_CLASSES = [
    'rifle',
    'bazooka',
    'shotgun', 
    'handgun',
    'knife',
    'grenade',
    'weapon'  # Generic weapon class for others
]

def setup_kaggle_api():
    """Setup Kaggle API credentials"""
    # Check if kaggle module is installed
    try:
        import kaggle
    except ImportError:
        logger.info("Installing kaggle module...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        
    # Check for Kaggle API credentials
    kaggle_dir = os.path.join(os.path.expanduser('~'), '.kaggle')
    if not os.path.exists(kaggle_dir):
        os.makedirs(kaggle_dir)
        
    kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')
    if not os.path.exists(kaggle_json):
        # Create a sample kaggle.json file
        logger.warning("Kaggle API credentials not found.")
        logger.info("Please provide your Kaggle username and API key:")
        username = input("Kaggle Username: ")
        key = input("Kaggle API Key: ")
        
        with open(kaggle_json, 'w') as f:
            f.write(f'{{"username":"{username}","key":"{key}"}}')
        
        # Set proper permissions
        os.chmod(kaggle_json, 0o600)
        
    logger.info("Kaggle API setup complete.")

def download_kaggle_dataset():
    """Download weapons dataset from Kaggle"""
    import kaggle
    
    # Define possible datasets - we'll use a dataset with good weapon images
    # You can change this to another weapons dataset on Kaggle
    dataset_slug = "moussatantan/weapons-dataset-images"
    
    # Create data directory
    data_dir = os.path.join(settings.BASE_DIR, "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    # Download dataset
    logger.info(f"Downloading dataset {dataset_slug}...")
    kaggle.api.dataset_download_files(dataset_slug, path=data_dir, unzip=True)
    
    logger.info("Dataset downloaded successfully.")
    return os.path.join(data_dir, "weapons-dataset-images")

def prepare_yolo_dataset(dataset_path):
    """Prepare dataset in YOLO format"""
    # Create YOLO dataset directories
    yolo_dataset_dir = os.path.join(settings.BASE_DIR, "data", "weapons-yolo")
    if os.path.exists(yolo_dataset_dir):
        shutil.rmtree(yolo_dataset_dir)
        
    os.makedirs(yolo_dataset_dir)
    
    # Create train, val directories
    for split in ['train', 'val']:
        os.makedirs(os.path.join(yolo_dataset_dir, split, 'images'))
        os.makedirs(os.path.join(yolo_dataset_dir, split, 'labels'))
    
    # Create dataset.yaml
    dataset_yaml = {
        'path': os.path.abspath(yolo_dataset_dir),
        'train': 'train/images',
        'val': 'val/images',
        'names': {i: name for i, name in enumerate(WEAPON_CLASSES)}
    }
    
    with open(os.path.join(yolo_dataset_dir, 'dataset.yaml'), 'w') as f:
        yaml.dump(dataset_yaml, f)
    
    # Process images and create annotations
    process_weapons_dataset(dataset_path, yolo_dataset_dir)
    
    return os.path.join(yolo_dataset_dir, 'dataset.yaml')

def process_weapons_dataset(dataset_path, output_dir):
    """Process the downloaded dataset and convert to YOLO format"""
    import cv2
    import random
    import numpy as np
    from glob import glob
    from tqdm import tqdm
    
    # Map folder names to our weapon classes
    class_mapping = {
        'rifle': 'rifle',
        'guns': 'handgun',  # Mapping guns folder to handgun
        'shotgun': 'shotgun',
        'launchers': 'bazooka',  # Mapping launchers to bazooka
        'knife': 'knife',
        'grenade': 'grenade',
        # Add any other mappings based on dataset structure
    }
    
    # Generic detector for weapons not in our specific classes
    def detect_weapon_type(img_path):
        # For demonstration, we're using simple heuristics based on image name
        # In a real scenario, you might use a classifier
        filename = os.path.basename(img_path).lower()
        
        if 'rifle' in filename or 'ar' in filename or 'ak' in filename:
            return 'rifle'
        elif 'pistol' in filename or 'handgun' in filename:
            return 'handgun'
        elif 'shotgun' in filename:
            return 'shotgun'
        elif 'launcher' in filename or 'bazooka' in filename or 'rpg' in filename:
            return 'bazooka'
        elif 'knife' in filename or 'blade' in filename:
            return 'knife'
        elif 'grenade' in filename:
            return 'grenade'
        else:
            return 'weapon'  # Generic weapon class
    
    # Get all image paths
    all_images = []
    for class_dir in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_dir)
        if os.path.isdir(class_path):
            weapons_class = class_mapping.get(class_dir.lower(), 'weapon')
            
            # Get images for this class
            img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            for ext in img_extensions:
                img_paths = glob(os.path.join(class_path, '**', ext), recursive=True)
                for img_path in img_paths:
                    # Determine specific weapon type based on filename or path
                    weapon_type = weapons_class
                    if weapon_type == 'weapon':
                        weapon_type = detect_weapon_type(img_path)
                    
                    all_images.append((img_path, weapon_type))
    
    # Shuffle images
    random.shuffle(all_images)
    
    # Split into train/val (80/20)
    split_idx = int(len(all_images) * 0.8)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    # Function to create YOLO annotation
    def create_yolo_annotation(img_path, weapon_type, output_img_path, output_label_path):
        try:
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                return False
                
            h, w = img.shape[:2]
            
            # Get class index
            class_idx = WEAPON_CLASSES.index(weapon_type)
            
            # Create a simple bounding box covering most of the image
            # In real training, you'd have precise annotations
            # For this example, we'll assume the weapon covers 70-90% of the image
            box_w = w * random.uniform(0.7, 0.9)
            box_h = h * random.uniform(0.7, 0.9)
            box_x = (w - box_w) / 2
            box_y = (h - box_h) / 2
            
            # Convert to YOLO format: class_idx, x_center, y_center, width, height
            # All normalized to 0-1
            x_center = (box_x + box_w/2) / w
            y_center = (box_y + box_h/2) / h
            norm_width = box_w / w
            norm_height = box_h / h
            
            # Save image
            cv2.imwrite(output_img_path, img)
            
            # Save annotation
            with open(output_label_path, 'w') as f:
                f.write(f"{class_idx} {x_center} {y_center} {norm_width} {norm_height}\n")
                
            return True
        except Exception as e:
            logger.error(f"Error processing {img_path}: {str(e)}")
            return False
    
    # Process train images
    logger.info("Processing training images...")
    for i, (img_path, weapon_type) in enumerate(tqdm(train_images)):
        output_img_path = os.path.join(output_dir, 'train', 'images', f'train_{i}.jpg')
        output_label_path = os.path.join(output_dir, 'train', 'labels', f'train_{i}.txt')
        create_yolo_annotation(img_path, weapon_type, output_img_path, output_label_path)
    
    # Process validation images
    logger.info("Processing validation images...")
    for i, (img_path, weapon_type) in enumerate(tqdm(val_images)):
        output_img_path = os.path.join(output_dir, 'val', 'images', f'val_{i}.jpg')
        output_label_path = os.path.join(output_dir, 'val', 'labels', f'val_{i}.txt')
        create_yolo_annotation(img_path, weapon_type, output_img_path, output_label_path)
    
    logger.info(f"Dataset prepared: {len(train_images)} training images, {len(val_images)} validation images")

def train_model(dataset_yaml, args):
    """Train YOLO model on the weapons dataset"""
    # Ensure models directory exists
    os.makedirs(settings.MODELS_DIR, exist_ok=True)
    
    # Start with a pre-trained YOLOv8 model
    model = YOLO('yolov8n.pt')
    
    # Train the model
    logger.info("Starting model training...")
    results = model.train(
        data=dataset_yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.img_size,
        patience=10,
        device=0 if torch.cuda.is_available() and settings.USE_GPU else 'cpu'
    )
    
    # Save the final model
    final_model_path = os.path.join(settings.MODELS_DIR, "weapons_detector.pt")
    shutil.copy(results.checkpoint.last, final_model_path)
    logger.info(f"Model saved to {final_model_path}")
    
    return final_model_path

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train weapons detection model")
    
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--no-kaggle", action="store_true", help="Skip Kaggle dataset download")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    logger.info("Starting weapons detection model training")
    
    # Setup Kaggle API
    if not args.no_kaggle:
        setup_kaggle_api()
        dataset_path = download_kaggle_dataset()
    else:
        # If skipping Kaggle, expect dataset to already exist
        dataset_path = os.path.join(settings.BASE_DIR, "data", "weapons-dataset-images")
        if not os.path.exists(dataset_path):
            logger.error("Dataset not found. Please download manually or use Kaggle API.")
            sys.exit(1)
    
    # Prepare YOLO dataset
    dataset_yaml = prepare_yolo_dataset(dataset_path)
    
    # Train model
    model_path = train_model(dataset_yaml, args)
    
    logger.info(f"Training completed. Model saved to {model_path}")
    logger.info("To use this model, update the config/settings.py file to use the new model.") 