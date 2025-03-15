import os
import shutil
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
import random

def convert_to_yolo_format(shape_attributes, img_width, img_height):
    """Convert VIA annotations to YOLO format."""
    if 'x' not in shape_attributes or 'y' not in shape_attributes or \
       'width' not in shape_attributes or 'height' not in shape_attributes:
        return None

    x = shape_attributes['x']
    y = shape_attributes['y']
    width = shape_attributes['width']
    height = shape_attributes['height']

    # Convert to YOLO format (normalized coordinates)
    x_center = (x + width/2) / img_width
    y_center = (y + height/2) / img_height
    norm_width = width / img_width
    norm_height = height / img_height

    # Ensure values are between 0 and 1
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    norm_width = max(0, min(1, norm_width))
    norm_height = max(0, min(1, norm_height))

    return [x_center, y_center, norm_width, norm_height]

def find_image_in_subdirs(img_path: Path, img_name: str) -> Optional[Path]:
    # Vérifier d'abord dans le dossier principal
    img_file = img_path / img_name
    if img_file.exists():
        return img_file
    return None

def prepare_dataset(img_path: Path, base_dir: Path, anno_path: Path):
    # Create output directories
    train_img_dir = base_dir / 'images' / 'train'
    val_img_dir = base_dir / 'images' / 'val'
    train_label_dir = base_dir / 'labels' / 'train'
    val_label_dir = base_dir / 'labels' / 'val'

    for dir_path in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Load annotations
    with open(anno_path, 'r') as f:
        data = json.load(f)
    
    # Check if it's VIA format
    if '_via_img_metadata' in data:
        annotations = data['_via_img_metadata']
    else:
        annotations = data

    # Process each image
    for key, value in annotations.items():
        # Extract the actual filename from the key (remove size suffix if present)
        filename = value['filename']  # Use the filename from the value instead of the key
        if not filename.endswith(('.jpg', '.jpeg', '.png')):
            continue

        # Find the image file
        img_file = find_image_in_subdirs(img_path, filename)
        if img_file is None:
            print(f"Image not found: {img_path}/{filename}")
            continue

        # Read image dimensions
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"Cannot read image: {img_file}")
            continue
        
        img_height, img_width = img.shape[:2]

        # Randomly assign to train or val set
        is_train = random.random() < 0.8

        # Copy image to appropriate directory
        dst_img_dir = train_img_dir if is_train else val_img_dir
        dst_label_dir = train_label_dir if is_train else val_label_dir

        # Copy image
        shutil.copy2(img_file, dst_img_dir / filename)

        # Convert and save annotations if they exist
        if 'regions' in value and value['regions']:
            yolo_annotations = []
            for region in value['regions']:
                if 'shape_attributes' in region:
                    yolo_bbox = convert_to_yolo_format(region['shape_attributes'], img_width, img_height)
                    if yolo_bbox:
                        yolo_annotations.append(f"0 {' '.join(map(str, yolo_bbox))}")

            if yolo_annotations:
                label_file = dst_label_dir / (Path(filename).stem + '.txt')
                with open(label_file, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
                print(f"Processed {filename} with {len(yolo_annotations)} annotations")

if __name__ == "__main__":
    # Chemins des dossiers et fichiers
    base_dir = Path("data")
    datasets = [
        ("sm", "annotation.json"),
        ("bh", "annotation.json"),
        ("bh-phone", "annotation.json")
    ]
    
    # Traiter chaque dataset
    for img_dir, anno_file in datasets:
        img_path = base_dir / img_dir
        anno_path = img_path / anno_file
        
        if not img_path.exists():
            print(f"Dossier non trouvé: {img_path}")
            continue
            
        if not anno_path.exists():
            print(f"Fichier d'annotations non trouvé: {anno_path}")
            continue
            
        print(f"\nTraitement du dataset {img_dir}...")
        prepare_dataset(img_path, base_dir, anno_path) 