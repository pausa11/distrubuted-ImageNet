#!/usr/bin/env python3
"""
Reorganize Tiny-ImageNet validation directory to work with ImageFolder.

The original structure has all images in val/images/ with labels in val_annotations.txt.
This script reorganizes them into val/n01443537/, val/n01629819/, etc.
"""

import os
import shutil
from pathlib import Path

def reorganize_val_directory(data_dir):
    """
    Reorganize validation directory from flat structure to class-based structure.
    
    Original:
        val/
            images/
                val_0.JPEG
                val_1.JPEG
            val_annotations.txt
    
    New:
        val/
            n01443537/
                val_0.JPEG
            n01629819/
                val_1.JPEG
    """
    val_dir = Path(data_dir) / 'val'
    images_dir = val_dir / 'images'
    annotations_file = val_dir / 'val_annotations.txt'
    
    # Check if already reorganized
    if not images_dir.exists():
        print("✅ Validation directory already reorganized!")
        return
    
    print("Reorganizing validation directory...")
    print(f"Reading annotations from: {annotations_file}")
    
    # Read annotations
    annotations = {}
    with open(annotations_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                img_name = parts[0]
                class_id = parts[1]
                annotations[img_name] = class_id
    
    print(f"Found {len(annotations)} annotations")
    
    # Create class directories and move images
    moved_count = 0
    for img_name, class_id in annotations.items():
        # Create class directory if it doesn't exist
        class_dir = val_dir / class_id
        class_dir.mkdir(exist_ok=True)
        
        # Move image from images/ to class_id/
        src = images_dir / img_name
        dst = class_dir / img_name
        
        if src.exists():
            shutil.move(str(src), str(dst))
            moved_count += 1
            if moved_count % 1000 == 0:
                print(f"  Moved {moved_count} images...")
    
    print(f"✅ Moved {moved_count} images into {len(set(annotations.values()))} class directories")
    
    # Remove empty images directory
    if images_dir.exists() and not any(images_dir.iterdir()):
        images_dir.rmdir()
        print("✅ Removed empty images directory")
    
    print("\n✅ Validation directory reorganization complete!")
    print(f"   Structure: val/<class_id>/<image_name>.JPEG")

if __name__ == '__main__':
    import sys
    
    data_dir = 'data/tiny-imagenet-200'
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    
    if not os.path.exists(data_dir):
        print(f"❌ Error: Directory not found: {data_dir}")
        sys.exit(1)
    
    reorganize_val_directory(data_dir)
