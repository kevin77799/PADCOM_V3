"""
Quick Dataset Organizer
Works with any medical images you already have or download manually
"""

import os
import shutil
from pathlib import Path

def count_images(directory):
    """Count image files in a directory"""
    if not os.path.exists(directory):
        return 0
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    count = sum(1 for f in Path(directory).rglob('*') if f.suffix.lower() in image_extensions)
    return count

def check_dataset_status():
    """Check current status of datasets"""
    print("="*60)
    print("PADCOM Dataset Status Check")
    print("="*60)
    print()
    
    datasets = {
        'Brain Tumor Positive': 'medical_data/brain_tumor/positive',
        'Brain Tumor Negative': 'medical_data/brain_tumor/negative',
        'Pneumonia Positive': 'medical_data/pneumonia/positive',
        'Pneumonia Negative': 'medical_data/pneumonia/negative'
    }
    
    total_images = 0
    ready_to_train = True
    
    for name, path in datasets.items():
        count = count_images(path)
        total_images += count
        status = "✓" if count >= 100 else "✗"  # At least 100 images to start
        print(f"{status} {name:25} {count:6,} images")
        if count < 100:
            ready_to_train = False
    
    print()
    print(f"Total images: {total_images:,}")
    print()
    
    if ready_to_train:
        print("✓ Dataset ready! You can start training.")
        print()
        print("Run: python train_models.py")
    else:
        print("✗ Not enough images yet. Need at least 100 per category to start.")
        print()
        print("MANUAL DOWNLOAD OPTIONS:")
        print()
        print("Option 1 - Download from Kaggle (requires account):")
        print("  1. Create account at https://kaggle.com")
        print("  2. Download datasets:")
        print("     Brain: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection")
        print("     Pneumonia: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
        print("  3. Extract and place images in folders above")
        print()
        print("Option 2 - Use any medical images you have:")
        print("  - Place tumor MRI images in: medical_data/brain_tumor/positive/")
        print("  - Place normal brain MRI in: medical_data/brain_tumor/negative/")
        print("  - Place pneumonia X-rays in: medical_data/pneumonia/positive/")
        print("  - Place normal chest X-rays in: medical_data/pneumonia/negative/")
        print()
        print("Option 3 - Start with smaller dataset:")
        print("  - Even 500-1000 images per category can give decent results")
        print("  - The model will train faster but may be less accurate")
    
    print("="*60)
    return ready_to_train

if __name__ == "__main__":
    ready = check_dataset_status()
    
    if ready:
        response = input("\nStart training now? (y/n): ")
        if response.lower() == 'y':
            print("\nStarting training...")
            import subprocess
            subprocess.run(['python', 'train_models.py'])
