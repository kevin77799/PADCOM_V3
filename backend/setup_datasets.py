"""
Automated Dataset Setup for PADCOM Medical Models
Creates folder structure and provides download instructions
"""

import os

def create_folder_structure():
    """Create the required folder structure for medical datasets"""
    base_dir = 'medical_data'
    diseases = ['brain_tumor', 'pneumonia']
    categories = ['positive', 'negative']
    
    print("="*60)
    print("PADCOM Medical Dataset Setup")
    print("="*60)
    print()
    
    for disease in diseases:
        for category in categories:
            path = os.path.join(base_dir, disease, category)
            os.makedirs(path, exist_ok=True)
            print(f'✓ Created: {path}')
    
    print()
    print("="*60)
    print("Folder structure created successfully!")
    print("="*60)
    print()
    print("NEXT STEPS:")
    print()
    print("1. Download datasets from Kaggle:")
    print()
    print("   Brain Tumor Dataset:")
    print("   https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection")
    print()
    print("   Pneumonia Dataset:")
    print("   https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
    print()
    print("2. Extract and organize images:")
    print(f"   - Place 10,000 tumor images in: {os.path.join(base_dir, 'brain_tumor', 'positive')}")
    print(f"   - Place 10,000 normal brain images in: {os.path.join(base_dir, 'brain_tumor', 'negative')}")
    print(f"   - Place 10,000 pneumonia X-rays in: {os.path.join(base_dir, 'pneumonia', 'positive')}")
    print(f"   - Place 10,000 normal chest X-rays in: {os.path.join(base_dir, 'pneumonia', 'negative')}")
    print()
    print("3. Train models:")
    print("   python train_models.py")
    print()
    print("="*60)

if __name__ == "__main__":
    create_folder_structure()
