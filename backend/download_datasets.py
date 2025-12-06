"""
Kaggle Dataset Download Script
Automatically downloads medical imaging datasets
"""

import os
import subprocess
import sys

def check_kaggle_api():
    """Check if kaggle API is installed"""
    try:
        import kaggle
        print("✓ Kaggle API is installed")
        return True
    except ImportError:
        print("✗ Kaggle API not found")
        print("\nInstalling kaggle package...")
        subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"])
        return True

def check_kaggle_credentials():
    """Check if kaggle credentials are configured"""
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.exists(kaggle_json):
        print("✓ Kaggle credentials found")
        return True
    else:
        print("✗ Kaggle credentials not found")
        print("\nPlease set up Kaggle API credentials:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Save kaggle.json to: ~/.kaggle/kaggle.json")
        print("   (On Windows: C:\\Users\\YourName\\.kaggle\\kaggle.json)")
        return False

def download_datasets():
    """Download medical imaging datasets from Kaggle"""
    print("="*60)
    print("PADCOM Dataset Download")
    print("="*60)
    print()
    
    if not check_kaggle_api():
        return False
    
    if not check_kaggle_credentials():
        return False
    
    datasets = [
        {
            'name': 'Brain Tumor MRI',
            'kaggle_id': 'navoneel/brain-mri-images-for-brain-tumor-detection',
            'output_dir': 'medical_data/brain_tumor_raw'
        },
        {
            'name': 'Chest X-Ray Pneumonia',
            'kaggle_id': 'paultimothymooney/chest-xray-pneumonia',
            'output_dir': 'medical_data/pneumonia_raw'
        }
    ]
    
    for dataset in datasets:
        print(f"\n📥 Downloading {dataset['name']}...")
        print(f"   Dataset: {dataset['kaggle_id']}")
        
        try:
            # Create output directory
            os.makedirs(dataset['output_dir'], exist_ok=True)
            
            # Download using kaggle CLI
            cmd = [
                'kaggle', 'datasets', 'download',
                dataset['kaggle_id'],
                '-p', dataset['output_dir'],
                '--unzip'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"   ✓ Downloaded to {dataset['output_dir']}")
            else:
                print(f"   ✗ Error: {result.stderr}")
                
        except Exception as e:
            print(f"   ✗ Error downloading: {e}")
    
    print()
    print("="*60)
    print("Download Complete!")
    print("="*60)
    print()
    print("NEXT STEPS:")
    print("1. Organize downloaded images into:")
    print("   - medical_data/brain_tumor/positive (10k images)")
    print("   - medical_data/brain_tumor/negative (10k images)")
    print("   - medical_data/pneumonia/positive (10k images)")
    print("   - medical_data/pneumonia/negative (10k images)")
    print()
    print("2. Run training:")
    print("   python train_models.py")
    print()

if __name__ == "__main__":
    download_datasets()
