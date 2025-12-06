# PADCOM Medical Dataset Preparation Guide

## Required Directory Structure

Create this folder structure in `backend/`:

```
medical_data/
├── brain_tumor/
│   ├── positive/     # 10,000 brain tumor images
│   └── negative/     # 10,000 normal brain images
└── pneumonia/
    ├── positive/     # 10,000 pneumonia X-ray images
    └── negative/     # 10,000 normal chest X-ray images
```

## Dataset Sources

### Brain Tumor Dataset (20,000 images total)
- **Kaggle - Brain MRI Images**: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection
- **BraTS Dataset**: https://www.med.upenn.edu/cbica/brats2020/
- **Figshare Brain Tumor**: https://figshare.com/articles/dataset/brain_tumor_dataset/1512427

### Pneumonia Dataset (20,000 images total)
- **Chest X-Ray Images (Pneumonia)**: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- **NIH Chest X-rays**: https://www.kaggle.com/datasets/nih-chest-xrays/data
- **RSNA Pneumonia Detection**: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge

## Quick Setup Script

Run this Python script to create the folder structure:

```python
import os

base_dir = 'medical_data'
diseases = ['brain_tumor', 'pneumonia']
categories = ['positive', 'negative']

for disease in diseases:
    for category in categories:
        path = os.path.join(base_dir, disease, category)
        os.makedirs(path, exist_ok=True)
        print(f'Created: {path}')
```

## Download Instructions

### Option 1: Manual Download
1. Download datasets from Kaggle links above
2. Extract and organize into the folder structure
3. Ensure exactly 10,000 images per category

### Option 2: Kaggle API (Automated)
```bash
pip install kaggle

# Download brain tumor dataset
kaggle datasets download -d navoneel/brain-mri-images-for-brain-tumor-detection
unzip brain-mri-images-for-brain-tumor-detection.zip -d medical_data/brain_tumor/

# Download pneumonia dataset
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d medical_data/pneumonia/
```

## Training the Models

Once you have the datasets ready:

```bash
cd backend
python train_models.py
```

This will:
- Load all 40,000 images (10k per class × 2 diseases × 2 classes)
- Train ResNet50 models with 80/20 train/validation split
- Save trained models to `backend/models/`
- Generate accuracy reports

## Expected Results

With 10,000 images per class:
- **Training time**: ~2-4 hours per model (on GPU)
- **Expected accuracy**: 95-98% on validation set
- **Model size**: ~90MB per model

## Troubleshooting

**Out of Memory Error:**
- Reduce BATCH_SIZE in train_models.py (try 16 or 8)
- Use smaller image size (try 128x128 instead of 224x224)

**Slow Training:**
- Ensure PyTorch is using GPU: Check with `torch.cuda.is_available()`
- Reduce number of workers in DataLoader

**Not Enough Images:**
- Use data augmentation (already enabled in training script)
- Consider using pre-trained models without retraining
