# PADCOM Medical Image Training - Complete Guide

## ✅ Step 1: Folder Structure (DONE)

The following folders have been created:
```
backend/medical_data/
├── brain_tumor/
│   ├── positive/  ← Place 10,000 tumor MRI images here
│   └── negative/  ← Place 10,000 normal brain MRI images here
└── pneumonia/
    ├── positive/  ← Place 10,000 pneumonia X-ray images here
    └── negative/  ← Place 10,000 normal chest X-ray images here
```

## 📥 Step 2: Download & Organize Datasets

### Method 1: Kaggle API (Recommended for 10k images)

1. **Set up Kaggle credentials:**
   ```bash
   # Go to https://www.kaggle.com/account
   # Click "Create New API Token"
   # Download kaggle.json
   # Place it in: C:\Users\YOUR_USERNAME\.kaggle\kaggle.json
   ```

2. **Run the download script:**
   ```bash
   python download_datasets.py
   ```

### Method 2: Manual Download (Easiest)

1. **Download from Kaggle website:**
   - Brain Tumor: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection
   - Pneumonia: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

2. **Extract and organize:**
   - Separate images into positive/negative folders
   - Aim for 10,000 images per category (or at least 1,000 for decent results)

### Method 3: Start Small (Testing)

For quick testing, you can start with just 100-500 images per category. The model will train faster but be less accurate.

## 🔍 Check Your Dataset Status

Run this anytime to see how many images you have:
```bash
python check_datasets.py
```

## 🚀 Step 3: Train the Models

Once you have images in the folders:

```bash
python train_models.py
```

**Training Configuration:**
- Model: ResNet50 (pre-trained on ImageNet)
- Epochs: 50 (adjust in train_models.py)
- Batch Size: 32 (reduce if out of memory)
- Image Size: 224x224

**Expected Training Time:**
- With GPU: 2-4 hours per model (for 10k images)
- With CPU: 12-24 hours per model

**Expected Accuracy:**
- 10,000 images per class: **95-98%**
- 1,000 images per class: **85-90%**
- 100 images per class: **70-80%**

## 📊 Monitoring Training

The script will show:
- Real-time training progress
- Training and validation loss
- Training and validation accuracy
- Best model is automatically saved

## ✅ After Training

Models will be saved to:
- `backend/models/brain_tumor_model.pth`
- `backend/models/pneumonia_model.pth`

Training history saved to:
- `backend/models/brain_tumor_history.json`
- `backend/models/pneumonia_history.json`

## 🧪 Testing the Models

1. **Start the backend:**
   ```bash
   cd backend
   python run.py
   ```

2. **Start the frontend:**
   ```bash
   cd frontend
   npm start
   ```

3. **Access the app:**
   - Chat: http://localhost:3000/
   - Medical Analysis: http://localhost:3000/medical

## 🔧 Troubleshooting

**Out of Memory Error:**
```python
# In train_models.py, reduce:
BATCH_SIZE = 16  # or even 8
IMG_SIZE = 128   # instead of 224
```

**Slow Training:**
```python
# Check if GPU is being used:
import torch
print(torch.cuda.is_available())  # Should be True

# If False, install CUDA-enabled PyTorch:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Not Enough Images:**
- You can train with fewer images (minimum ~100 per class)
- Models will be less accurate but still functional
- Consider data augmentation (already enabled in script)

## 📚 Alternative Datasets

If you can't access Kaggle, try these:

**Brain Tumor:**
- BraTS Challenge: https://www.med.upenn.edu/cbica/brats2020/
- Figshare: https://figshare.com/articles/dataset/brain_tumor_dataset/1512427

**Pneumonia:**
- NIH Chest X-rays: https://nihcc.app.box.com/v/ChestXray-NIHCC
- RSNA Pneumonia: https://www.rsna.org/education/ai-resources-and-training

## 💡 Tips for Best Results

1. **Quality over quantity** - 5,000 high-quality images > 10,000 poor quality
2. **Balance your classes** - Equal numbers of positive/negative
3. **Clean your data** - Remove corrupted or mislabeled images
4. **Use GPU** - Training will be 10-20x faster
5. **Monitor validation accuracy** - Stop if it stops improving

## 🎯 Current System Status

Run to check everything:
```bash
python check_datasets.py
```

This will show:
- ✓ How many images you have
- ✓ If you're ready to train
- ✓ What steps remain

---

**Need help?** Check the output of `check_datasets.py` for guidance!
