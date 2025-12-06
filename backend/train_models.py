"""
PADCOM Medical Model Training Script
Trains brain tumor and pneumonia classification models with large datasets
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
IMG_SIZE = 224

class MedicalImageDataset(Dataset):
    """Custom dataset for medical images"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_model(num_classes=2):
    """Create ResNet50 model for binary classification"""
    model = models.resnet50(pretrained=True)
    
    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace final layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    
    return model.to(DEVICE)

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    """Train the model"""
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1}/{epochs}')
        print('-' * 60)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc='Training'):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validation'):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            return model, history, best_acc
    
    return model, history, best_acc

def prepare_dataset(data_dir, disease_type):
    """Prepare dataset from directory structure"""
    print(f"\nPreparing {disease_type} dataset...")
    
    positive_dir = os.path.join(data_dir, disease_type, 'positive')
    negative_dir = os.path.join(data_dir, disease_type, 'negative')
    
    image_paths = []
    labels = []
    
    # Load positive images
    if os.path.exists(positive_dir):
        for img_file in os.listdir(positive_dir):
            if img_file.endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(positive_dir, img_file))
                labels.append(1)  # Positive
    
    # Load negative images
    if os.path.exists(negative_dir):
        for img_file in os.listdir(negative_dir):
            if img_file.endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(negative_dir, img_file))
                labels.append(0)  # Negative
    
    print(f"Total images: {len(image_paths)}")
    print(f"Positive: {labels.count(1)} | Negative: {labels.count(0)}")
    
    return image_paths, labels

def main():
    """Main training pipeline"""
    print("="*60)
    print("PADCOM Medical Model Training")
    print(f"Device: {DEVICE}")
    print("="*60)
    
    # Data augmentation and normalization
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    data_dir = 'medical_data'  # Root directory for datasets
    diseases = ['brain_tumor', 'pneumonia']
    
    for disease in diseases:
        print(f"\n{'='*60}")
        print(f"Training {disease.upper()} Model")
        print(f"{'='*60}")
        
        # Prepare data
        image_paths, labels = prepare_dataset(data_dir, disease)
        
        if len(image_paths) == 0:
            print(f"No images found for {disease}. Skipping...")
            continue
        
        # Split data
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Create datasets
        train_dataset = MedicalImageDataset(train_paths, train_labels, train_transform)
        val_dataset = MedicalImageDataset(val_paths, val_labels, val_transform)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        # Create model
        model = create_model(num_classes=2)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
        
        # Train model
        model, history, best_acc = train_model(
            model, train_loader, val_loader, criterion, optimizer, EPOCHS
        )
        
        # Save model
        model_path = f'models/{disease}_model.pth'
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"\n✓ Model saved: {model_path}")
        print(f"✓ Best Validation Accuracy: {best_acc:.2f}%")
        
        # Save training history
        history_path = f'models/{disease}_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"✓ Training history saved: {history_path}")

if __name__ == "__main__":
    main()
