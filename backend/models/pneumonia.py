"""
Pneumonia Detection Model
Uses trained ResNet50 model for binary classification
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import os

# Model configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/pneumonia_model.pth')
IMG_SIZE = 224

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Global model variable (lazy loading)
_model = None
_model_loaded = False

def load_model():
    """Load the trained pneumonia model"""
    global _model, _model_loaded
    
    if _model_loaded:
        return _model
    
    try:
        # Create model architecture
        model = models.resnet50(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
        
        # Load trained weights
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            _model = model
            _model_loaded = True
            print(f"✓ Pneumonia model loaded from {MODEL_PATH}")
            return model
        else:
            print(f"⚠ Model file not found: {MODEL_PATH}")
            print("  Please train the model first using: python train_models.py")
            return None
    
    except Exception as e:
        print(f"✗ Error loading pneumonia model: {e}")
        return None

def predict_pneumonia(image_bytes):
    """
    Predict pneumonia from chest X-ray image bytes
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        dict: Prediction results with confidence and class
    """
    try:
        # Load model if not already loaded
        model = load_model()
        if model is None:
            return {
                'prediction': 'Model Not Available',
                'confidence': '0%',
                'class': 'unknown',
                'message': 'Model not trained yet. Please run train_models.py first.'
            }
        
        # Load and preprocess image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            confidence_pct = confidence.item() * 100
            predicted_class = predicted.item()
            
            # Map class to label
            class_labels = {
                0: 'Normal',
                1: 'Pneumonia Detected'
            }
            
            return {
                'prediction': class_labels[predicted_class],
                'confidence': f'{confidence_pct:.2f}%',
                'class': 'positive' if predicted_class == 1 else 'negative',
                'probabilities': {
                    'normal': f'{probabilities[0][0].item() * 100:.2f}%',
                    'pneumonia': f'{probabilities[0][1].item() * 100:.2f}%'
                }
            }
    
    except Exception as e:
        print(f"Error in pneumonia prediction: {e}")
        return {
            'prediction': 'Error',
            'confidence': '0%',
            'class': 'error',
            'message': str(e)
        }
