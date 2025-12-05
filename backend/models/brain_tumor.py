import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
import re
from ollama_service import OllamaService

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize Ollama service for brain tumor analysis
ollama_service = OllamaService(model_name="llava:13b")
use_ollama = ollama_service.is_available()

if use_ollama:
    print("✓ Ollama service detected. Using llava:13b for brain tumor analysis.")
else:
    print("✗ Ollama service not available. Falling back to PyTorch model.")
    # Initialize PyTorch model as fallback
    brain_model = models.resnet18(weights=None)
    brain_model.fc = nn.Linear(brain_model.fc.in_features, 1)
    
    # Try to load the model weights if they exist
    try:
        brain_model.load_state_dict(torch.load('models/tumor_classification_resnet18.pth', map_location=device))
    except FileNotFoundError:
        print("Warning: tumor_classification_resnet18.pth not found. Using untrained model.")
    
    brain_model.to(device)
    brain_model.eval()
    brain_model = None  # Set to None if Ollama is available

def preprocess_image(image_bytes):
    """Image preprocessing with error handling"""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return transform(image)
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

def parse_ollama_response(response_text):
    """Parse Ollama response to extract prediction and confidence"""
    response_lower = response_text.lower()
    
    # Determine if tumor is detected
    tumor_detected = any(keyword in response_lower for keyword in 
                        ['tumor detected', 'tumor present', 'abnormality', 'detected', 'positive'])
    
    # Extract confidence percentage if mentioned
    confidence = 50.0  # Default confidence
    
    # Look for percentage patterns
    import re
    percentage_match = re.search(r'(\d+(?:\.\d+)?)\s*%', response_text)
    if percentage_match:
        confidence = float(percentage_match.group(1))
    elif 'high' in response_lower or 'certain' in response_lower:
        confidence = 85.0
    elif 'moderate' in response_lower or 'likely' in response_lower:
        confidence = 70.0
    elif 'low' in response_lower or 'unlikely' in response_lower:
        confidence = 30.0
    
    confidence = max(0, min(100, confidence))  # Clamp between 0-100
    
    return tumor_detected, confidence, response_text

def predict_brain_tumor_ollama(image_bytes):
    """Brain tumor prediction using Ollama llava:13b"""
    try:
        result = ollama_service.analyze_medical_image(image_bytes, analysis_type="brain_tumor")
        
        if result.get("success"):
            analysis_text = result.get("analysis", "")
            tumor_detected, confidence, full_analysis = parse_ollama_response(analysis_text)
            
            return {
                'prediction': 'Tumor Detected' if tumor_detected else 'No Tumor Found',
                'confidence': f"{confidence:.2f}%",
                'probability': f"{confidence/100:.3f}",
                'analysis': full_analysis,
                'model': 'ollama_llava:13b'
            }
        else:
            raise ValueError(result.get("error", "Unknown error"))
    
    except Exception as e:
        raise ValueError(f"Error in Ollama brain tumor prediction: {str(e)}")

def predict_brain_tumor_pytorch(image_bytes):
    """Brain tumor prediction using PyTorch fallback"""
    try:
        image_tensor = preprocess_image(image_bytes).unsqueeze(0).to(device)
        with torch.no_grad():
            output = brain_model(image_tensor)
            probability = torch.sigmoid(output).item()
            confidence = max(probability, 1 - probability) * 100
            
            result = {
                'prediction': 'Tumor Detected' if probability > 0.5 else 'No Tumor Found',
                'confidence': f"{confidence:.2f}%",
                'probability': f"{probability:.3f}",
                'analysis': 'PyTorch model prediction',
                'model': 'pytorch_resnet18'
            }
            return result
    except Exception as e:
        raise ValueError(f"Error in PyTorch brain tumor prediction: {str(e)}")

def predict_brain_tumor(image_bytes):
    """Brain tumor prediction with Ollama preferred, PyTorch fallback"""
    if use_ollama:
        return predict_brain_tumor_ollama(image_bytes)
    else:
        return predict_brain_tumor_pytorch(image_bytes) 