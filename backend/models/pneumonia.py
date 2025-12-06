try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from PIL import Image
import io
import re
from ollama_service import OllamaService

# Device configuration (only if torch is available)
if TORCH_AVAILABLE:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Image preprocessing transforms
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
else:
    device = None
    transform = None

# Initialize Ollama service for pneumonia analysis
ollama_service = OllamaService(model_name="llava:7b")
use_ollama = ollama_service.is_available()

if use_ollama:
    print("✓ Ollama service detected. Using llava:7b for pneumonia analysis.")
    pneumonia_model = None
else:
    print("✗ Ollama service not available. Falling back to PyTorch model.")
    if TORCH_AVAILABLE:
        pneumonia_model = None  # Will be initialized below if needed
    else:
        pneumonia_model = None

if TORCH_AVAILABLE:
    class PneumoniaModel(nn.Module):
        def __init__(self):
            super(PneumoniaModel, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            self.pool = nn.MaxPool2d(2, 2)
            self.dropout = nn.Dropout(0.5)
            self.fc1 = nn.Linear(128 * 32 * 32, 512)
            self.fc2 = nn.Linear(512, 1)

        def forward(self, x):
            x = self.pool(torch.relu(self.bn1(self.conv1(x))))
            x = self.pool(torch.relu(self.bn2(self.conv2(x))))
            x = self.pool(torch.relu(self.bn3(self.conv3(x))))
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize PyTorch model as fallback
if not use_ollama:
    pneumonia_model = PneumoniaModel()

    # Try to load the model weights if they exist
    try:
        pneumonia_model.load_state_dict(torch.load('models/chest_xray_model.pth', map_location=device))
    except FileNotFoundError:
        print("Warning: chest_xray_model.pth not found. Using untrained model.")
    
    pneumonia_model.to(device)
    pneumonia_model.eval()

def preprocess_image(image_bytes):
    """Image preprocessing with error handling"""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return transform(image)
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

def parse_ollama_response_pneumonia(response_text):
    """Parse Ollama response for pneumonia analysis"""
    response_lower = response_text.lower()
    
    # Determine if pneumonia is detected
    pneumonia_detected = any(keyword in response_lower for keyword in 
                            ['pneumonia', 'infection', 'infiltrate', 'consolidation', 'abnormal', 'positive'])
    
    # Extract confidence percentage
    confidence = 50.0
    
    percentage_match = re.search(r'(\d+(?:\.\d+)?)\s*%', response_text)
    if percentage_match:
        confidence = float(percentage_match.group(1))
    elif 'high' in response_lower or 'certain' in response_lower or 'severe' in response_lower:
        confidence = 85.0
    elif 'moderate' in response_lower or 'likely' in response_lower:
        confidence = 70.0
    elif 'low' in response_lower or 'unlikely' in response_lower:
        confidence = 30.0
    
    confidence = max(0, min(100, confidence))
    
    return pneumonia_detected, confidence, response_text

def predict_pneumonia_ollama(image_bytes):
    """Pneumonia prediction using Ollama llava:7b"""
    try:
        result = ollama_service.analyze_medical_image(image_bytes, analysis_type="pneumonia")
        
        if result.get("success"):
            analysis_text = result.get("analysis", "")
            pneumonia_detected, confidence, full_analysis = parse_ollama_response_pneumonia(analysis_text)
            
            return {
                'prediction': 'PNEUMONIA' if pneumonia_detected else 'NORMAL',
                'confidence': f"{confidence:.2f}%",
                'probability': f"{confidence/100:.3f}",
                'analysis': full_analysis,
                'model': 'ollama_llava:7b'
            }
        else:
            raise ValueError(result.get("error", "Unknown error"))
    
    except Exception as e:
        raise ValueError(f"Error in Ollama pneumonia prediction: {str(e)}")

def predict_pneumonia_pytorch(image_bytes):
    """Pneumonia prediction using PyTorch fallback"""
    try:
        image_tensor = preprocess_image(image_bytes).unsqueeze(0).to(device)
        with torch.no_grad():
            output = pneumonia_model(image_tensor)
            probability = torch.sigmoid(output).item()
            confidence = max(probability, 1 - probability) * 100
            
            result = {
                'prediction': 'PNEUMONIA' if probability > 0.5 else 'NORMAL',
                'confidence': f"{confidence:.2f}%",
                'probability': f"{probability:.3f}",
                'analysis': 'PyTorch model prediction',
                'model': 'pytorch_custom_cnn'
            }
            return result
    except Exception as e:
        raise ValueError(f"Error in PyTorch pneumonia prediction: {str(e)}")

def predict_pneumonia(image_bytes):
    """Pneumonia prediction with Ollama preferred, PyTorch fallback"""
    if use_ollama:
        return predict_pneumonia_ollama(image_bytes)
    else:
        return predict_pneumonia_pytorch(image_bytes) 