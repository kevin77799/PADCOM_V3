from PIL import Image
import io
import re

# Lazy-loaded globals - avoid slow torch import at module load
TORCH_AVAILABLE = False
ollama_service = None
use_ollama = False
pneumonia_model = None
device = None
transform = None
_initialized = False
PneumoniaModel = None

def _initialize():
    """Lazy initialization - called on first use."""
    global TORCH_AVAILABLE, ollama_service, use_ollama, pneumonia_model, device, transform, _initialized, PneumoniaModel
    
    if _initialized:
        return
    
    # Try to import and initialize torch
    try:
        import torch
        import torch.nn as nn
        import torchvision.transforms as tv_transforms
        TORCH_AVAILABLE = True
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        transform = tv_transforms.Compose([
            tv_transforms.Grayscale(num_output_channels=1),
            tv_transforms.Resize((256, 256)),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485], std=[0.229])
        ])
        
        # Define PneumoniaModel class
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
                x = torch.sigmoid(self.fc2(x))
                return x
    except ImportError:
        TORCH_AVAILABLE = False
    
    # Initialize Ollama service
    try:
        from ollama_service import OllamaService
        ollama_service = OllamaService(model_name="llava:7b")
        use_ollama = ollama_service.is_available()
        
        if use_ollama:
            print("✓ Ollama service detected. Using llava:7b for pneumonia analysis.")
        else:
            print("✗ Ollama service not available. Using PyTorch fallback.")
            if TORCH_AVAILABLE:
                pneumonia_model = PneumoniaModel()
                try:
                    import torch
                    pneumonia_model.load_state_dict(torch.load('models/pneumonia_classification.pth', map_location=device))
                except FileNotFoundError:
                    print("Warning: pneumonia_classification.pth not found. Using untrained model.")
                pneumonia_model.to(device)
                pneumonia_model.eval()
    except Exception as e:
        print(f"Warning: Failed to initialize: {e}")
        use_ollama = False
    
    _initialized = True


def preprocess_image(image_bytes):
    """Image preprocessing with error handling"""
    _initialize()
    if transform is None:
        raise RuntimeError("Image transform not available. PyTorch not installed.")
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        return transform(image)
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")


def parse_ollama_response(response_text):
    """Parse Ollama response to extract prediction and confidence"""
    response_lower = response_text.lower()
    
    pneumonia_detected = any(keyword in response_lower for keyword in 
                            ['pneumonia detected', 'pneumonia present', 'pneumonia', 'detected', 'positive', 'abnormality'])
    
    confidence = 50.0
    
    percentage_match = re.search(r'(\d+(?:\.\d+)?)\s*%', response_text)
    if percentage_match:
        confidence = float(percentage_match.group(1))
    elif 'high' in response_lower or 'certain' in response_lower:
        confidence = 85.0
    elif 'moderate' in response_lower or 'likely' in response_lower:
        confidence = 70.0
    elif 'low' in response_lower or 'unlikely' in response_lower:
        confidence = 30.0
    
    confidence = max(0, min(100, confidence))
    
    return pneumonia_detected, confidence, response_text


def predict_pneumonia_ollama(image_bytes):
    """Pneumonia prediction using Ollama llava:7b"""
    _initialize()
    if not use_ollama:
        raise RuntimeError("Ollama not available")
    
    try:
        result = ollama_service.analyze_medical_image(image_bytes, analysis_type="pneumonia")
        
        if result.get("success"):
            analysis_text = result.get("analysis", "")
            pneumonia_detected, confidence, full_analysis = parse_ollama_response(analysis_text)
            
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
    _initialize()
    if not TORCH_AVAILABLE or pneumonia_model is None:
        raise RuntimeError("PyTorch not available and Ollama not configured")
    
    try:
        import torch
        image_tensor = preprocess_image(image_bytes).unsqueeze(0).to(device)
        with torch.no_grad():
            output = pneumonia_model(image_tensor)
            probability = output.item()
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
    _initialize()
    if use_ollama:
        return predict_pneumonia_ollama(image_bytes)
    else:
        return predict_pneumonia_pytorch(image_bytes) 