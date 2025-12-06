from PIL import Image
import io
import re

# Lazy-loaded globals - avoid slow torch import at module load
TORCH_AVAILABLE = False
ollama_service = None
use_ollama = False
brain_model = None
device = None
transform = None
_initialized = False

def _initialize():
    """Lazy initialization - called on first use."""
    global TORCH_AVAILABLE, ollama_service, use_ollama, brain_model, device, transform, _initialized
    
    if _initialized:
        return
    
    # Try to import and initialize torch
    try:
        import torch
        import torch.nn as nn
        import torchvision.transforms as tv_transforms
        from torchvision import models
        TORCH_AVAILABLE = True
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        transform = tv_transforms.Compose([
            tv_transforms.Resize((224, 224)),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    except ImportError:
        TORCH_AVAILABLE = False
    
    # Initialize Ollama service
    try:
        from ollama_service import OllamaService
        ollama_service = OllamaService(model_name="llava:13b")
        use_ollama = ollama_service.is_available()
        
        if use_ollama:
            print("✓ Ollama service detected. Using llava:13b for brain tumor analysis.")
        else:
            print("✗ Ollama service not available. Using PyTorch fallback.")
            if TORCH_AVAILABLE:
                import torch
                from torchvision import models
                brain_model = models.resnet18(weights=None)
                brain_model.fc = nn.Linear(brain_model.fc.in_features, 1)
                try:
                    brain_model.load_state_dict(torch.load('models/tumor_classification_resnet18.pth', map_location=device))
                except FileNotFoundError:
                    print("Warning: tumor_classification_resnet18.pth not found. Using untrained model.")
                brain_model.to(device)
                brain_model.eval()
    except Exception as e:
        print(f"Warning: Failed to initialize: {e}")
        use_ollama = False
    
    _initialized = True

def preprocess_image(image_bytes):
    """Image preprocessing with error handling"""
    _initialize()  # Lazy load torch if needed
    if transform is None:
        raise RuntimeError("Image transform not available. PyTorch not installed.")
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
    _initialize()  # Lazy load if needed
    if not use_ollama:
        raise RuntimeError("Ollama not available")
    
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
    _initialize()  # Lazy load if needed
    if not TORCH_AVAILABLE or brain_model is None:
        raise RuntimeError("PyTorch not available and Ollama not configured")
    
    try:
        import torch
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
    _initialize()  # Lazy load if needed
    if use_ollama:
        return predict_brain_tumor_ollama(image_bytes)
    else:
        return predict_brain_tumor_pytorch(image_bytes) 