from flask import Blueprint, request, jsonify
import os
import tempfile
from langchain_core.messages import HumanMessage

# Import agent and voice service
try:
    from agent.graph import agent_workflow, LANGGRAPH_AVAILABLE
except ImportError:
    agent_workflow = None
    LANGGRAPH_AVAILABLE = False

try:
    from services.voice_service import VoiceService
    voice_service = VoiceService()
    VOICE_AVAILABLE = True
except Exception as e:
    print(f"[WARNING] Voice service not available: {e}")
    voice_service = None
    VOICE_AVAILABLE = False

# Import medical models
try:
    from models.brain_tumor import predict_brain_tumor
    from models.pneumonia import predict_pneumonia
    MEDICAL_MODELS_AVAILABLE = True
except Exception as e:
    print(f"[WARNING] Medical models not available: {e}")
    MEDICAL_MODELS_AVAILABLE = False

api = Blueprint('api', __name__)

@api.route('/api/chat', methods=['POST'])
def chat():
    """
    Main chat endpoint supporting both text and audio input
    Returns: { "text": "...", "audio": "base64...", "language": "..." }
    """
    try:
        user_text = None
        detected_language = "en"
        audio_mode = False
        
        # Check if audio file was sent
        if 'audio' in request.files:
            audio_mode = True
            audio_file = request.files['audio']
            
            if not VOICE_AVAILABLE or voice_service is None:
                return jsonify({'error': 'Voice processing not available'}), 503
            
            # Save audio to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
                audio_file.save(temp_audio.name)
                temp_audio_path = temp_audio.name
            
            # Transcribe audio
            user_text, detected_language = voice_service.transcribe(temp_audio_path)
            os.unlink(temp_audio_path)
            
            if not user_text:
                return jsonify({'error': 'Could not transcribe audio'}), 400
                
        # Check if text was sent
        elif request.is_json:
            data = request.get_json()
            user_text = data.get('text', '')
            audio_mode = data.get('audioMode', False)
        else:
            return jsonify({'error': 'No text or audio provided'}), 400
        
        if not user_text:
            return jsonify({'error': 'Empty message'}), 400
        
        # Check if agent is available
        if not LANGGRAPH_AVAILABLE or agent_workflow is None:
            return jsonify({'error': 'Agent not available. Install langchain and langgraph.'}), 503
        
        # Invoke agent workflow
        result = agent_workflow.invoke({
            "messages": [HumanMessage(content=user_text)],
            "context": "",
            "language": detected_language
        })
        
        # Extract response text
        response_text = result["messages"][-1].content
        
        # Generate audio if in audio mode
        audio_base64 = None
        if audio_mode and VOICE_AVAILABLE and voice_service:
            audio_base64 = voice_service.speak(response_text, detected_language)
        
        return jsonify({
            'text': response_text,
            'audio': audio_base64,
            'language': detected_language
        })
        
    except Exception as e:
        print(f"Chat Error: {e}")
        return jsonify({'error': str(e)}), 500


# Medical Image Analysis Endpoints
@api.route('/api/medical/brain-tumor', methods=['POST'])
def analyze_brain_tumor():
    """
    Brain tumor detection endpoint
    Accepts: Image file (multipart/form-data)
    Returns: Prediction with confidence
    """
    try:
        if not MEDICAL_MODELS_AVAILABLE:
            return jsonify({'error': 'Medical models not available'}), 503
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({'error': 'Invalid file format. Please upload PNG or JPG images.'}), 400
            
        image_bytes = file.read()
        result = predict_brain_tumor(image_bytes)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Brain tumor analysis error: {e}")
        return jsonify({'error': str(e)}), 500


@api.route('/api/medical/pneumonia', methods=['POST'])
def analyze_pneumonia():
    """
    Pneumonia detection endpoint
    Accepts: Chest X-ray image file (multipart/form-data)
    Returns: Prediction with confidence
    """
    try:
        if not MEDICAL_MODELS_AVAILABLE:
            return jsonify({'error': 'Medical models not available'}), 503
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({'error': 'Invalid file format. Please upload PNG or JPG images.'}), 400
            
        image_bytes = file.read()
        result = predict_pneumonia(image_bytes)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Pneumonia analysis error: {e}")
        return jsonify({'error': str(e)}), 500


@api.route('/api/medical/status', methods=['GET'])
def medical_status():
    """Get status of medical models"""
    return jsonify({
        'medical_models_available': MEDICAL_MODELS_AVAILABLE,
        'models': {
            'brain_tumor': 'Available' if MEDICAL_MODELS_AVAILABLE else 'Not trained',
            'pneumonia': 'Available' if MEDICAL_MODELS_AVAILABLE else 'Not trained'
        },
        'message': 'Run train_models.py to train models' if not MEDICAL_MODELS_AVAILABLE else 'Models ready'
    }) 