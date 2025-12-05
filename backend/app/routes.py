from flask import Blueprint, request, jsonify
from models.brain_tumor import predict_brain_tumor
from models.pneumonia import predict_pneumonia

api = Blueprint('api', __name__)

@api.route('/api/analyze/brain-tumor', methods=['POST'])
def analyze_brain_tumor():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({'error': 'Invalid file format. Please upload PNG or JPG images.'}), 400
            
        image_bytes = file.read()
        result = predict_brain_tumor(image_bytes)
        
        return jsonify({
            'result': 'Positive' if result['prediction'] == 'Tumor Detected' else 'Negative',
            'confidence': float(result['confidence'].strip('%')) / 100,
            'prediction': result['prediction'],
            'analysis': result.get('analysis', ''),
            'model': result.get('model', 'unknown'),
            'regions': [
                {
                    'x': 30,
                    'y': 40,
                    'width': 20,
                    'height': 15
                }
            ] if result['prediction'] == 'Tumor Detected' else []
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/api/analyze/pneumonia', methods=['POST'])
def analyze_pneumonia():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({'error': 'Invalid file format. Please upload PNG or JPG images.'}), 400
            
        image_bytes = file.read()
        result = predict_pneumonia(image_bytes)
        
        return jsonify({
            'result': 'Positive' if result['prediction'] == 'PNEUMONIA' else 'Negative',
            'confidence': float(result['confidence'].strip('%')) / 100,
            'prediction': result['prediction'],
            'analysis': result.get('analysis', ''),
            'model': result.get('model', 'unknown'),
            'affectedAreas': [
                {
                    'region': 'Lower Right Lung',
                    'severity': 'Moderate'
                },
                {
                    'region': 'Upper Left Lung',
                    'severity': 'Mild'
                }
            ] if result['prediction'] == 'PNEUMONIA' else []
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get information about currently used AI models"""
    return jsonify({
        'models': {
            'brain_tumor': {
                'primary': 'ollama_llava:13b',
                'fallback': 'pytorch_resnet18',
                'description': 'Vision model for MRI brain tumor detection'
            },
            'pneumonia': {
                'primary': 'ollama_llava:7b',
                'fallback': 'pytorch_custom_cnn',
                'description': 'Vision model for chest X-ray pneumonia detection'
            },
            '3d_reconstruction': {
                'model': 'mistral',
                'description': 'Text model for generating 3D volumetric data'
            }
        },
        'backend': 'ollama',
        'status': 'active'
    }) 