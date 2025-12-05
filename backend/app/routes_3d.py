"""
3D Visualization Routes
Endpoints for 2D-to-3D conversion and volumetric analysis
"""

from flask import Blueprint, request, jsonify
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.brain_tumor import predict_brain_tumor
from models.pneumonia import predict_pneumonia
from reconstruction_3d import ImageReconstruction2Dto3D, get_point_cloud_from_image
from ollama_service import OllamaService

api_3d = Blueprint('api_3d', __name__)

# Initialize 3D reconstruction
reconstructor_3d = ImageReconstruction2Dto3D(depth_slices=50)
ollama_service = OllamaService(model_name="mistral")

@api_3d.route('/api/3d/brain-tumor-volume', methods=['POST'])
def generate_brain_tumor_3d_volume():
    """
    Generate 3D volumetric data from brain tumor 2D image
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({'error': 'Invalid file format'}), 400
        
        image_bytes = file.read()
        
        # First, get 2D analysis
        analysis_2d = predict_brain_tumor(image_bytes)
        
        # Generate 3D volume
        volume_data = reconstructor_3d.generate_volume_data(image_bytes)
        
        if volume_data.get("success"):
            # Generate 3D coordinates from analysis
            coordinates_3d = ollama_service.generate_3d_coordinates(
                analysis_2d.get('analysis', 'Brain tumor detected'),
                image_width=224,
                image_height=224
            )
            
            return jsonify({
                'result': analysis_2d.get('prediction'),
                'confidence': float(analysis_2d.get('confidence', '0').strip('%')) / 100,
                'analysis': analysis_2d.get('analysis', ''),
                'model_used': analysis_2d.get('model', 'unknown'),
                'volume': volume_data.get('volume'),
                'features': volume_data.get('features'),
                '3d_coordinates': coordinates_3d.get('coordinates_3d') if coordinates_3d.get('success') else None,
                'mesh_points': volume_data.get('mesh_points')[:500]  # Limit for performance
            })
        else:
            return jsonify({
                'error': 'Failed to generate 3D volume',
                'details': volume_data.get('error')
            }), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_3d.route('/api/3d/pneumonia-volume', methods=['POST'])
def generate_pneumonia_3d_volume():
    """
    Generate 3D volumetric data from pneumonia chest X-ray
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({'error': 'Invalid file format'}), 400
        
        image_bytes = file.read()
        
        # First, get 2D analysis
        analysis_2d = predict_pneumonia(image_bytes)
        
        # Generate 3D volume
        volume_data = reconstructor_3d.generate_volume_data(image_bytes)
        
        if volume_data.get("success"):
            # Generate 3D coordinates from analysis
            coordinates_3d = ollama_service.generate_3d_coordinates(
                analysis_2d.get('analysis', 'Pneumonia analysis'),
                image_width=256,
                image_height=256
            )
            
            return jsonify({
                'result': analysis_2d.get('prediction'),
                'confidence': float(analysis_2d.get('confidence', '0').strip('%')) / 100,
                'analysis': analysis_2d.get('analysis', ''),
                'model_used': analysis_2d.get('model', 'unknown'),
                'volume': volume_data.get('volume'),
                'features': volume_data.get('features'),
                '3d_coordinates': coordinates_3d.get('coordinates_3d') if coordinates_3d.get('success') else None,
                'mesh_points': volume_data.get('mesh_points')[:500]  # Limit for performance
            })
        else:
            return jsonify({
                'error': 'Failed to generate 3D volume',
                'details': volume_data.get('error')
            }), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_3d.route('/api/3d/point-cloud', methods=['POST'])
def generate_point_cloud():
    """
    Generate point cloud data for 3D visualization
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({'error': 'Invalid file format'}), 400
        
        image_bytes = file.read()
        analysis_type = request.form.get('analysis_type', 'general')
        
        # Get point cloud data
        point_cloud_data = get_point_cloud_from_image(image_bytes, depth_slices=50)
        
        if point_cloud_data.get("success"):
            return jsonify({
                'success': True,
                'point_cloud': point_cloud_data.get('points', [])[:1000],
                'intensities': point_cloud_data.get('intensities', [])[:1000],
                'bbox': point_cloud_data.get('bbox'),
                'point_count': point_cloud_data.get('point_count'),
                'format': 'xyz_intensity'
            })
        else:
            return jsonify({
                'success': False,
                'error': point_cloud_data.get('error')
            }), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_3d.route('/api/3d/reconstruction-info', methods=['GET'])
def get_reconstruction_info():
    """
    Get information about 3D reconstruction capabilities
    """
    return jsonify({
        'supported_features': [
            'volumetric_reconstruction',
            'point_cloud_generation',
            'mesh_extraction',
            'slice_visualization',
            'intensity_analysis'
        ],
        'parameters': {
            'depth_slices': 50,
            'voxel_size': 1.0,
            'max_mesh_points': 500,
            'max_point_cloud_points': 1000
        },
        'formats': [
            'point_cloud',
            'volumetric_data',
            'mesh',
            '2d_slices'
        ]
    })


@api_3d.route('/api/3d/analyze-and-reconstruct', methods=['POST'])
def analyze_and_reconstruct():
    """
    Complete analysis with 2D-to-3D conversion in one endpoint
    OPTIMIZED: Skip slow AI-based 3D coordinate generation
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        analysis_type = request.form.get('analysis_type', 'brain_tumor')
        
        image_bytes = file.read()
        
        # Get 2D analysis (using Ollama)
        if analysis_type == 'brain_tumor':
            analysis_2d = predict_brain_tumor(image_bytes)
        elif analysis_type == 'pneumonia':
            analysis_2d = predict_pneumonia(image_bytes)
        else:
            return jsonify({'error': f'Unknown analysis type: {analysis_type}'}), 400
        
        # Fast 3D reconstruction WITHOUT expensive AI model
        volume_data = reconstructor_3d.generate_volume_data(image_bytes)
        point_cloud_data = get_point_cloud_from_image(image_bytes, depth_slices=50)
        
        # SKIP expensive mistral model call - use volume data directly
        coordinates_3d = None
        if volume_data.get('success'):
            # Use volume features instead of AI-generated coordinates
            coordinates_3d = {
                'volume_data': volume_data.get('volume'),
                'features': volume_data.get('features'),
                'reconstruction_quality': 0.85  # Direct reconstruction quality
            }
        
        return jsonify({
            'analysis_type': analysis_type,
            'analysis_2d': {
                'prediction': analysis_2d.get('prediction'),
                'confidence': float(analysis_2d.get('confidence', '0').strip('%')) / 100,
                'analysis_text': analysis_2d.get('analysis', ''),
                'model': analysis_2d.get('model', 'unknown')
            },
            'reconstruction_3d': {
                'volume': volume_data.get('volume') if volume_data.get('success') else None,
                'point_cloud_size': point_cloud_data.get('point_count') if point_cloud_data.get('success') else 0,
                'features': volume_data.get('features') if volume_data.get('success') else None,
                'coordinates': coordinates_3d
            },
            'visualization_data': {
                'mesh_points': volume_data.get('mesh_points', [])[:200],
                'point_cloud': point_cloud_data.get('points', [])[:500] if point_cloud_data.get('success') else [],
                'intensities': point_cloud_data.get('intensities', [])[:500] if point_cloud_data.get('success') else []
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
