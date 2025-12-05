"""
Ollama Service Module
Handles integration with Ollama for AI-powered medical image analysis
"""

import requests
import base64
import json
from PIL import Image
import io

OLLAMA_API_URL = "http://localhost:11434/api"

class OllamaService:
    """Service class for Ollama API interactions"""
    
    def __init__(self, model_name="llava:13b"):
        self.model_name = model_name
        self.api_url = OLLAMA_API_URL
        
    def is_available(self):
        """Check if Ollama service is running"""
        try:
            response = requests.get(f"{self.api_url}/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def image_to_base64(self, image_bytes):
        """Convert image bytes to base64 string"""
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def analyze_medical_image(self, image_bytes, analysis_type="general"):
        """
        Analyze medical image using Ollama vision model
        
        Args:
            image_bytes: Raw image data
            analysis_type: "brain_tumor", "pneumonia", or "general"
        
        Returns:
            dict: Analysis results with prediction and description
        """
        try:
            image_base64 = self.image_to_base64(image_bytes)
            
            # Define prompts based on analysis type
            prompts = {
                "brain_tumor": "Analyze this MRI scan for brain tumors. Describe any abnormalities, their location, size estimate, and severity (mild/moderate/severe). Also provide confidence level (0-100%).",
                "pneumonia": "Analyze this chest X-ray for signs of pneumonia. Describe affected areas, severity, and confidence level (0-100%). List which lungs or regions are affected.",
                "general": "Analyze this medical image and provide detailed observations about any abnormalities or areas of concern."
            }
            
            prompt = prompts.get(analysis_type, prompts["general"])
            
            # Call Ollama API with vision model
            response = requests.post(
                f"{self.api_url}/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "images": [image_base64],
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "analysis": result.get("response", ""),
                    "model": self.model_name
                }
            else:
                return {
                    "success": False,
                    "error": f"Ollama API error: {response.status_code}",
                    "analysis": ""
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "analysis": ""
            }
    
    def generate_3d_coordinates(self, analysis_description, image_width=224, image_height=224):
        """
        Generate 3D coordinates and volumetric data from analysis
        Uses text model to generate structured 3D data
        
        Args:
            analysis_description: Text description from image analysis
            image_width: Width of original 2D image
            image_height: Height of original 2D image
        
        Returns:
            dict: 3D coordinate data and volume information
        """
        try:
            prompt = f"""Based on this medical image analysis, generate 3D reconstruction data in JSON format.
Analysis: {analysis_description}

Image dimensions: {image_width}x{image_height}

Generate a JSON response with this structure:
{{
    "volume_data": {{
        "depth": 50,
        "slices": [slice_z_positions_0_to_50],
        "voxel_size": 1.0
    }},
    "regions_of_interest": [
        {{
            "name": "region_name",
            "center": {{"x": x_percent, "y": y_percent, "z": z_percent}},
            "dimensions": {{"width": w_mm, "height": h_mm, "depth": d_mm}},
            "intensity": 0.0_to_1.0
        }}
    ],
    "reconstruction_quality": 0.0_to_1.0,
    "estimated_volume_mm3": number
}}

Only output valid JSON, no other text."""
            
            response = requests.post(
                f"{self.api_url}/generate",
                json={
                    "model": "mistral",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                
                try:
                    # Try to parse JSON from response
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = response_text[json_start:json_end]
                        coordinates = json.loads(json_str)
                        return {
                            "success": True,
                            "coordinates_3d": coordinates,
                            "model": "mistral"
                        }
                except:
                    pass
                
                # If JSON parsing fails, return text response
                return {
                    "success": True,
                    "coordinates_3d": {"raw_analysis": response_text},
                    "model": "mistral"
                }
            else:
                return {
                    "success": False,
                    "error": f"Ollama API error: {response.status_code}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def generate_3d_visualization_data(self, image_bytes, analysis_description):
        """
        Generate complete 3D visualization data including point cloud
        
        Args:
            image_bytes: Raw image data
            analysis_description: Analysis text
        
        Returns:
            dict: 3D visualization data
        """
        try:
            # Convert image to get dimensions
            img = Image.open(io.BytesIO(image_bytes))
            width, height = img.size
            
            # Get 3D coordinates
            coordinates = self.generate_3d_coordinates(analysis_description, width, height)
            
            if coordinates.get("success"):
                return {
                    "success": True,
                    "image_dimensions": {"width": width, "height": height},
                    "coordinates_3d": coordinates.get("coordinates_3d"),
                    "visualization_format": "point_cloud_with_volume"
                }
            else:
                return coordinates
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


# Helper functions
def get_ollama_service(model_name="llava:13b"):
    """Factory function to get Ollama service instance"""
    return OllamaService(model_name)


def is_ollama_ready():
    """Check if Ollama is ready to use"""
    service = OllamaService()
    return service.is_available()
