"""
2D to 3D Image Reconstruction Module
Converts 2D medical images to 3D volumetric data
"""

import numpy as np
import cv2
from PIL import Image
import io
from skimage import filters, measure, morphology
import json

class ImageReconstruction2Dto3D:
    """Convert 2D medical images to 3D volumetric representations"""
    
    def __init__(self, depth_slices=50):
        """
        Initialize reconstruction engine
        
        Args:
            depth_slices: Number of virtual 3D slices to generate
        """
        self.depth_slices = depth_slices
        
    def image_bytes_to_numpy(self, image_bytes):
        """Convert image bytes to numpy array"""
        image = Image.open(io.BytesIO(image_bytes)).convert('L')  # Convert to grayscale
        return np.array(image, dtype=np.float32) / 255.0
    
    def generate_volume_data(self, image_bytes, analysis_regions=None):
        """
        Generate 3D volume data from 2D image
        
        Args:
            image_bytes: Raw image data
            analysis_regions: List of regions with abnormalities
        
        Returns:
            dict: 3D volume information
        """
        try:
            # Load and preprocess image
            image_2d = self.image_bytes_to_numpy(image_bytes)
            height, width = image_2d.shape
            
            # Create 3D volume by replicating slices with slight variations
            volume = self._create_volume_from_2d(image_2d)
            
            # Extract features
            features = self._extract_3d_features(volume)
            
            # Generate mesh points
            mesh_points = self._generate_mesh_points(volume)
            
            return {
                "success": True,
                "volume": {
                    "shape": volume.shape,
                    "slices": self.depth_slices,
                    "width": width,
                    "height": height,
                    "voxel_size": 1.0
                },
                "features": features,
                "mesh_points": mesh_points,
                "intensity_range": {
                    "min": float(np.min(volume)),
                    "max": float(np.max(volume)),
                    "mean": float(np.mean(volume))
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _create_volume_from_2d(self, image_2d):
        """
        Create 3D volume by extending 2D image into z-dimension
        Simulates MRI/CT-like volumetric data
        """
        height, width = image_2d.shape
        volume = np.zeros((height, width, self.depth_slices), dtype=np.float32)
        
        # Populate slices with slight variations to simulate depth
        for z in range(self.depth_slices):
            # Apply varying blur to create depth effect
            sigma = 1.0 + (z / self.depth_slices) * 2.0
            blurred = filters.gaussian(image_2d, sigma=sigma)
            
            # Fade in/out based on z position for depth perception
            alpha = 1.0 - abs(z - self.depth_slices/2) / (self.depth_slices/2) * 0.5
            
            volume[:, :, z] = blurred * alpha
        
        return volume
    
    def _extract_3d_features(self, volume):
        """Extract 3D features from volume"""
        features = {
            "connected_components": [],
            "surface_area": 0.0,
            "center_of_mass": None
        }
        
        try:
            # Threshold for feature extraction
            threshold = filters.threshold_otsu(volume)
            binary_volume = volume > threshold
            
            # Label connected components
            labeled_volume = measure.label(binary_volume)
            n_components = labeled_volume.max()
            
            # Extract component properties
            for i in range(1, min(n_components + 1, 6)):  # Limit to 5 components
                component_mask = labeled_volume == i
                props = measure.regionprops(component_mask.astype(int))[0]
                
                features["connected_components"].append({
                    "id": i,
                    "volume_voxels": int(props.area),
                    "centroid": list(props.centroid),
                    "bbox": list(props.bbox),
                    "solidity": float(props.solidity)
                })
            
            # Calculate surface area approximation
            features["surface_area"] = float(np.sum(np.gradient(binary_volume.astype(float))) / 2)
            
            # Center of mass
            center = measure.center_of_mass(volume)
            features["center_of_mass"] = list(center) if center else None
            
        except Exception as e:
            features["error"] = str(e)
        
        return features
    
    def _generate_mesh_points(self, volume, sample_rate=5):
        """
        Generate mesh points for 3D visualization
        
        Args:
            volume: 3D numpy array
            sample_rate: Sample every nth point to reduce data
        
        Returns:
            list: 3D point coordinates
        """
        points = []
        
        try:
            # Threshold for edge detection
            threshold = np.mean(volume) + np.std(volume)
            edges = volume > threshold
            
            # Sample points from edges
            z_indices, y_indices, x_indices = np.where(edges[::sample_rate, ::sample_rate, ::sample_rate])
            
            for z, y, x in zip(z_indices, y_indices, x_indices):
                points.append({
                    "x": int(x * sample_rate),
                    "y": int(y * sample_rate),
                    "z": int(z * sample_rate),
                    "intensity": float(volume[z*sample_rate, y*sample_rate, x*sample_rate])
                })
        
        except Exception as e:
            print(f"Error generating mesh points: {e}")
        
        return points
    
    def generate_slice_images(self, volume, num_slices=10):
        """
        Generate 2D slice images from 3D volume for preview
        
        Args:
            volume: 3D numpy array
            num_slices: Number of evenly spaced slices to extract
        
        Returns:
            dict: Slice data as base64 strings
        """
        slices = {}
        
        try:
            total_depth = volume.shape[2]
            slice_indices = np.linspace(0, total_depth-1, num_slices, dtype=int)
            
            for idx, z in enumerate(slice_indices):
                slice_2d = volume[:, :, z]
                # Normalize to 0-255
                slice_normalized = (slice_2d * 255).astype(np.uint8)
                slices[f"slice_{idx}"] = slice_normalized.tolist()
        
        except Exception as e:
            slices["error"] = str(e)
        
        return slices
    
    def create_point_cloud_data(self, volume, threshold_factor=1.2):
        """
        Create point cloud data for 3D visualization
        
        Returns:
            dict: Point cloud in various formats
        """
        try:
            # Generate threshold
            mean_intensity = np.mean(volume)
            std_intensity = np.std(volume)
            threshold = mean_intensity + (threshold_factor * std_intensity)
            
            # Get points above threshold
            points = np.argwhere(volume >= threshold)
            intensities = volume[volume >= threshold]
            
            # Normalize intensities
            intensities_norm = (intensities - intensities.min()) / (intensities.max() - intensities.min() + 1e-8)
            
            return {
                "success": True,
                "point_count": len(points),
                "points": points.tolist()[:1000],  # Limit to 1000 points for performance
                "intensities": intensities_norm.tolist()[:1000],
                "bbox": {
                    "min": [int(np.min(points[:, i])) for i in range(3)],
                    "max": [int(np.max(points[:, i])) for i in range(3)]
                }
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


# Helper functions
def convert_2d_to_3d(image_bytes, depth_slices=50):
    """Helper function to convert 2D medical image to 3D data"""
    reconstructor = ImageReconstruction2Dto3D(depth_slices)
    return reconstructor.generate_volume_data(image_bytes)


def get_point_cloud_from_image(image_bytes, depth_slices=50):
    """Helper function to get point cloud from 2D image"""
    reconstructor = ImageReconstruction2Dto3D(depth_slices)
    volume_data = reconstructor.generate_volume_data(image_bytes)
    
    if volume_data.get("success"):
        volume = reconstructor._create_volume_from_2d(
            reconstructor.image_bytes_to_numpy(image_bytes)
        )
        return reconstructor.create_point_cloud_data(volume)
    
    return volume_data
