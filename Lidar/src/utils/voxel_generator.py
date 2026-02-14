"""
Complete PointPillars Voxelization Implementation
This file shows the correct way to implement voxelization for PointPillars.
"""

import numpy as np
from shapely import points
import torch
from pathlib import Path


class VoxelGenerator:
    """
    Converts point cloud to voxel representation.
    This is the key component that was missing in the detect() method.
    """
    
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        """
        Args:
            voxel_size: [vx, vy, vz] size of each voxel
            point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
            max_num_points: Maximum points per voxel
            max_voxels: Maximum number of voxels
        """
        self.voxel_size = np.array(voxel_size)
        self.point_cloud_range = np.array(point_cloud_range)
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels
        
        # Calculate grid size
        self.grid_size = np.round(
            (self.point_cloud_range[3:] - self.point_cloud_range[:3]) / self.voxel_size
        ).astype(np.int32)
        
        print(f"VoxelGenerator initialized: grid_size={self.grid_size}")
    
    def generate(self, points):
        """
        Convert points to voxels.
        
        Args:
            points: Nx4 array (x, y, z, intensity)
            
        Returns:
            voxels: (num_voxels, max_num_points, 4)
            coords: (num_voxels, 3) - voxel coordinates (z, y, x)
            num_points: (num_voxels,) - number of points in each voxel
        """
        # 1. Filter points within range
        mask = (
            (points[:, 0] >= self.point_cloud_range[0]) &
            (points[:, 0] < self.point_cloud_range[3]) &
            (points[:, 1] >= self.point_cloud_range[1]) &
            (points[:, 1] < self.point_cloud_range[4]) &
            (points[:, 2] >= self.point_cloud_range[2]) &
            (points[:, 2] < self.point_cloud_range[5])
        )
        points = points[mask]
        
        # 2. Compute voxel coordinates
        voxel_coords = np.floor(
            (points[:, :3] - self.point_cloud_range[:3]) / self.voxel_size
        ).astype(np.int32)
        
        # 3. Group points by voxel
        voxel_dict = {}
        for i, coord in enumerate(voxel_coords):
            key = (coord[2], coord[1], coord[0])  # (z, y, x) format for consistency
            
            if key not in voxel_dict:
                voxel_dict[key] = []
            
            if len(voxel_dict[key]) < self.max_num_points:
                voxel_dict[key].append(points[i])
        
        # 4. Convert to arrays
        num_voxels = min(len(voxel_dict), self.max_voxels)
        voxels = np.zeros((num_voxels, self.max_num_points, 4), dtype=np.float32)
        coords = np.zeros((num_voxels, 3), dtype=np.int32)
        num_points = np.zeros(num_voxels, dtype=np.int32)
        
        for i, (coord, point_list) in enumerate(list(voxel_dict.items())[:num_voxels]):
            num_pts = len(point_list)
            voxels[i, :num_pts] = point_list
            coords[i] = coord
            num_points[i] = num_pts
        
        return voxels, coords, num_points


class PointPillarsDetector:
    """
    Simplified PointPillars detector showing the correct voxelization flow.
    """
    
    def __init__(self, config):
        # Model configuration
        self.voxel_size = config.get('voxel_size', [0.16, 0.16, 4.0])
        self.point_cloud_range = config.get('point_cloud_range', [0, -39.68, -3, 69.12, 39.68, 1])
        self.max_num_points = config.get('max_num_points', 32)
        self.max_voxels = config.get('max_voxels', (16000, 40000))
        
        # Initialize voxel generator
        self.voxel_generator = VoxelGenerator(
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range,
            max_num_points=self.max_num_points,
            max_voxels=self.max_voxels[0]  # Use training max_voxels
        )
        
        # Initialize model components (VFE, backbone, head)
        # ... (your existing model initialization)
        
    def voxelize(self, batch_dict):
        points = batch_dict['points']
    
        # Handle tensor input
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
    
        # Handle batch dimension
        if points.ndim == 3:
            points = points.squeeze(0)
    
        # ✅ Correct indices for point cloud range
        mask = (
            (points[:, 0] >= self.point_cloud_range[0]) & 
        (points[:, 0] <= self.point_cloud_range[3]) & 
        (points[:, 1] >= self.point_cloud_range[1]) & 
        (points[:, 1] <= self.point_cloud_range[4]) &  # ✅ Correct Y_max
        (points[:, 2] >= self.point_cloud_range[2]) & 
        (points[:, 2] <= self.point_cloud_range[5])
            )
        points = points[mask]
    
        voxels, coords, num_points = self.voxel_generator.generate(points)
    
        # ✅ Use self.device instead of hardcoded cuda()
        batch_dict['voxels'] = torch.from_numpy(voxels).float().to(self.device)
        batch_dict['voxel_coords'] = torch.from_numpy(coords).int().to(self.device)
        batch_dict['voxel_num_points'] = torch.from_numpy(num_points).int().to(self.device)
    
        return batch_dict
    
    def forward(self, batch_dict):
        """
        Forward pass through the network.
        Now this can safely access batch_dict['voxels'].
        """
        # 1. VFE (Voxel Feature Encoder)
        voxel_features = self.vfe(
            batch_dict['voxels'],
            batch_dict['voxel_num_points']
        )
        
        # 2. Create pseudo-image from voxel features
        batch_dict['pillar_features'] = self.scatter(
            voxel_features,
            batch_dict['voxel_coords'],
            batch_dict['batch_size']
        )
        
        # 3. Backbone (2D CNN)
        spatial_features = self.backbone(batch_dict['pillar_features'])
        
        # 4. Detection head
        batch_dict['spatial_features'] = spatial_features
        batch_dict = self.head(batch_dict)
        
        return batch_dict
    
    def detect(self, points, conf_threshold=0.3):
        # ✅ Set model to eval mode
        self.eval()
    
        if isinstance(points, (str, Path)):
            points = np.fromfile(points, dtype=np.float32).reshape(-1, 5)[:, :4]
    
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
    
        batch_dict = {
            'points': points,
            'batch_size': 1
            }
    
        batch_dict = self.voxelize(batch_dict)
    
        # ✅ Use no_grad context
        with torch.no_grad():
            pred_dict = self.forward(batch_dict)
    
        detections = self.postprocess(pred_dict, conf_threshold)
    
        return detections
    
    def post_process(self, batch_dict, conf_threshold):
        """
        Convert raw predictions to detection boxes.
        """
        # Extract predictions from batch_dict
        # (This depends on your specific head implementation)
        cls_preds = batch_dict.get('cls_preds')
        box_preds = batch_dict.get('box_preds')
        
        # Apply NMS and confidence filtering
        # ... (your post-processing logic)
        
        detections = []
        # ... populate detections list
        
        return detections


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def test_voxelization():
    """
    Example showing the correct usage.
    """
    # Configuration
    config = {
        'voxel_size': [0.16, 0.16, 4.0],
        'point_cloud_range': [0, -39.68, -3, 69.12, 39.68, 1],
        'max_num_points': 32,
        'max_voxels': (16000, 40000)
    }
    
    # Initialize detector
    model = PointPillarsDetector(config)
    
    # Load point cloud
    points_path = "path/to/pointcloud.bin"
    points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 5)[:, :4]
    
    # Detect objects
    detections = model.detect(points, conf_threshold=0.3)
    
    print(f"✅ Detection complete: {len(detections)} objects found")


if __name__ == "__main__":
    test_voxelization()