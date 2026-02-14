"""
Base Detector Class for 3D Object Detection

Defines the standard interface that all detectors must implement.
Provides common functionality for training, inference, and evaluation.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
from shapely import points
from shapely import points
import torch
import torch.nn as nn
from dataclasses import dataclass
import logging
from src.utils.voxel_generator import VoxelGenerator


logger = logging.getLogger(__name__)


@dataclass
class Detection3D:
    """
    Standard 3D detection output format.
    
    Attributes:
        box: 3D bounding box [x, y, z, l, w, h, yaw] in LiDAR coordinates
        score: Confidence score [0, 1]
        label: Class index (0=car, 1=truck, etc.)
        label_name: Human-readable class name
        metadata: Additional information (num_points, features, etc.)
    """
    box: np.ndarray  # [7] - x, y, z, length, width, height, yaw
    score: float
    label: int
    label_name: str = ""
    metadata: Optional[Dict] = None
    
    def __post_init__(self):
        """Validate and normalize data."""
        self.box = np.asarray(self.box, dtype=np.float32)
        if self.box.shape != (7,):
            raise ValueError(f"Box must be [7], got shape {self.box.shape}")
        
        self.score = float(np.clip(self.score, 0, 1))
        self.label = int(self.label)
        
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'box': self.box.tolist(),
            'score': self.score,
            'label': self.label,
            'label_name': self.label_name,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Detection3D':
        """Create from dictionary."""
        return cls(
            box=np.array(data['box']),
            score=data['score'],
            label=data['label'],
            label_name=data.get('label_name', ''),
            metadata=data.get('metadata', {})
        )
    
    def iou_3d(self, other: 'Detection3D') -> float:
        """Compute 3D IoU with another detection (simplified BEV IoU)."""
        from .geometry import compute_iou_bev
        return compute_iou_bev(self.box, other.box)


class BaseDetector(ABC, nn.Module):
    """
    Abstract base class for all 3D object detectors.
    
    All detectors must inherit from this and implement the abstract methods.
    Provides common training loop, evaluation, and inference utilities.
    """
    
    # Class names for NuScenes dataset
    CLASS_NAMES = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
        'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ]
    
    def __init__(
        self,
        num_classes: int = 10,
        max_num_points: int = 40000,
        voxel_size: Tuple[float, float, float] = (0.1, 0.1, 0.2),
        point_cloud_range: Tuple[float, ...] = (-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
        **kwargs
    ):
        """
        Initialize base detector.
        
        Args:
            num_classes: Number of object classes
            max_num_points: Maximum number of points to process
            voxel_size: Voxel size for discretization [x, y, z]
            point_cloud_range: Detection range [x_min, y_min, z_min, x_max, y_max, z_max]
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.max_num_points = max_num_points
        self.voxel_size = np.array(voxel_size)
        self.point_cloud_range = np.array(point_cloud_range)
        
        # Training state
        self.training_mode = False
        self.device = torch.device('cpu')
        
        # Performance tracking
        self.inference_times = []

        # Initialize voxel generator with instance parameters
        max_voxels = kwargs.get('max_voxels', (16000, 40000))
        max_points_per_voxel = kwargs.get('max_points_per_voxel', 32)
        
        self.voxel_generator = VoxelGenerator(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_num_points=max_points_per_voxel,
            max_voxels=max_voxels[0] if isinstance(max_voxels, tuple) else max_voxels
        )
    
    @abstractmethod
    def forward(self, batch_dict: Dict) -> Dict:
        """
        Forward pass of the network.
        
        Args:
            batch_dict: Dictionary containing:
                - points: [B, N, C] point cloud features
                - batch_size: int
                - voxels: [M, max_points, C] voxelized points (optional)
                - voxel_coords: [M, 4] voxel coordinates (optional)
                - voxel_num_points: [M] number of points per voxel (optional)
        
        Returns:
            Dictionary containing:
                - pred_boxes: [B, K, 7] predicted boxes
                - pred_scores: [B, K] confidence scores  
                - pred_labels: [B, K] class labels
                - batch_cls_preds: [B, K, num_classes] class probabilities (training)
                - batch_box_preds: [B, K, 7] box predictions (training)
        """
        pass
    
    @abstractmethod
    def get_loss(self, batch_dict: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Compute training loss.
        
        Args:
            batch_dict: Batch data including ground truth
        
        Returns:
            loss: Total loss tensor
            loss_dict: Dictionary with individual loss components
        """
        pass
    
    def preprocess(self, points: np.ndarray) -> Dict:
        """
        Preprocess raw point cloud for inference.
        
        Args:
            points: [N, C] raw points (x, y, z, intensity, ...)
        
        Returns:
            batch_dict: Preprocessed data ready for forward pass
        """
        # Filter by range
        mask = self._filter_by_range(points)
        points = points[mask]
        
        # Limit number of points
        if len(points) > self.max_num_points:
            indices = np.random.choice(len(points), self.max_num_points, replace=False)
            points = points[indices]
        
        # Convert to tensor
        points_tensor = torch.from_numpy(points).float()
        
        batch_dict = {
            'points': points_tensor.unsqueeze(0),  # [1, N, C]
            'batch_size': 1
        }
        
        return batch_dict
    
    def _filter_by_range(self, points: np.ndarray) -> np.ndarray:
        """Filter points by detection range."""
        pc_range = self.point_cloud_range
        mask = (
            (points[:, 0] >= pc_range[0]) & (points[:, 0] <= pc_range[3]) &
            (points[:, 1] >= pc_range[1]) & (points[:, 1] <= pc_range[4]) &
            (points[:, 2] >= pc_range[2]) & (points[:, 2] <= pc_range[5])
        )
        return mask
    
    def postprocess(self, pred_dict: Dict, conf_threshold: float = 0.3) -> List[Detection3D]:
        """
        Convert network predictions to Detection3D objects.
        
        Args:
            pred_dict: Network output dictionary
            conf_threshold: Minimum confidence score
        
        Returns:
            List of Detection3D objects
        """
        boxes = pred_dict['pred_boxes'][0].cpu().numpy()  # [K, 7]
        scores = pred_dict['pred_scores'][0].cpu().numpy()  # [K]
        labels = pred_dict['pred_labels'][0].cpu().numpy()  # [K]
        
        detections = []
        for box, score, label in zip(boxes, scores, labels):
            if score < conf_threshold:
                continue
            
            detection = Detection3D(
                box=box,
                score=float(score),
                label=int(label),
                label_name=self.CLASS_NAMES[int(label)] if int(label) < len(self.CLASS_NAMES) else 'unknown'
            )
            detections.append(detection)
        
        return detections
    
    def voxelize(self, batch_dict):
        """
        Convert point cloud to voxel representation.
        """
        points = batch_dict['points']
    
        # If points is already a tensor, convert to numpy
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
        
        # Ensure points is 2D (N, C)
        if points.ndim == 3:
            points = points.squeeze(0)
        
        # Filter points within point cloud range
        mask = (
            (points[:, 0] >= self.point_cloud_range[0]) & 
            (points[:, 0] <= self.point_cloud_range[3]) & 
            (points[:, 1] >= self.point_cloud_range[1]) & 
            (points[:, 1] <= self.point_cloud_range[4]) & 
            (points[:, 2] >= self.point_cloud_range[2]) & 
            (points[:, 2] <= self.point_cloud_range[5])
        )
        points = points[mask]
    
        # Generate voxels using VoxelGenerator
        voxels, coords, num_points = self.voxel_generator.generate(points)
    
        # Convert to tensors
        batch_dict['voxels'] = torch.from_numpy(voxels).float().to(self.device)
        batch_dict['voxel_coords'] = torch.from_numpy(coords).int().to(self.device)
        batch_dict['voxel_num_points'] = torch.from_numpy(num_points).int().to(self.device)
    
        return batch_dict
    
    def detect(self, points, conf_threshold=0.3):
        """
        Detect objects from point cloud.
    
        Args:
            points: Nx4 array (x, y, z, intensity) or path to .bin file
            conf_threshold: Minimum confidence score
        
        Returns:
            List of Detection3D objects
        """
        # Set model to eval mode
        self.eval()
        
        # 1. Load points if path provided
        if isinstance(points, (str, Path)):
            points = np.fromfile(points, dtype=np.float32).reshape(-1, 5)[:, :4]
        
        # Ensure points is numpy array
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
    
        # 2. Create batch dict with raw points
        batch_dict = {
            'points': points,
            'batch_size': 1
        }
    
        # 3. CRITICAL: Voxelize the points first
        batch_dict = self.voxelize(batch_dict)
    
        # 4. Now forward can work because voxels exist
        with torch.no_grad():
            pred_dict = self.forward(batch_dict)
    
        # 5. Post-process
        detections = self.postprocess(pred_dict, conf_threshold)
    
        return detections
    
    def _load_points(self, path: Union[str, Path]) -> np.ndarray:
        """Load point cloud from file."""
        path = Path(path)
        
        if path.suffix == '.bin':
            # NuScenes/KITTI binary format
            points = np.fromfile(str(path), dtype=np.float32)
            points = points.reshape(-1, 5)  # x, y, z, intensity, ring
            return points[:, :4]  # Keep x, y, z, intensity
        
        elif path.suffix == '.pcd':
            # PCD format
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(str(path))
            points = np.asarray(pcd.points)
            # Add intensity channel (zeros)
            intensity = np.zeros((len(points), 1))
            return np.hstack([points, intensity])
        
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def to(self, device: Union[str, torch.device]):
        """Move model to device."""
        self.device = torch.device(device)
        return super().to(device)
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load model weights from checkpoint."""
        checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        
        self.load_state_dict(new_state_dict, strict=False)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def save_checkpoint(self, save_path: Union[str, Path], epoch: int = 0, **kwargs):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'config': {
                'num_classes': self.num_classes,
                'voxel_size': self.voxel_size.tolist(),
                'point_cloud_range': self.point_cloud_range.tolist()
            }
        }
        checkpoint.update(kwargs)
        
        torch.save(checkpoint, str(save_path))
        logger.info(f"Saved checkpoint to {save_path}")
    
    def get_model_info(self) -> Dict:
        """Get model information and statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_class': self.__class__.__name__,
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'voxel_size': self.voxel_size.tolist(),
            'point_cloud_range': self.point_cloud_range.tolist(),
            'device': str(self.device)
        }