"""
Dataset class for loading and processing IceCube neutrino events.

IceCube events are represented as 3D point clouds where each point 
corresponds to a PMT (PhotoMultiplier Tube) hit with features like:
- Position (x, y, z)
- Time
- Charge
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Union
import os
import h5py


# Constants
EPSILON = 1e-6  # Small constant to avoid division by zero


class IceCubeDataset(Dataset):
    """
    Dataset for IceCube neutrino events.
    
    Expected data format:
    - HDF5 files with groups for each event containing:
      - 'positions': (N, 3) array of PMT positions
      - 'features': (N, F) array of features (time, charge, etc.)
      - 'label': integer class label
    
    Args:
        data_path: Path to HDF5 file or directory of HDF5 files
        split: 'train', 'val', or 'test'
        max_points: Maximum number of points per event (for padding/sampling)
        normalize: Whether to normalize spatial coordinates
        augment: Whether to apply data augmentation
        random_seed: Random seed for reproducibility (None for non-deterministic)
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        max_points: int = 5000,
        normalize: bool = True,
        augment: bool = False,
        class_names: Optional[List[str]] = None,
        random_seed: Optional[int] = None
    ):
        super().__init__()
        self.data_path = data_path
        self.split = split
        self.max_points = max_points
        self.normalize = normalize
        self.augment = augment and split == 'train'
        self.random_seed = random_seed
        
        # Default IceCube event classes
        self.class_names = class_names or [
            'track',      # Muon tracks
            'cascade',    # Electromagnetic/hadronic cascades
            'noise',      # Background noise
        ]
        self.num_classes = len(self.class_names)
        
        # Load dataset indices
        self.events = self._load_event_list()
        
    def _load_event_list(self) -> List[Tuple[str, int]]:
        """
        Load list of events from data path.
        
        Returns:
            List of (file_path, event_index) tuples
        """
        events = []
        
        if os.path.isfile(self.data_path):
            # Single HDF5 file
            with h5py.File(self.data_path, 'r') as f:
                if self.split in f:
                    n_events = len(f[self.split])
                    events = [(self.data_path, i) for i in range(n_events)]
                else:
                    # Assume all events in file
                    n_events = len(f.keys())
                    events = [(self.data_path, i) for i in range(n_events)]
        else:
            # Directory of HDF5 files
            for filename in sorted(os.listdir(self.data_path)):
                if filename.endswith('.h5') or filename.endswith('.hdf5'):
                    filepath = os.path.join(self.data_path, filename)
                    with h5py.File(filepath, 'r') as f:
                        if self.split in f:
                            n_events = len(f[self.split])
                            events.extend([(filepath, i) for i in range(n_events)])
        
        return events
    
    def __len__(self) -> int:
        return len(self.events)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single event.
        
        Returns:
            Dictionary containing:
            - 'coord': (N, 3) point coordinates
            - 'feat': (N, F) point features
            - 'label': scalar class label
            - 'event_id': event identifier
        """
        filepath, event_idx = self.events[idx]
        
        with h5py.File(filepath, 'r') as f:
            if self.split in f:
                event_group = f[self.split][str(event_idx)]
            else:
                event_group = f[str(event_idx)]
            
            # Load positions and features
            positions = np.array(event_group['positions'])  # (N, 3)
            features = np.array(event_group['features'])    # (N, F)
            label = int(event_group['label'][()])
        
        # Sample or pad to max_points
        n_points = positions.shape[0]
        if n_points > self.max_points:
            # Random sampling (with seed for reproducibility if set)
            if self.random_seed is not None:
                rng = np.random.RandomState(self.random_seed + idx)
                indices = rng.choice(n_points, self.max_points, replace=False)
            else:
                indices = np.random.choice(n_points, self.max_points, replace=False)
            positions = positions[indices]
            features = features[indices]
        elif n_points < self.max_points:
            # Pad with zeros
            pad_size = self.max_points - n_points
            positions = np.vstack([
                positions,
                np.zeros((pad_size, 3))
            ])
            features = np.vstack([
                features,
                np.zeros((pad_size, features.shape[1]))
            ])
        
        # Normalize coordinates
        if self.normalize:
            positions = self._normalize_positions(positions)
        
        # Apply augmentation
        if self.augment:
            positions, features = self._augment(positions, features)
        
        # Convert to tensors
        coord = torch.from_numpy(positions).float()
        feat = torch.from_numpy(features).float()
        
        return {
            'coord': coord,
            'feat': feat,
            'label': torch.tensor(label, dtype=torch.long),
            'event_id': f"{os.path.basename(filepath)}_{event_idx}"
        }
    
    def _normalize_positions(self, positions: np.ndarray) -> np.ndarray:
        """Normalize positions to unit cube."""
        if positions.shape[0] == 0:
            return positions
        
        # Center at origin
        positions = positions - positions.mean(axis=0)
        # Scale to [-1, 1]
        scale = np.abs(positions).max()
        if scale > 0:
            positions = positions / scale
        return positions
    
    def _augment(
        self, 
        positions: np.ndarray, 
        features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random augmentations."""
        # Random rotation around z-axis (vertical)
        angle = np.random.uniform(0, 2 * np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        positions = positions @ rot_matrix.T
        
        # Random jittering
        positions += np.random.normal(0, 0.01, positions.shape)
        
        # Random feature scaling
        if features.shape[1] > 0:
            scale = np.random.uniform(0.9, 1.1, features.shape[1])
            features = features * scale
        
        return positions, features
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for imbalanced datasets.
        
        Returns:
            Tensor of class weights
        """
        # Count labels
        label_counts = np.zeros(self.num_classes)
        for filepath, event_idx in self.events:
            with h5py.File(filepath, 'r') as f:
                if self.split in f:
                    label = int(f[self.split][str(event_idx)]['label'][()])
                else:
                    label = int(f[str(event_idx)]['label'][()])
                label_counts[label] += 1
        
        # Inverse frequency weighting
        weights = 1.0 / (label_counts + EPSILON)
        weights = weights / weights.sum() * self.num_classes
        
        return torch.from_numpy(weights).float()
