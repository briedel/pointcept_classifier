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
import pyarrow.parquet as pq


# Constants
EPSILON = 1e-6  # Small constant to avoid division by zero


class IceCubeDataset(Dataset):
    """
    Dataset for IceCube neutrino events.
    
    Expected data formats:
    
    HDF5 format:
    - HDF5 files with groups for each event containing:
      - 'positions': (N, 3) array of PMT positions
      - 'features': (N, F) array of features (time, charge, etc.)
      - 'label': integer class label
    
    Parquet format:
    - Parquet files with columns:
      - 'event_id': event identifier
      - 'x', 'y', 'z': PMT positions
      - feature columns (e.g., 'time', 'charge')
      - 'label': integer class label
    
    Args:
        data_path: Path to HDF5/Parquet file or directory of files
        split: 'train', 'val', or 'test'
        max_points: Maximum number of points per event (for padding/sampling)
        normalize: Whether to normalize spatial coordinates
        augment: Whether to apply data augmentation
        random_seed: Random seed for reproducibility (None for non-deterministic)
        file_format: 'hdf5' or 'parquet' (auto-detected if None)
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        max_points: int = 5000,
        normalize: bool = True,
        augment: bool = False,
        class_names: Optional[List[str]] = None,
        random_seed: Optional[int] = None,
        file_format: Optional[str] = None
    ):
        super().__init__()
        self.data_path = data_path
        self.split = split
        self.max_points = max_points
        self.normalize = normalize
        self.augment = augment and split == 'train'
        self.random_seed = random_seed
        
        # Auto-detect file format if not specified
        if file_format is None:
            if os.path.isfile(data_path):
                if data_path.endswith('.parquet'):
                    self.file_format = 'parquet'
                else:
                    self.file_format = 'hdf5'
            else:
                # Check directory for file types
                files = os.listdir(data_path)
                if any(f.endswith('.parquet') for f in files):
                    self.file_format = 'parquet'
                else:
                    self.file_format = 'hdf5'
        else:
            self.file_format = file_format.lower()
        
        # Default IceCube event classes
        self.class_names = class_names or [
            'track',      # Muon tracks
            'cascade',    # Electromagnetic/hadronic cascades
            'noise',      # Background noise
        ]
        self.num_classes = len(self.class_names)
        
        # Load dataset indices
        if self.file_format == 'parquet':
            self.events = self._load_parquet_event_list()
        else:
            self.events = self._load_hdf5_event_list()
        
    def _load_hdf5_event_list(self) -> List[Tuple[str, int]]:
        """
        Load list of events from HDF5 data path.
        
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
    
    def _load_parquet_event_list(self) -> List[Tuple[str, Union[int, str]]]:
        """
        Load list of events from Parquet data path.
        
        Returns:
            List of (file_path, event_id) tuples
        """
        events = []
        
        if os.path.isfile(self.data_path):
            # Single Parquet file
            table = pq.read_table(self.data_path)
            df = table.to_pandas()
            
            # Filter by split if 'split' column exists
            if 'split' in df.columns:
                df = df[df['split'] == self.split]
            
            # Get unique event IDs
            if 'event_id' in df.columns:
                event_ids = df['event_id'].unique()
                events = [(self.data_path, eid) for eid in event_ids]
            else:
                # Use row groups if no event_id column
                n_row_groups = pq.ParquetFile(self.data_path).num_row_groups
                events = [(self.data_path, i) for i in range(n_row_groups)]
        else:
            # Directory of Parquet files
            for filename in sorted(os.listdir(self.data_path)):
                if filename.endswith('.parquet'):
                    filepath = os.path.join(self.data_path, filename)
                    table = pq.read_table(filepath)
                    df = table.to_pandas()
                    
                    # Filter by split if 'split' column exists
                    if 'split' in df.columns:
                        df = df[df['split'] == self.split]
                    
                    # Get unique event IDs
                    if 'event_id' in df.columns:
                        event_ids = df['event_id'].unique()
                        events.extend([(filepath, eid) for eid in event_ids])
        
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
        
        if self.file_format == 'parquet':
            positions, features, label = self._load_parquet_event(filepath, event_idx)
        else:
            positions, features, label = self._load_hdf5_event(filepath, event_idx)
        
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
    
    def _load_hdf5_event(self, filepath: str, event_idx: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """Load a single event from HDF5 file."""
        with h5py.File(filepath, 'r') as f:
            if self.split in f:
                event_group = f[self.split][str(event_idx)]
            else:
                event_group = f[str(event_idx)]
            
            # Load positions and features
            positions = np.array(event_group['positions'])  # (N, 3)
            features = np.array(event_group['features'])    # (N, F)
            label = int(event_group['label'][()])
        
        return positions, features, label
    
    def _load_parquet_event(self, filepath: str, event_id: Union[int, str]) -> Tuple[np.ndarray, np.ndarray, int]:
        """Load a single event from Parquet file."""
        table = pq.read_table(filepath)
        df = table.to_pandas()
        
        # Filter for this event
        if 'event_id' in df.columns:
            event_df = df[df['event_id'] == event_id]
        else:
            # If no event_id, assume event_id is row group index
            # This is a fallback - not recommended for production
            event_df = df.iloc[event_id:event_id+1]
        
        # Extract positions (x, y, z columns)
        if all(col in event_df.columns for col in ['x', 'y', 'z']):
            positions = event_df[['x', 'y', 'z']].values
        elif 'positions' in event_df.columns:
            # If positions stored as array column
            positions = np.array(event_df['positions'].iloc[0])
        else:
            raise ValueError("Parquet file must contain 'x', 'y', 'z' columns or 'positions' column")
        
        # Extract features (all numeric columns except positions and label)
        exclude_cols = {'x', 'y', 'z', 'positions', 'label', 'event_id', 'split'}
        feature_cols = [col for col in event_df.columns 
                       if col not in exclude_cols and event_df[col].dtype in [np.float32, np.float64, np.int32, np.int64]]
        
        if feature_cols:
            features = event_df[feature_cols].values
        elif 'features' in event_df.columns:
            # If features stored as array column
            features = np.array(event_df['features'].iloc[0])
        else:
            # If no features, create dummy feature
            features = np.ones((positions.shape[0], 1))
        
        # Extract label
        if 'label' in event_df.columns:
            label = int(event_df['label'].iloc[0])
        else:
            label = 0  # Default label if not present
        
        return positions, features, label
    
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
        
        if self.file_format == 'parquet':
            for filepath, event_id in self.events:
                _, _, label = self._load_parquet_event(filepath, event_id)
                label_counts[label] += 1
        else:
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
