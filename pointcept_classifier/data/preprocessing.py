"""
Preprocessing utilities for IceCube events.
"""

import numpy as np
from typing import Dict, Tuple, Optional


def preprocess_event(
    positions: np.ndarray,
    features: np.ndarray,
    normalize: bool = True,
    remove_noise: bool = False,
    noise_threshold: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess a single IceCube event.

    Args:
        positions: (N, 3) array of PMT positions
        features: (N, F) array of features
        normalize: Whether to normalize coordinates
        remove_noise: Whether to remove low-charge hits
        noise_threshold: Charge threshold for noise removal

    Returns:
        Preprocessed positions and features
    """
    # Remove noise hits (low charge)
    if remove_noise and features.shape[1] > 1:
        # Assuming second feature is charge
        charge = features[:, 1]
        mask = charge > noise_threshold
        positions = positions[mask]
        features = features[mask]

    # Normalize coordinates
    if normalize:
        positions = normalize_coordinates(positions)

    # Normalize features
    features = normalize_features(features)

    return positions, features


def normalize_coordinates(positions: np.ndarray) -> np.ndarray:
    """
    Normalize 3D coordinates to unit cube centered at origin.

    Args:
        positions: (N, 3) array of positions

    Returns:
        Normalized positions
    """
    if positions.shape[0] == 0:
        return positions

    # Center at origin
    center = positions.mean(axis=0)
    positions = positions - center

    # Scale to [-1, 1]
    max_coord = np.abs(positions).max()
    if max_coord > 0:
        positions = positions / max_coord

    return positions


def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    Normalize features using z-score normalization.

    Args:
        features: (N, F) array of features

    Returns:
        Normalized features
    """
    if features.shape[0] == 0:
        return features

    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-6
    features = (features - mean) / std

    return features


def icecube_to_pointcloud(
    event_data: Dict, feature_names: Optional[list] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert IceCube event data to point cloud format.

    Args:
        event_data: Dictionary with event information
        feature_names: List of feature names to extract

    Returns:
        positions and features arrays
    """
    # Default features
    if feature_names is None:
        feature_names = ["time", "charge"]

    # Extract positions
    if "dom_x" in event_data and "dom_y" in event_data and "dom_z" in event_data:
        positions = np.column_stack(
            [event_data["dom_x"], event_data["dom_y"], event_data["dom_z"]]
        )
    elif "positions" in event_data:
        positions = event_data["positions"]
    else:
        raise ValueError("Event data must contain position information")

    # Extract features
    features_list = []
    for feat_name in feature_names:
        if feat_name in event_data:
            features_list.append(event_data[feat_name])

    if features_list:
        features = np.column_stack(features_list)
    else:
        # If no features, use ones as placeholder
        features = np.ones((positions.shape[0], 1))

    return positions, features


def split_dataset(
    n_events: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset indices into train/val/test.

    Args:
        n_events: Total number of events
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        random_seed: Random seed for reproducibility

    Returns:
        train_indices, val_indices, test_indices
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    np.random.seed(random_seed)
    indices = np.random.permutation(n_events)

    n_train = int(n_events * train_ratio)
    n_val = int(n_events * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    return train_idx, val_idx, test_idx
