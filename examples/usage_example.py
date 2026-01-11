"""
Example usage of the IceCube event classifier.

This script demonstrates how to use the classifier API programmatically.
"""

import torch
import numpy as np
from pathlib import Path

from pointcept_classifier.data import IceCubeDataset
from pointcept_classifier.models import build_pointcept_classifier
from pointcept_classifier.utils import load_config


def example_training():
    """Example: Training a model programmatically."""
    print("Example 1: Training a model\n")
    
    # Create dataset
    dataset = IceCubeDataset(
        data_path="synthetic_icecube_data.h5",
        split='train',
        max_points=5000,
        normalize=True,
        augment=True,
        class_names=['track', 'cascade', 'noise']
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Class names: {dataset.class_names}")
    
    # Get a sample
    sample = dataset[0]
    print(f"\nSample data:")
    print(f"  Coordinates shape: {sample['coord'].shape}")
    print(f"  Features shape: {sample['feat'].shape}")
    print(f"  Label: {sample['label']}")
    print(f"  Event ID: {sample['event_id']}")
    
    # Create model
    model = build_pointcept_classifier(
        model_name='simple',
        num_classes=3,
        in_channels=2,
        hidden_dim=256
    )
    
    print(f"\nModel created successfully")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


def example_inference():
    """Example: Running inference on a trained model."""
    print("\nExample 2: Running inference\n")
    
    # Create dataset
    dataset = IceCubeDataset(
        data_path="synthetic_icecube_data.h5",
        split='test',
        max_points=5000,
        normalize=True,
        augment=False,
        class_names=['track', 'cascade', 'noise']
    )
    
    # Create model
    model = build_pointcept_classifier(
        model_name='simple',
        num_classes=3,
        in_channels=2
    )
    model.eval()
    
    # Get a sample
    sample = dataset[0]
    coord = sample['coord'].unsqueeze(0)  # Add batch dimension
    feat = sample['feat'].unsqueeze(0)
    
    # Run inference
    with torch.no_grad():
        # Get predictions
        predictions = model.predict(coord, feat)
        probabilities = model.predict_proba(coord, feat)
    
    print("Inference results:")
    print(f"  Predicted class: {predictions.item()}")
    print(f"  Class name: {dataset.class_names[predictions.item()]}")
    print(f"  Probabilities: {probabilities.squeeze().numpy()}")
    print(f"  True label: {sample['label'].item()}")


def example_custom_preprocessing():
    """Example: Custom preprocessing of IceCube events."""
    print("\nExample 3: Custom preprocessing\n")
    
    from pointcept_classifier.data import (
        preprocess_event,
        normalize_coordinates,
        icecube_to_pointcloud
    )
    
    # Simulate raw event data
    raw_data = {
        'dom_x': np.random.randn(1000) * 100,
        'dom_y': np.random.randn(1000) * 100,
        'dom_z': np.random.randn(1000) * 200,
        'time': np.random.uniform(0, 1000, 1000),
        'charge': np.random.exponential(10, 1000)
    }
    
    # Convert to point cloud format
    positions, features = icecube_to_pointcloud(
        raw_data,
        feature_names=['time', 'charge']
    )
    
    print(f"Raw data converted:")
    print(f"  Positions shape: {positions.shape}")
    print(f"  Features shape: {features.shape}")
    
    # Preprocess
    positions_norm, features_norm = preprocess_event(
        positions,
        features,
        normalize=True,
        remove_noise=True,
        noise_threshold=1.0
    )
    
    print(f"\nAfter preprocessing:")
    print(f"  Positions shape: {positions_norm.shape}")
    print(f"  Features shape: {features_norm.shape}")
    print(f"  Position range: [{positions_norm.min():.2f}, {positions_norm.max():.2f}]")


def example_batch_prediction():
    """Example: Batch prediction on multiple events."""
    print("\nExample 4: Batch prediction\n")
    
    from torch.utils.data import DataLoader
    
    # Create dataset
    dataset = IceCubeDataset(
        data_path="synthetic_icecube_data.h5",
        split='test',
        max_points=5000,
        normalize=True,
        augment=False,
        class_names=['track', 'cascade', 'noise']
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    model = build_pointcept_classifier(
        model_name='simple',
        num_classes=3,
        in_channels=2
    )
    model.eval()
    
    # Process first batch
    batch = next(iter(dataloader))
    
    with torch.no_grad():
        predictions = model.predict(batch['coord'], batch['feat'])
        probabilities = model.predict_proba(batch['coord'], batch['feat'])
    
    print(f"Batch size: {len(predictions)}")
    print(f"Predictions: {predictions.numpy()}")
    print(f"\nPer-event probabilities:")
    for i, (pred, prob, true_label) in enumerate(zip(
        predictions, probabilities, batch['label']
    )):
        print(f"  Event {i}: pred={dataset.class_names[pred]} "
              f"(true={dataset.class_names[true_label]}), "
              f"confidence={prob[pred]:.3f}")


def main():
    """Run all examples."""
    print("="*60)
    print("IceCube Event Classifier - Usage Examples")
    print("="*60)
    
    try:
        example_training()
    except Exception as e:
        print(f"Example 1 failed (might need data): {e}")
    
    try:
        example_inference()
    except Exception as e:
        print(f"Example 2 failed (might need data): {e}")
    
    try:
        example_custom_preprocessing()
    except Exception as e:
        print(f"Example 3 failed: {e}")
    
    try:
        example_batch_prediction()
    except Exception as e:
        print(f"Example 4 failed (might need data): {e}")
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)


if __name__ == '__main__':
    main()
