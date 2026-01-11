"""
Example script to create synthetic IceCube-like data for testing.

This demonstrates the expected data format for the classifier.
"""

import numpy as np
import h5py
from pathlib import Path
import argparse


def generate_synthetic_event(event_type: str, num_points: int = 1000) -> tuple:
    """
    Generate a synthetic IceCube event.
    
    Args:
        event_type: 'track', 'cascade', or 'noise'
        num_points: Number of PMT hits
        
    Returns:
        positions: (N, 3) PMT positions
        features: (N, 2) features [time, charge]
        label: class label
    """
    class_map = {'track': 0, 'cascade': 1, 'noise': 2}
    label = class_map[event_type]
    
    if event_type == 'track':
        # Generate a track-like pattern (linear)
        t = np.linspace(0, 100, num_points)
        direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)
        
        positions = t[:, np.newaxis] * direction
        positions += np.random.randn(num_points, 3) * 5  # Add noise
        
        # Time increases along track
        time = t + np.random.randn(num_points) * 2
        charge = np.random.exponential(10, num_points)
        
    elif event_type == 'cascade':
        # Generate a cascade-like pattern (spherical)
        r = np.random.exponential(20, num_points)
        theta = np.random.uniform(0, np.pi, num_points)
        phi = np.random.uniform(0, 2*np.pi, num_points)
        
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
        positions = np.column_stack([x, y, z])
        
        # Time roughly proportional to distance
        time = r / 0.3 + np.random.randn(num_points) * 5  # speed of light in ice
        charge = np.random.exponential(15, num_points) * (1 + r/50)
        
    else:  # noise
        # Random positions
        positions = np.random.randn(num_points, 3) * 50
        time = np.random.uniform(0, 1000, num_points)
        charge = np.random.exponential(2, num_points)
    
    features = np.column_stack([time, charge])
    
    return positions.astype(np.float32), features.astype(np.float32), label


def create_synthetic_dataset(
    output_file: str,
    num_events: int = 1000,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
):
    """
    Create a synthetic dataset for testing.
    
    Args:
        output_file: Path to output HDF5 file
        num_events: Total number of events to generate
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
    """
    print(f"Creating synthetic dataset with {num_events} events...")
    
    # Calculate split sizes
    n_train = int(num_events * train_ratio)
    n_val = int(num_events * val_ratio)
    n_test = num_events - n_train - n_val
    
    splits = {
        'train': n_train,
        'val': n_val,
        'test': n_test
    }
    
    event_types = ['track', 'cascade', 'noise']
    
    with h5py.File(output_file, 'w') as f:
        for split_name, split_size in splits.items():
            print(f"Generating {split_size} {split_name} events...")
            split_group = f.create_group(split_name)
            
            for i in range(split_size):
                # Randomly choose event type
                event_type = np.random.choice(event_types)
                num_points = np.random.randint(500, 2000)
                
                # Generate event
                positions, features, label = generate_synthetic_event(
                    event_type, num_points
                )
                
                # Create event group
                event_group = split_group.create_group(str(i))
                event_group.create_dataset('positions', data=positions)
                event_group.create_dataset('features', data=features)
                event_group.create_dataset('label', data=label)
                
                if (i + 1) % 100 == 0:
                    print(f"  Generated {i + 1}/{split_size} events")
    
    print(f"Dataset saved to {output_file}")
    print(f"  Train: {n_train} events")
    print(f"  Val: {n_val} events")
    print(f"  Test: {n_test} events")


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic IceCube-like data for testing'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='synthetic_icecube_data.h5',
        help='Output HDF5 file path'
    )
    parser.add_argument(
        '--num_events',
        type=int,
        default=1000,
        help='Total number of events to generate'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.7,
        help='Fraction of events for training'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.15,
        help='Fraction of events for validation'
    )
    args = parser.parse_args()
    
    create_synthetic_dataset(
        output_file=args.output,
        num_events=args.num_events,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )


if __name__ == '__main__':
    main()
