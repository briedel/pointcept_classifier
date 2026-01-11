"""
Example script to create synthetic IceCube-like data in Parquet format.

This demonstrates the expected Parquet format for the classifier.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import argparse


def generate_synthetic_event(event_type: str, num_points: int = 1000) -> dict:
    """
    Generate a synthetic IceCube event.

    Args:
        event_type: 'track', 'cascade', or 'noise'
        num_points: Number of PMT hits

    Returns:
        Dictionary with event data
    """
    class_map = {"track": 0, "cascade": 1, "noise": 2}
    label = class_map[event_type]

    if event_type == "track":
        # Generate a track-like pattern (linear)
        t = np.linspace(0, 100, num_points)
        direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)

        positions = t[:, np.newaxis] * direction
        positions += np.random.randn(num_points, 3) * 5  # Add noise

        # Time increases along track
        time = t + np.random.randn(num_points) * 2
        charge = np.random.exponential(10, num_points)

    elif event_type == "cascade":
        # Generate a cascade-like pattern (spherical)
        r = np.random.exponential(20, num_points)
        theta = np.random.uniform(0, np.pi, num_points)
        phi = np.random.uniform(0, 2 * np.pi, num_points)

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        positions = np.column_stack([x, y, z])

        # Time roughly proportional to distance
        time = r / 0.3 + np.random.randn(num_points) * 5  # speed of light in ice
        charge = np.random.exponential(15, num_points) * (1 + r / 50)

    else:  # noise
        # Random positions
        positions = np.random.randn(num_points, 3) * 50
        time = np.random.uniform(0, 1000, num_points)
        charge = np.random.exponential(2, num_points)

    return {
        "x": positions[:, 0].astype(np.float32),
        "y": positions[:, 1].astype(np.float32),
        "z": positions[:, 2].astype(np.float32),
        "time": time.astype(np.float32),
        "charge": charge.astype(np.float32),
        "label": np.full(num_points, label, dtype=np.int32),
    }


def create_synthetic_parquet_dataset(
    output_file: str,
    num_events: int = 1000,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    """
    Create a synthetic dataset in Parquet format.

    Args:
        output_file: Path to output Parquet file
        num_events: Total number of events to generate
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
    """
    print(f"Creating synthetic Parquet dataset with {num_events} events...")

    # Calculate split sizes
    n_train = int(num_events * train_ratio)
    n_val = int(num_events * val_ratio)
    n_test = num_events - n_train - n_val

    event_types = ["track", "cascade", "noise"]
    all_data = []

    for split_name, split_size in [
        ("train", n_train),
        ("val", n_val),
        ("test", n_test),
    ]:
        print(f"Generating {split_size} {split_name} events...")

        for i in range(split_size):
            # Randomly choose event type
            event_type = np.random.choice(event_types)
            num_points = np.random.randint(500, 2000)

            # Generate event
            event_data = generate_synthetic_event(event_type, num_points)

            # Add metadata
            event_id = f"{split_name}_{i}"
            event_data["event_id"] = [event_id] * num_points
            event_data["split"] = [split_name] * num_points

            # Create DataFrame for this event
            event_df = pd.DataFrame(event_data)
            all_data.append(event_df)

            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{split_size} events")

    # Combine all events
    print("Combining all events...")
    full_df = pd.concat(all_data, ignore_index=True)

    # Save to Parquet
    print(f"Saving to {output_file}...")
    table = pa.Table.from_pandas(full_df)
    pq.write_table(table, output_file, compression="snappy")

    print(f"Dataset saved to {output_file}")
    print(f"  Train: {n_train} events")
    print(f"  Val: {n_val} events")
    print(f"  Test: {n_test} events")
    print(f"  Total rows: {len(full_df)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic IceCube-like data in Parquet format"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="synthetic_icecube_data.parquet",
        help="Output Parquet file path",
    )
    parser.add_argument(
        "--num_events",
        type=int,
        default=1000,
        help="Total number of events to generate",
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.7, help="Fraction of events for training"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Fraction of events for validation",
    )
    args = parser.parse_args()

    create_synthetic_parquet_dataset(
        output_file=args.output,
        num_events=args.num_events,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
