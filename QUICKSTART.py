"""
Quick start guide for pointcept_classifier.

This notebook/script walks through the basic usage of the package.
"""

# QUICK START GUIDE
# ==================

print("Pointcept Classifier for IceCube - Quick Start Guide")
print("=" * 60)

# 1. Installation
print("\n1. INSTALLATION")
print("-" * 60)
print(
    """
# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Optional: Install Pointcept for advanced models
# Follow instructions at: https://github.com/Pointcept/Pointcept
"""
)

# 2. Data Preparation
print("\n2. DATA PREPARATION")
print("-" * 60)
print(
    """
Your IceCube data should be in HDF5 format:

Structure:
  data.h5
  ├── train/
  │   ├── 0/
  │   │   ├── positions: (N, 3) array
  │   │   ├── features: (N, F) array
  │   │   └── label: integer
  │   └── ...
  ├── val/
  └── test/

Create synthetic data for testing:
  python examples/create_synthetic_data.py --output synthetic_data.h5 --num_events 1000
"""
)

# 3. Training
print("\n3. TRAINING")
print("-" * 60)
print(
    """
Edit configs/default_config.yaml to set your data path, then:

  python train.py --config configs/default_config.yaml --output_dir outputs/my_experiment

Training options:
  --config: Path to config file
  --data_path: Override data path in config
  --output_dir: Where to save checkpoints and logs
  --resume: Resume from checkpoint
  --device: cuda or cpu
"""
)

# 4. Evaluation
print("\n4. EVALUATION")
print("-" * 60)
print(
    """
Evaluate a trained model:

  python inference.py \\
    --checkpoint outputs/my_experiment/best_checkpoint.pth \\
    --data_path /path/to/test/data.h5 \\
    --output_dir outputs/evaluation

This will:
  - Load the trained model
  - Evaluate on test data
  - Print metrics and confusion matrix
  - Save results to JSON
"""
)

# 5. Programmatic Usage
print("\n5. PROGRAMMATIC USAGE")
print("-" * 60)
print(
    """
Use the classifier in your own code:

```python
import torch
from pointcept_classifier.data import IceCubeDataset
from pointcept_classifier.models import build_pointcept_classifier

# Load data
dataset = IceCubeDataset(
    data_path="data.h5",
    split='test',
    class_names=['track', 'cascade', 'noise']
)

# Create model
model = build_pointcept_classifier(
    model_name='simple',
    num_classes=3,
    in_channels=2
)

# Load checkpoint
checkpoint = torch.load("best_checkpoint.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
sample = dataset[0]
coord = sample['coord'].unsqueeze(0)
feat = sample['feat'].unsqueeze(0)

with torch.no_grad():
    predictions = model.predict(coord, feat)
    probabilities = model.predict_proba(coord, feat)

print(f"Predicted: {dataset.class_names[predictions.item()]}")
print(f"Confidence: {probabilities.max():.3f}")
```
"""
)

# 6. Example Scripts
print("\n6. EXAMPLE SCRIPTS")
print("-" * 60)
print(
    """
Try the example scripts:

  # Create synthetic data
  python examples/create_synthetic_data.py

  # Run usage examples
  python examples/usage_example.py
"""
)

# 7. Configuration
print("\n7. CONFIGURATION")
print("-" * 60)
print(
    """
Key configuration options in configs/default_config.yaml:

data:
  path: "/path/to/data.h5"
  max_points: 5000           # Max points per event
  normalize: true            # Normalize coordinates
  augment: true              # Data augmentation
  class_names:
    - "track"
    - "cascade"
    - "noise"

model:
  name: "simple"             # Model architecture
  in_channels: 2             # Number of features
  hidden_dim: 256            # Hidden dimension
  dropout: 0.5               # Dropout rate

training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 0.001
  use_class_weights: true    # For imbalanced data
"""
)

# 8. Advanced Features
print("\n8. ADVANCED FEATURES")
print("-" * 60)
print(
    """
- Class weights for imbalanced datasets
- Data augmentation (rotation, jittering, scaling)
- Flexible model architectures
- Integration with Pointcept models
- Comprehensive evaluation metrics
- Custom preprocessing pipelines
"""
)

# 9. Next Steps
print("\n9. NEXT STEPS")
print("-" * 60)
print(
    """
1. Generate synthetic data to test the pipeline
2. Convert your IceCube data to the HDF5 format
3. Train a simple baseline model
4. Evaluate and iterate on model architecture
5. Try advanced Pointcept models for better performance
6. Fine-tune hyperparameters using validation set

For more details, see README.md
"""
)

print("\n" + "=" * 60)
print("Ready to classify IceCube events!")
print("=" * 60)
