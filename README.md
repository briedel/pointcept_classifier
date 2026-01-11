# Pointcept Classifier for IceCube Events

A Python framework for classifying IceCube neutrino events using [Pointcept](https://github.com/Pointcept/Pointcept) point cloud models.

## Overview

This repository provides tools to apply state-of-the-art point cloud deep learning models to IceCube neutrino event classification. IceCube events are naturally represented as 3D point clouds, where each point corresponds to a PhotoMultiplier Tube (PMT) hit with associated features like time and charge.

### Features

- **IceCube-specific dataset loader** for HDF5 data format
- **Flexible model architecture** supporting simple baselines and advanced Pointcept models
- **Data preprocessing and augmentation** tailored for detector geometry
- **Training and inference scripts** with extensive configuration options
- **Class-weighted training** for handling imbalanced datasets
- **Comprehensive evaluation metrics** including per-class accuracy and confusion matrices

## Installation

### Basic Installation

```bash
git clone https://github.com/briedel/pointcept_classifier.git
cd pointcept_classifier
pip install -r requirements.txt
pip install -e .
```

### Installing Pointcept (Optional but Recommended)

For advanced models like PointTransformer, install Pointcept:

```bash
# Follow the official Pointcept installation guide
# https://github.com/Pointcept/Pointcept
```

## Data Format

The classifier expects IceCube events in HDF5 format with the following structure:

```
data.h5
├── train/
│   ├── 0/
│   │   ├── positions: (N, 3) array of PMT positions [x, y, z]
│   │   ├── features: (N, F) array of features [time, charge, ...]
│   │   └── label: integer class label
│   ├── 1/
│   └── ...
├── val/
└── test/
```

## Quick Start

### 1. Prepare Your Data

Organize your IceCube events into HDF5 format as described above.

### 2. Configure Training

Edit `configs/default_config.yaml` to set your data path and parameters:

```yaml
data:
  path: "/path/to/your/icecube/data.h5"
  max_points: 5000
  class_names:
    - "track"
    - "cascade"
    - "noise"

model:
  name: "simple"
  in_channels: 2
  hidden_dim: 256

training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 0.001
```

### 3. Train the Model

```bash
python train.py --config configs/default_config.yaml --output_dir outputs/experiment1
```

### 4. Evaluate the Model

```bash
python inference.py \
  --checkpoint outputs/experiment1/best_checkpoint.pth \
  --data_path /path/to/test/data.h5 \
  --output_dir outputs/experiment1/evaluation
```

## Usage Examples

### Training with Custom Configuration

```bash
python train.py \
  --config configs/pointtransformer_config.yaml \
  --output_dir outputs/pointtransformer \
  --device cuda
```

### Resuming Training from Checkpoint

```bash
python train.py \
  --config configs/default_config.yaml \
  --resume outputs/experiment1/latest_checkpoint.pth \
  --output_dir outputs/experiment1
```

### Inference on Test Set

```bash
python inference.py \
  --checkpoint outputs/experiment1/best_checkpoint.pth \
  --data_path /path/to/test/data.h5 \
  --split test \
  --batch_size 64
```

## Model Architectures

### Simple Baseline (Default)

A lightweight PointNet-style architecture suitable for quick experiments:

```yaml
model:
  name: "simple"
  in_channels: 2
  hidden_dim: 256
  dropout: 0.5
```

### Advanced Models (Requires Pointcept)

When Pointcept is installed, you can use advanced architectures:

- **PointTransformer**: State-of-the-art transformer-based point cloud model
- **PointNet++**: Hierarchical feature learning
- **PointMLP**: Efficient MLP-based architecture

```yaml
model:
  name: "pointtransformer"
  in_channels: 3
  hidden_dim: 512
  num_layers: 4
  num_heads: 8
```

## Event Classes

Common IceCube event classifications:

- **track**: Muon tracks from charged-current muon neutrino interactions
- **cascade**: Electromagnetic or hadronic cascades
- **noise**: Background noise events
- **nu_e**: Electron neutrino events
- **nu_mu**: Muon neutrino events
- **nu_tau**: Tau neutrino events

Customize class names in your configuration file.

## Data Augmentation

Training includes several augmentation techniques:

- Random rotation around vertical axis
- Random jittering of coordinates
- Random feature scaling
- Random point sampling

Disable augmentation for validation/testing:

```python
dataset = IceCubeDataset(
    data_path="data.h5",
    split="test",
    augment=False
)
```

## Project Structure

```
pointcept_classifier/
├── pointcept_classifier/      # Main package
│   ├── __init__.py
│   ├── data/                  # Data loading and preprocessing
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   ├── models/                # Model definitions
│   │   └── classifier.py
│   └── utils/                 # Utilities
│       ├── config.py
│       └── logger.py
├── configs/                   # Configuration files
│   ├── default_config.yaml
│   └── pointtransformer_config.yaml
├── train.py                   # Training script
├── inference.py               # Inference script
├── setup.py                   # Package setup
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Configuration Options

### Data Configuration

- `path`: Path to HDF5 data file(s)
- `max_points`: Maximum points per event (for padding/sampling)
- `normalize`: Normalize spatial coordinates
- `augment`: Apply data augmentation
- `class_names`: List of event class names

### Model Configuration

- `name`: Model architecture name
- `in_channels`: Number of input features
- `hidden_dim`: Hidden layer dimension
- `dropout`: Dropout probability

### Training Configuration

- `batch_size`: Training batch size
- `num_epochs`: Number of training epochs
- `learning_rate`: Initial learning rate
- `weight_decay`: L2 regularization weight
- `lr_step_size`: Learning rate scheduler step
- `lr_gamma`: Learning rate decay factor
- `use_class_weights`: Enable class weighting for imbalanced data

## Advanced Usage

### Using with Custom Pointcept Models

```python
from pointcept.models import build_model
from pointcept_classifier.models import PointceptClassifier

# Build a Pointcept backbone
backbone = build_model("pointtransformer", ...)

# Wrap with classifier
classifier = PointceptClassifier(
    backbone=backbone,
    num_classes=3,
    hidden_dim=512
)
```

### Custom Preprocessing

```python
from pointcept_classifier.data import preprocess_event

positions, features = preprocess_event(
    positions=raw_positions,
    features=raw_features,
    normalize=True,
    remove_noise=True,
    noise_threshold=0.1
)
```

### Batch Inference

```python
import torch
from pointcept_classifier.models import build_pointcept_classifier

# Load model
model = build_pointcept_classifier(num_classes=3)
checkpoint = torch.load("best_checkpoint.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict
with torch.no_grad():
    logits = model(coord, feat)
    predictions = torch.argmax(logits, dim=-1)
    probabilities = torch.softmax(logits, dim=-1)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{pointcept_classifier,
  title = {Pointcept Classifier for IceCube Events},
  author = {IceCube Collaboration},
  year = {2026},
  url = {https://github.com/briedel/pointcept_classifier}
}
```

## Acknowledgments

- [Pointcept](https://github.com/Pointcept/Pointcept) for the point cloud models
- [IceCube Collaboration](https://icecube.wisc.edu/) for the detector and data
- The PyTorch team for the deep learning framework

## Contact

For questions or issues, please open an issue on GitHub or contact the IceCube Collaboration.

## References

- [Pointcept: A Codebase for Point Cloud Perception Research](https://github.com/Pointcept/Pointcept)
- [IceCube Neutrino Observatory](https://icecube.wisc.edu/)
- [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593)
- [Point Transformer](https://arxiv.org/abs/2012.09164)