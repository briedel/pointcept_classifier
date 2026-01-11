# Architecture Overview

## System Components

The Pointcept Classifier for IceCube consists of several key components:

### 1. Data Layer (`pointcept_classifier/data/`)

**IceCubeDataset** (`dataset.py`)
- PyTorch Dataset for loading IceCube events from HDF5 files
- Handles padding/sampling to fixed point count
- Implements data augmentation (rotation, jittering, scaling)
- Supports class weighting for imbalanced datasets

**Preprocessing** (`preprocessing.py`)
- Coordinate normalization
- Feature scaling
- Noise removal
- Point cloud format conversion

### 2. Model Layer (`pointcept_classifier/models/`)

**PointceptClassifier** (`classifier.py`)
- Wrapper around Pointcept backbone models
- Classification head with dropout and batch normalization
- Global pooling (max + average)
- Inference methods for predictions and probabilities

**SimplePointNet**
- Fallback baseline when Pointcept is not available
- Lightweight PointNet-style architecture
- Good for quick experiments

### 3. Utilities (`pointcept_classifier/utils/`)

**Configuration** (`config.py`)
- YAML/JSON config file loading
- Config merging and management
- Attribute-style access to configs

**Logging** (`logger.py`)
- Console and file logging
- Configurable log levels
- Context managers for temporary level changes

### 4. Training & Inference Scripts

**train.py**
- Full training pipeline
- Checkpoint saving and resuming
- Learning rate scheduling
- Validation during training

**inference.py**
- Model evaluation
- Comprehensive metrics (accuracy, confusion matrix)
- Results saving to JSON

## Data Flow

```
Raw IceCube Data (HDF5)
    ↓
IceCubeDataset
    ↓ (preprocessing, augmentation)
Point Cloud (coordinates + features)
    ↓
PointceptClassifier
    ↓ (backbone feature extraction)
Point Features
    ↓ (global pooling)
Global Feature Vector
    ↓ (classification head)
Class Logits → Predictions
```

## Key Design Decisions

### 1. Point Cloud Representation
IceCube events naturally fit the point cloud paradigm:
- Each PMT hit is a point in 3D space
- Position: (x, y, z) coordinates of the DOM
- Features: time, charge, and optional auxiliary info

### 2. Fixed Point Count
Events are sampled/padded to a fixed number of points:
- Enables efficient batching
- Configurable via `max_points` parameter
- Default: 5000 points per event

### 3. Normalization
Spatial coordinates are normalized to unit cube:
- Centers point cloud at origin
- Scales to [-1, 1] range
- Makes model agnostic to detector scale

### 4. Data Augmentation
Training augmentation includes:
- Random rotation around vertical axis (z-axis)
- Small jittering of coordinates
- Random feature scaling
- Not applied during validation/testing

### 5. Class Weighting
Handles imbalanced datasets:
- Inverse frequency weighting
- Computed automatically from training data
- Optional via configuration

### 6. Modular Architecture
Easy to swap components:
- Different backbone models
- Custom preprocessing
- Alternative augmentation strategies
- Flexible configuration system

## Integration with Pointcept

The package is designed to work with or without Pointcept:

**Without Pointcept:**
- Uses built-in SimplePointNet backbone
- Sufficient for many use cases
- Quick to set up and test

**With Pointcept:**
- Access to state-of-the-art models
- PointTransformer, PointNet++, PointMLP, etc.
- Better performance on complex tasks
- Requires separate Pointcept installation

## Extension Points

The architecture is designed for extensibility:

### Adding New Models
```python
from pointcept.models import MyModel
from pointcept_classifier.models import PointceptClassifier

backbone = MyModel(...)
classifier = PointceptClassifier(backbone=backbone, num_classes=3)
```

### Custom Preprocessing
```python
from pointcept_classifier.data import IceCubeDataset

class CustomDataset(IceCubeDataset):
    def _preprocess(self, positions, features):
        # Your custom preprocessing
        return positions, features
```

### New Event Types
Just update `class_names` in configuration:
```yaml
data:
  class_names:
    - "new_type_1"
    - "new_type_2"
    - "new_type_3"
```

## Performance Considerations

### Memory
- Point clouds can be large (5000+ points per event)
- Batch size may need adjustment based on GPU memory
- Consider point sampling strategies for very large events

### Training Speed
- Data augmentation adds computational cost
- Use `num_workers > 0` for parallel data loading
- Consider mixed precision training for larger models

### Inference
- Batch processing for efficiency
- Model can be exported to ONNX/TorchScript for deployment
- Consider quantization for edge deployment

## Future Enhancements

Potential areas for improvement:
- Support for irregular point clouds (no padding)
- Attention-based pooling mechanisms
- Multi-task learning (classification + regression)
- Self-supervised pre-training on unlabeled data
- Online hard example mining
- Knowledge distillation from larger models
