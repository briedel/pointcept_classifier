"""
Pointcept Classifier for IceCube Events

A framework for using Pointcept point cloud models to classify
neutrino events from the IceCube detector.
"""

__version__ = "0.1.0"
__author__ = "IceCube Collaboration"

from .models import PointceptClassifier
from .data import IceCubeDataset

__all__ = ["PointceptClassifier", "IceCubeDataset"]
