"""
Classifier wrapper for Pointcept models.

This module provides a wrapper around Pointcept models for 
IceCube event classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
import logging


logger = logging.getLogger(__name__)


class PointceptClassifier(nn.Module):
    """
    Wrapper for Pointcept models for IceCube event classification.
    
    This class wraps a Pointcept backbone model and adds a classification head
    for IceCube event type prediction.
    
    Args:
        backbone: Pointcept backbone model (e.g., PointTransformer, PointNet++)
        num_classes: Number of output classes
        in_channels: Number of input feature channels
        hidden_dim: Hidden dimension for classification head
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        backbone: Optional[nn.Module] = None,
        num_classes: int = 3,
        in_channels: int = 1,
        hidden_dim: int = 256,
        dropout: float = 0.5,
        use_global_pooling: bool = True
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.use_global_pooling = use_global_pooling
        
        # Use provided backbone or create a simple default
        if backbone is not None:
            self.backbone = backbone
            # Try to infer output dimension
            self.backbone_out_dim = self._infer_backbone_dim()
        else:
            # Simple PointNet-style backbone as default
            self.backbone = self._create_simple_backbone()
            self.backbone_out_dim = hidden_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone_out_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def _infer_backbone_dim(self) -> int:
        """Infer output dimension of backbone."""
        # Try to get output dimension from backbone
        if hasattr(self.backbone, 'out_channels'):
            return self.backbone.out_channels
        elif hasattr(self.backbone, 'output_dim'):
            return self.backbone.output_dim
        else:
            # Default assumption
            return 256
    
    def _create_simple_backbone(self) -> nn.Module:
        """Create a simple PointNet-style backbone."""
        return SimplePointNet(
            in_channels=self.in_channels,
            out_channels=self.hidden_dim
        )
    
    def forward(
        self, 
        coord: torch.Tensor, 
        feat: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            coord: (B, N, 3) point coordinates
            feat: (B, N, F) point features
            
        Returns:
            logits: (B, num_classes) classification logits
        """
        # Extract features with backbone
        try:
            # Try Pointcept-style interface
            features = self.backbone(
                coord=coord,
                feat=feat,
                **kwargs
            )
        except TypeError:
            # Fallback to simple interface
            # Concatenate coordinates and features
            x = torch.cat([coord, feat], dim=-1)  # (B, N, 3+F)
            features = self.backbone(x)  # (B, N, D) or (B, D)
        
        # Global pooling if needed
        if len(features.shape) == 3:  # (B, N, D)
            if self.use_global_pooling:
                # Max + Average pooling
                max_pool = torch.max(features, dim=1)[0]  # (B, D)
                avg_pool = torch.mean(features, dim=1)     # (B, D)
                features = torch.cat([max_pool, avg_pool], dim=-1)  # (B, 2*D)
                
                # Adjust classifier input if needed
                if features.shape[1] != self.backbone_out_dim:
                    self.backbone_out_dim = features.shape[1]
                    self._rebuild_classifier()
            else:
                features = features[:, 0, :]  # Use first point feature
        
        # Classification
        logits = self.classifier(features)
        
        return logits
    
    def _rebuild_classifier(self):
        """Rebuild classifier with correct input dimension."""
        hidden_dim = self.hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone_out_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim // 2, self.num_classes)
        )
    
    def predict(
        self,
        coord: torch.Tensor,
        feat: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Get predicted class labels.
        
        Args:
            coord: (B, N, 3) point coordinates
            feat: (B, N, F) point features
            
        Returns:
            predictions: (B,) predicted class indices
        """
        logits = self.forward(coord, feat, **kwargs)
        return torch.argmax(logits, dim=-1)
    
    def predict_proba(
        self,
        coord: torch.Tensor,
        feat: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Get class probabilities.
        
        Args:
            coord: (B, N, 3) point coordinates
            feat: (B, N, F) point features
            
        Returns:
            probabilities: (B, num_classes) class probabilities
        """
        logits = self.forward(coord, feat, **kwargs)
        return F.softmax(logits, dim=-1)


class SimplePointNet(nn.Module):
    """
    Simple PointNet-style feature extractor.
    
    This is a fallback backbone when no Pointcept model is provided.
    """
    
    def __init__(
        self,
        in_channels: int = 4,  # 3 coords + 1 feature
        out_channels: int = 256
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Shared MLP
        self.mlp1 = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (B, N, C) input point cloud
            
        Returns:
            features: (B, N, out_channels) point features
        """
        batch_size, num_points, in_channels = x.shape
        
        # Reshape to process all points: (B*N, C)
        x = x.reshape(-1, in_channels)
        
        # Apply MLP to each point
        x = self.mlp1(x)
        
        # Reshape back to (B, N, out_channels)
        x = x.reshape(batch_size, num_points, self.out_channels)
        
        return x


def build_pointcept_classifier(
    model_name: str = 'simple',
    num_classes: int = 3,
    in_channels: int = 1,
    **kwargs
) -> PointceptClassifier:
    """
    Build a Pointcept classifier.
    
    Args:
        model_name: Name of the backbone model
        num_classes: Number of classes
        in_channels: Number of input feature channels
        **kwargs: Additional arguments for the model
        
    Returns:
        PointceptClassifier instance
    """
    # Try to import Pointcept models
    backbone = None
    
    if model_name != 'simple':
        try:
            # Attempt to import from Pointcept
            # This is a placeholder - actual import depends on Pointcept installation
            from pointcept.models import build_model
            
            # Build Pointcept model with config
            backbone = build_model(model_name, **kwargs)
            logger.info(f"Loaded Pointcept model: {model_name}")
        except ImportError:
            logger.warning(
                f"Could not import Pointcept. Using simple backbone. "
                f"Install Pointcept for advanced models."
            )
    
    return PointceptClassifier(
        backbone=backbone,
        num_classes=num_classes,
        in_channels=in_channels,
        **kwargs
    )
