"""
Inference script for IceCube event classifier.
"""

import argparse
import os
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from pointcept_classifier.data import IceCubeDataset
from pointcept_classifier.models import build_pointcept_classifier
from pointcept_classifier.utils import load_config, setup_logger


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: list
) -> dict:
    """
    Evaluate model and compute detailed metrics.
    
    Returns:
        Dictionary with evaluation results
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            coord = batch['coord'].to(device)
            feat = batch['feat'].to(device)
            label = batch['label'].to(device)
            
            # Forward pass
            logits = model(coord, feat)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    num_classes = len(class_names)
    
    # Overall accuracy
    accuracy = (all_preds == all_labels).mean() * 100
    
    # Per-class metrics
    class_metrics = {}
    for i, class_name in enumerate(class_names):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc = (all_preds[mask] == i).mean() * 100
            class_metrics[class_name] = {
                'accuracy': class_acc,
                'count': int(mask.sum())
            }
    
    # Confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(all_labels, all_preds):
        confusion_matrix[true_label, pred_label] += 1
    
    results = {
        'overall_accuracy': float(accuracy),
        'class_metrics': class_metrics,
        'confusion_matrix': confusion_matrix.tolist(),
        'num_samples': len(all_labels)
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate IceCube event classifier')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, required=True, help='Path to test data')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--split', type=str, default='test', help='Data split to evaluate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(
        name='inference',
        level=logging.INFO,
        log_file=output_dir / 'inference.log'
    )
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get('config', {})
    
    # Create dataset
    logger.info("Loading dataset...")
    dataset = IceCubeDataset(
        data_path=args.data_path,
        split=args.split,
        max_points=config.get('data', {}).get('max_points', 5000),
        normalize=config.get('data', {}).get('normalize', True),
        augment=False,
        class_names=config.get('data', {}).get('class_names', None)
    )
    
    logger.info(f"Dataset: {len(dataset)} events")
    logger.info(f"Classes: {dataset.class_names}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    logger.info("Creating model...")
    model = build_pointcept_classifier(
        model_name=config.get('model', {}).get('name', 'simple'),
        num_classes=dataset.num_classes,
        in_channels=config.get('model', {}).get('in_channels', 1),
        hidden_dim=config.get('model', {}).get('hidden_dim', 256),
        dropout=config.get('model', {}).get('dropout', 0.5)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Evaluate
    logger.info("Evaluating model...")
    results = evaluate(model, dataloader, device, dataset.class_names)
    
    # Print results
    logger.info("\n" + "="*50)
    logger.info("EVALUATION RESULTS")
    logger.info("="*50)
    logger.info(f"Overall Accuracy: {results['overall_accuracy']:.2f}%")
    logger.info(f"Number of Samples: {results['num_samples']}")
    
    logger.info("\nPer-Class Metrics:")
    for class_name, metrics in results['class_metrics'].items():
        logger.info(f"  {class_name}: {metrics['accuracy']:.2f}% ({metrics['count']} samples)")
    
    logger.info("\nConfusion Matrix:")
    logger.info("           " + "  ".join([f"{name:>10}" for name in dataset.class_names]))
    for i, row in enumerate(results['confusion_matrix']):
        logger.info(f"{dataset.class_names[i]:>10} " + "  ".join([f"{val:>10}" for val in row]))
    
    # Save results
    results_file = output_dir / 'results.json'
    # Convert numpy arrays to lists for JSON serialization
    results_json = results.copy()
    with open(results_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    logger.info(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    import logging
    main()
