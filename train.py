"""
Training script for IceCube event classifier using Pointcept.
"""

import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

from pointcept_classifier.data import IceCubeDataset
from pointcept_classifier.models import PointceptClassifier, build_pointcept_classifier
from pointcept_classifier.utils import load_config, setup_logger, save_config


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    logger: logging.Logger
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        coord = batch['coord'].to(device)
        feat = batch['feat'].to(device)
        label = batch['label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(coord, feat)
        loss = criterion(logits, label)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        pred = torch.argmax(logits, dim=-1)
        correct += (pred == label).sum().item()
        total += label.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'acc': 100.0 * correct / total
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    logger: logging.Logger
) -> float:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            coord = batch['coord'].to(device)
            feat = batch['feat'].to(device)
            label = batch['label'].to(device)
            
            # Forward pass
            logits = model(coord, feat)
            loss = criterion(logits, label)
            
            # Statistics
            total_loss += loss.item()
            pred = torch.argmax(logits, dim=-1)
            correct += (pred == label).sum().item()
            total += label.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train IceCube event classifier')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data_path', type=str, help='Path to data (overrides config)')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.data_path:
        config['data']['path'] = args.data_path
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(
        name='train',
        level=logging.INFO,
        log_file=output_dir / 'train.log'
    )
    
    # Save configuration
    save_config(config, output_dir / 'config.yaml')
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = IceCubeDataset(
        data_path=config['data']['path'],
        split='train',
        max_points=config['data'].get('max_points', 5000),
        normalize=config['data'].get('normalize', True),
        augment=config['data'].get('augment', True),
        class_names=config['data'].get('class_names', None)
    )
    
    val_dataset = IceCubeDataset(
        data_path=config['data']['path'],
        split='val',
        max_points=config['data'].get('max_points', 5000),
        normalize=config['data'].get('normalize', True),
        augment=False,
        class_names=config['data'].get('class_names', None)
    )
    
    logger.info(f"Train dataset: {len(train_dataset)} events")
    logger.info(f"Val dataset: {len(val_dataset)} events")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )
    
    # Create model
    logger.info("Creating model...")
    model = build_pointcept_classifier(
        model_name=config['model'].get('name', 'simple'),
        num_classes=train_dataset.num_classes,
        in_channels=config['model'].get('in_channels', 1),
        hidden_dim=config['model'].get('hidden_dim', 256),
        dropout=config['model'].get('dropout', 0.5)
    )
    model = model.to(device)
    
    # Create criterion with class weights if specified
    if config['training'].get('use_class_weights', False):
        class_weights = train_dataset.get_class_weights().to(device)
        logger.info(f"Using class weights: {class_weights}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 1e-4)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['training'].get('lr_step_size', 20),
        gamma=config['training'].get('lr_gamma', 0.5)
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
    
    # Training loop
    logger.info("Starting training...")
    num_epochs = config['training']['num_epochs']
    
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, logger
        )
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, logger
        )
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'best_val_acc': best_val_acc,
            'config': config
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, output_dir / 'latest_checkpoint.pth')
        
        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint['best_val_acc'] = best_val_acc
            torch.save(checkpoint, output_dir / 'best_checkpoint.pth')
            logger.info(f"Saved best checkpoint with val_acc: {val_acc:.2f}%")
    
    logger.info("\nTraining completed!")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == '__main__':
    main()
