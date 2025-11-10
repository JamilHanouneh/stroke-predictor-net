#!/usr/bin/env python3
"""
Main training script for StrokePredictorNet.
Handles complete training loop with validation.
"""

import sys
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_parser import load_config
from src.utils.logger import setup_logger, log_config
from src.utils.device_manager import get_device_manager
from src.utils.seed import set_seed
from src.utils.checkpointing import CheckpointManager
from src.data.dataset import ISLESStrokeDataset
from src.models.multimodal_fusion import MultimodalStrokeNet
from src.models.loss import CombinedLoss
from src.models.metrics import SegmentationMetrics


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train StrokePredictorNet')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config.yaml')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    return parser.parse_args()


def create_dataloaders(config, device_mgr):
    """
    Create train/val/test dataloaders.
    
    Args:
        config (dict): Configuration
        device_mgr (DeviceManager): Device manager
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    logger = logging.getLogger('StrokePredictorNet')
    
    # Load subject splits
    splits_dir = Path(config['data']['paths']['splits'])
    
    with open(splits_dir / 'train_subjects.txt', 'r') as f:
        train_ids = [line.strip() for line in f]
    
    with open(splits_dir / 'val_subjects.txt', 'r') as f:
        val_ids = [line.strip() for line in f]
    
    with open(splits_dir / 'test_subjects.txt', 'r') as f:
        test_ids = [line.strip() for line in f]
    
    logger.info(f"Loaded splits: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")
    
    # Create datasets
    processed_dir = Path(config['data']['paths']['processed_data'])
    
    train_dataset = ISLESStrokeDataset(
        imaging_h5_path=processed_dir / 'imaging_tensors.h5',
        clinical_csv_path=config['data']['paths']['clinical_metadata'],
        masks_h5_path=processed_dir / 'lesion_masks.h5',
        subject_ids=train_ids,
        transform=None,  # TODO: Add augmentation
        normalize=True
    )
    
    val_dataset = ISLESStrokeDataset(
        imaging_h5_path=processed_dir / 'imaging_tensors.h5',
        clinical_csv_path=config['data']['paths']['clinical_metadata'],
        masks_h5_path=processed_dir / 'lesion_masks.h5',
        subject_ids=val_ids,
        transform=None,
        normalize=True
    )
    
    # Get dataloader settings from device manager
    batch_size = device_mgr.get_batch_size(config)
    num_workers = device_mgr.get_num_workers(config)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device_mgr.device.type == 'cuda'),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device_mgr.device.type == 'cuda')
    )
    
    logger.info(f"Created dataloaders: batch_size={batch_size}, num_workers={num_workers}")
    
    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, device_mgr, scaler, grad_accum_steps):
    """
    Train for one epoch.
    
    Args:
        model: Neural network model
        train_loader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device_mgr: Device manager
        scaler: GradScaler for mixed precision
        grad_accum_steps: Gradient accumulation steps
    
    Returns:
        dict: Training metrics
    """
    model.train()
    
    total_loss = 0.0
    total_dice_loss = 0.0
    total_ce_loss = 0.0
    
    metrics_calculator = SegmentationMetrics()
    total_dice_score = 0.0
    
    progress_bar = tqdm(train_loader, desc='Training')
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        imaging = device_mgr.to_device(batch['imaging'])
        clinical = device_mgr.to_device(batch['clinical'])
        masks = device_mgr.to_device(batch['mask'])
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=device_mgr.use_amp):
            outputs = model(imaging, clinical)
            segmentation = outputs['segmentation']
            
            # Compute loss
            loss_dict = criterion(segmentation, masks)
            loss = loss_dict['total'] / grad_accum_steps
        
        # Backward pass
        if device_mgr.use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % grad_accum_steps == 0:
            if device_mgr.use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()
        
        # Accumulate metrics
        total_loss += loss_dict['total'].item()
        total_dice_loss += loss_dict['dice'].item()
        total_ce_loss += loss_dict['ce'].item()
        
        # Compute Dice score
        with torch.no_grad():
            probs = torch.sigmoid(segmentation)
            dice = metrics_calculator.dice_score(probs, masks)
            total_dice_score += dice
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss_dict['total'].item():.4f}",
            'dice': f"{dice:.4f}"
        })
    
    # Average metrics
    num_batches = len(train_loader)
    
    return {
        'loss': total_loss / num_batches,
        'dice_loss': total_dice_loss / num_batches,
        'ce_loss': total_ce_loss / num_batches,
        'dice_score': total_dice_score / num_batches
    }


def validate(model, val_loader, criterion, device_mgr):
    """
    Validate model.
    
    Args:
        model: Neural network model
        val_loader: Validation dataloader
        criterion: Loss function
        device_mgr: Device manager
    
    Returns:
        dict: Validation metrics
    """
    model.eval()
    
    total_loss = 0.0
    metrics_calculator = SegmentationMetrics()
    
    all_metrics = {
        'dice': 0.0,
        'iou': 0.0,
        'sensitivity': 0.0,
        'specificity': 0.0
    }
    
    progress_bar = tqdm(val_loader, desc='Validation')
    
    with torch.no_grad():
        for batch in progress_bar:
            # Move to device
            imaging = device_mgr.to_device(batch['imaging'])
            clinical = device_mgr.to_device(batch['clinical'])
            masks = device_mgr.to_device(batch['mask'])
            
            # Forward pass
            outputs = model(imaging, clinical)
            segmentation = outputs['segmentation']
            
            # Compute loss
            loss_dict = criterion(segmentation, masks)
            total_loss += loss_dict['total'].item()
            
            # Compute metrics
            probs = torch.sigmoid(segmentation)
            batch_metrics = metrics_calculator.compute_all_metrics(probs, masks)
            
            for key in all_metrics:
                if key in batch_metrics:
                    all_metrics[key] += batch_metrics[key]
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss_dict['total'].item():.4f}",
                'dice': f"{batch_metrics['dice']:.4f}"
            })
    
    # Average metrics
    num_batches = len(val_loader)
    
    results = {
        'loss': total_loss / num_batches,
        **{k: v / num_batches for k, v in all_metrics.items()}
    }
    
    return results


def main():
    """Main training function"""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with CLI args
    if args.device != 'auto':
        config['system']['device'] = args.device
    
    if args.epochs is not None:
        config['training']['num_epochs'] = args.epochs
    
    # Setup logging
    logger = setup_logger(config)
    log_config(logger, config)
    
    # Set random seed
    set_seed(config['system']['seed'], config['system']['deterministic'])
    
    # Initialize device manager
    device_mgr = get_device_manager(config)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config, device_mgr)
    
    # Create model
    logger.info("Creating model...")
    model = MultimodalStrokeNet(config)
    model = device_mgr.to_device(model)
    
    # Create loss function
    loss_config = config['model']['loss']['segmentation']
    criterion = CombinedLoss(
        dice_weight=loss_config['dice_weight'],
        ce_weight=loss_config['ce_weight']
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs'],
        eta_min=config['training']['scheduler']['min_lr']
    )
    
    # Create checkpoint manager
    checkpoint_mgr = CheckpointManager(
        checkpoint_dir=config['paths']['checkpoints'],
        monitor='val_dice',
        mode='max'
    )
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if device_mgr.use_amp else None
    
    # Gradient accumulation
    grad_accum_steps = device_mgr.get_gradient_accumulation_steps(config)
    
    # TensorBoard
    if config['logging']['tensorboard']['enabled']:
        writer = SummaryWriter(config['logging']['tensorboard']['log_dir'])
    else:
        writer = None
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        checkpoint = checkpoint_mgr.load_checkpoint(
            args.resume, model, optimizer, scheduler
        )
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    logger.info("=" * 70)
    logger.info("Starting Training")
    logger.info("=" * 70)
    
    best_val_dice = 0.0
    patience_counter = 0
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        logger.info(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
        logger.info("-" * 70)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer,
            device_mgr, scaler, grad_accum_steps
        )
        
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                   f"Dice: {train_metrics['dice_score']:.4f}")
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device_mgr)
        
        logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                   f"Dice: {val_metrics['dice']:.4f}, "
                   f"IoU: {val_metrics['iou']:.4f}")
        
        # Learning rate step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"Learning Rate: {current_lr:.6f}")
        
        # TensorBoard logging
        if writer:
            writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
            writer.add_scalar('Train/Dice', train_metrics['dice_score'], epoch)
            writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
            writer.add_scalar('Val/Dice', val_metrics['dice'], epoch)
            writer.add_scalar('Val/IoU', val_metrics['iou'], epoch)
            writer.add_scalar('LR', current_lr, epoch)
        
        # Save checkpoint
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'config': config
        }
        
        # Save best model
        is_best = checkpoint_mgr.save_best_model(
            checkpoint_state,
            val_metrics['dice']
        )
        
        if is_best:
            best_val_dice = val_metrics['dice']
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Save periodic checkpoint
        if (epoch + 1) % config['training']['checkpointing']['save_every_n_epochs'] == 0:
            checkpoint_mgr.save_checkpoint(
                checkpoint_state,
                f'checkpoint_epoch_{epoch + 1}.pth'
            )
        
        # Early stopping
        if config['training']['early_stopping']['enabled']:
            if patience_counter >= config['training']['early_stopping']['patience']:
                logger.info(f"\nEarly stopping triggered after {epoch + 1} epochs")
                logger.info(f"Best validation Dice: {best_val_dice:.4f}")
                break
    
    # Save final model
    checkpoint_mgr.save_checkpoint(checkpoint_state, 'final_model.pth')
    
    logger.info("=" * 70)
    logger.info("Training Complete!")
    logger.info(f"Best Validation Dice: {best_val_dice:.4f}")
    logger.info("=" * 70)
    
    if writer:
        writer.close()


if __name__ == '__main__':
    main()
