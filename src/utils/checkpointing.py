"""
Model checkpointing utilities.
Based on your BrainGraphNet checkpointing style.
"""

import torch
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages model checkpoints during training.
    
    Features:
    - Save best model based on metric
    - Save periodic checkpoints
    - Resume from checkpoint
    - Load for inference
    """
    
    def __init__(self, checkpoint_dir, monitor='val_dice', mode='max'):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir (str or Path): Directory to save checkpoints
            monitor (str): Metric to monitor ('val_dice', 'val_loss', etc.)
            mode (str): 'max' for metrics to maximize, 'min' to minimize
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.monitor = monitor
        self.mode = mode
        
        # Track best metric
        if mode == 'max':
            self.best_metric = float('-inf')
        else:
            self.best_metric = float('inf')
        
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
        logger.info(f"Monitoring: {monitor} ({mode})")
    
    def save_checkpoint(self, state, filename='checkpoint.pth'):
        """
        Save checkpoint.
        
        Args:
            state (dict): State dictionary containing:
                - model_state_dict
                - optimizer_state_dict
                - scheduler_state_dict (optional)
                - epoch
                - metrics
                - config
            filename (str): Checkpoint filename
        """
        filepath = self.checkpoint_dir / filename
        torch.save(state, filepath)
        logger.info(f"✓ Saved checkpoint: {filepath}")
    
    def save_best_model(self, state, current_metric):
        """
        Save model if it's the best so far.
        
        Args:
            state (dict): State dictionary
            current_metric (float): Current value of monitored metric
        
        Returns:
            bool: True if model was saved as best
        """
        is_best = False
        
        if self.mode == 'max':
            if current_metric > self.best_metric:
                is_best = True
        else:
            if current_metric < self.best_metric:
                is_best = True
        
        if is_best:
            self.best_metric = current_metric
            self.save_checkpoint(state, 'best_model.pth')
            logger.info(f"✓ New best model! {self.monitor}: {current_metric:.4f}")
        
        return is_best
    
    def load_checkpoint(self, filepath, model, optimizer=None, scheduler=None):
        """
        Load checkpoint and restore state.
        
        Args:
            filepath (str or Path): Path to checkpoint
            model (nn.Module): Model to load state into
            optimizer (Optimizer, optional): Optimizer to restore
            scheduler (Scheduler, optional): Scheduler to restore
        
        Returns:
            dict: Checkpoint state
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        logger.info(f"Loading checkpoint: {filepath}")
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("✓ Model state loaded")
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("✓ Optimizer state loaded")
        
        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("✓ Scheduler state loaded")
        
        # Log checkpoint info
        if 'epoch' in checkpoint:
            logger.info(f"Resuming from epoch {checkpoint['epoch']}")
        
        if 'metrics' in checkpoint:
            logger.info(f"Checkpoint metrics: {checkpoint['metrics']}")
        
        return checkpoint
    
    def get_latest_checkpoint(self):
        """
        Get path to most recent checkpoint.
        
        Returns:
            Path or None: Path to latest checkpoint
        """
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        
        if not checkpoints:
            return None
        
        # Sort by epoch number
        checkpoints.sort(key=lambda p: int(p.stem.split('_')[-1]))
        return checkpoints[-1]
