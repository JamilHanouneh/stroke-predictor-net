"""
Training callbacks for monitoring and control.
"""

import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping callback.
    
    Stops training if monitored metric doesn't improve.
    """
    
    def __init__(self, patience=10, min_delta=0.001, monitor='val_loss', mode='min'):
        """
        Initialize early stopping.
        
        Args:
            patience (int): Number of epochs to wait
            min_delta (float): Minimum change to qualify as improvement
            monitor (str): Metric to monitor
            mode (str): 'min' or 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.should_stop = False
    
    def __call__(self, current_value):
        """
        Check if training should stop.
        
        Args:
            current_value (float): Current metric value
        
        Returns:
            bool: True if should stop
        """
        if self.mode == 'min':
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = current_value
            self.counter = 0
            logger.info(f"✓ {self.monitor} improved to {current_value:.4f}")
        else:
            self.counter += 1
            logger.info(f"⚠ {self.monitor} did not improve ({self.counter}/{self.patience})")
        
        if self.counter >= self.patience:
            self.should_stop = True
            logger.warning(f"Early stopping triggered!")
        
        return self.should_stop


class LearningRateSchedulerCallback:
    """Callback for learning rate scheduling"""
    
    def __init__(self, scheduler):
        self.scheduler = scheduler
    
    def on_epoch_end(self, epoch, metrics):
        """Step scheduler after epoch"""
        if hasattr(self.scheduler, 'step'):
            # ReduceLROnPlateau needs metric
            if 'ReduceLROnPlateau' in str(type(self.scheduler)):
                self.scheduler.step(metrics['val_loss'])
            else:
                self.scheduler.step()
