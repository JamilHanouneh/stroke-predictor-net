"""
Learning rate scheduler creation.
"""

import torch.optim.lr_scheduler as lr_scheduler
import logging

logger = logging.getLogger(__name__)


def create_scheduler(optimizer, config):
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        config (dict): Configuration
    
    Returns:
        Scheduler: Learning rate scheduler
    """
    sched_config = config['training']['scheduler']
    sched_type = sched_config['type'].lower()
    
    if sched_type == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs'],
            eta_min=sched_config['min_lr']
        )
    
    elif sched_type == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=sched_config['step_size'],
            gamma=sched_config['gamma']
        )
    
    elif sched_type == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=sched_config['mode'],
            factor=sched_config['factor'],
            patience=sched_config['patience'],
            threshold=sched_config['threshold']
        )
    
    elif sched_type == 'exponential':
        scheduler = lr_scheduler.ExponentialLR(
            optimizer,
            gamma=sched_config.get('gamma', 0.95)
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}")
    
    logger.info(f"✓ Created {sched_type} scheduler")
    
    return scheduler
