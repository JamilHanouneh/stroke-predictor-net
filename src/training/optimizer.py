"""
Optimizer creation utilities.
"""

import torch.optim as optim
import logging

logger = logging.getLogger(__name__)


def create_optimizer(model, config):
    """
    Create optimizer from configuration.
    
    Args:
        model: Neural network model
        config (dict): Optimizer configuration
    
    Returns:
        torch.optim.Optimizer: Configured optimizer
    """
    opt_config = config['training']['optimizer']
    opt_type = opt_config['type'].lower()
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    
    if opt_type == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            betas=opt_config['betas'],
            eps=opt_config['eps'],
            weight_decay=weight_decay
        )
    
    elif opt_type == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=opt_config['betas'],
            eps=opt_config['eps'],
            weight_decay=weight_decay,
            amsgrad=opt_config.get('amsgrad', False)
        )
    
    elif opt_type == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=opt_config.get('momentum', 0.9),
            weight_decay=weight_decay,
            nesterov=opt_config.get('nesterov', True)
        )
    
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")
    
    logger.info(f"✓ Created {opt_type.upper()} optimizer with lr={lr}")
    
    return optimizer
