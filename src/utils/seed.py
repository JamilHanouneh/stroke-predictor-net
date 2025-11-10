"""
Random seed utilities for reproducibility.
"""

import random
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


def set_seed(seed, deterministic=True):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed
        deterministic (bool): Use deterministic algorithms (slower but reproducible)
    """
    logger.info(f"Setting random seed: {seed}")
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    if deterministic:
        # Make operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set CUBLAS workspace config (PyTorch 1.8+)
        import os
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        # Use deterministic algorithms
        torch.use_deterministic_algorithms(True, warn_only=True)
        
        logger.info("✓ Deterministic mode enabled (full reproducibility)")
    else:
        # Allow non-deterministic for speed
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        
        logger.info("✓ Non-deterministic mode (faster training)")
