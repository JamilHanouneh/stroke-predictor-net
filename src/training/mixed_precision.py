"""
Automatic Mixed Precision (AMP) utilities.
"""

import torch
import logging

logger = logging.getLogger(__name__)


def create_grad_scaler(use_amp):
    """
    Create gradient scaler for mixed precision.
    
    Args:
        use_amp (bool): Whether to use AMP
    
    Returns:
        GradScaler or None
    """
    if use_amp and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        logger.info("✓ Created GradScaler for mixed precision training")
        return scaler
    else:
        return None


def autocast_context(use_amp):
    """
    Get autocast context manager.
    
    Args:
        use_amp (bool): Whether to use AMP
    
    Returns:
        Context manager
    """
    if use_amp:
        return torch.cuda.amp.autocast()
    else:
        # No-op context manager
        from contextlib import nullcontext
        return nullcontext()
