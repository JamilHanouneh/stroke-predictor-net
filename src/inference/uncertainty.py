"""
Uncertainty quantification for predictions.
"""

import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


def enable_dropout(model):
    """Enable dropout layers during inference"""
    for module in model.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()


def mc_dropout_predict(model, imaging, clinical, num_samples=10):
    """
    Monte Carlo Dropout for uncertainty estimation.
    
    Args:
        model: Neural network
        imaging: Input imaging
        clinical: Clinical features
        num_samples (int): Number of MC samples
    
    Returns:
        dict: Mean prediction and uncertainty
    """
    # Enable dropout
    enable_dropout(model)
    
    predictions = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            outputs = model(imaging, clinical)
            probs = torch.sigmoid(outputs['segmentation'])
            predictions.append(probs.cpu().numpy())
    
    # Stack predictions
    predictions = np.stack(predictions, axis=0)
    
    # Compute statistics
    mean = predictions.mean(axis=0)
    std = predictions.std(axis=0)
    entropy = -mean * np.log(mean + 1e-10) - (1 - mean) * np.log(1 - mean + 1e-10)
    
    return {
        'mean': mean,
        'std': std,
        'entropy': entropy
    }
