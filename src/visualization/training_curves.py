"""
Plot training curves.
"""

import matplotlib.pyplot as plt
import numpy as np
import logging

logger = logging.getLogger(__name__)


def plot_learning_curves(train_losses, val_losses, train_metrics=None, val_metrics=None, save_path=None):
    """
    Plot training and validation curves.
    
    Args:
        train_losses (list): Training losses
        val_losses (list): Validation losses
        train_metrics (dict, optional): Training metrics
        val_metrics (dict, optional): Validation metrics
        save_path (str, optional): Save path
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curves
    axes[0].plot(epochs, train_losses, 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Metrics
    if train_metrics and val_metrics:
        metric_name = list(train_metrics.keys())[0]
        axes[1].plot(epochs, train_metrics[metric_name], 'b-', label='Train', linewidth=2)
        axes[1].plot(epochs, val_metrics[metric_name], 'r-', label='Validation', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel(metric_name.capitalize(), fontsize=12)
        axes[1].set_title(f'{metric_name.capitalize()} Score', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved training curves to {save_path}")
    else:
        plt.show()
    
    plt.close()
