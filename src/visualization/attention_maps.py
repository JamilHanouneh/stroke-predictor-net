"""
Visualize cross-attention maps.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)


def plot_attention_heatmap(attention_weights, feature_names=None, save_path=None):
    """
    Plot attention weights as heatmap.
    
    Args:
        attention_weights (np.ndarray): Attention weights
        feature_names (list, optional): Feature names
        save_path (str, optional): Save path
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Average across heads if multi-head
    if len(attention_weights.shape) > 2:
        attention_weights = attention_weights.mean(axis=1)
    
    # Plot heatmap
    sns.heatmap(attention_weights, cmap='viridis', ax=ax, cbar=True)
    
    if feature_names:
        ax.set_yticklabels(feature_names, rotation=0)
    
    ax.set_title('Cross-Attention Weights', fontsize=16, fontweight='bold')
    ax.set_xlabel('Query')
    ax.set_ylabel('Key')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved attention heatmap to {save_path}")
    else:
        plt.show()
    
    plt.close()
