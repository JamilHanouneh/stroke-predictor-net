"""
Clinical feature importance analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import logging

logger = logging.getLogger(__name__)


def plot_feature_importance(feature_names, importance_scores, save_path=None):
    """
    Plot feature importance.
    
    Args:
        feature_names (list): Feature names
        importance_scores (np.ndarray): Importance scores
        save_path (str, optional): Save path
    """
    # Sort by importance
    sorted_idx = np.argsort(importance_scores)
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_scores = importance_scores[sorted_idx]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(sorted_features))
    ax.barh(y_pos, sorted_scores, align='center', color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title('Clinical Feature Importance', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved feature importance to {save_path}")
    else:
        plt.show()
    
    plt.close()
