"""
3D brain visualization utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging

logger = logging.getLogger(__name__)


def plot_3d_volume(volume, threshold=0.5, save_path=None):
    """
    Plot 3D volume rendering.
    
    Args:
        volume (np.ndarray): 3D volume
        threshold (float): Threshold for visualization
        save_path (str, optional): Path to save figure
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get coordinates where volume > threshold
    x, y, z = np.where(volume > threshold)
    
    # Plot
    ax.scatter(x, y, z, c=volume[volume > threshold], cmap='hot', s=1, alpha=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Volume Rendering')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved 3D plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_orthogonal_slices(volume, slice_indices=None, save_path=None):
    """
    Plot orthogonal slices (axial, sagittal, coronal).
    
    Args:
        volume (np.ndarray): 3D volume [H, W, D]
        slice_indices (tuple, optional): (axial, sagittal, coronal) indices
        save_path (str, optional): Save path
    """
    if slice_indices is None:
        # Use middle slices
        slice_indices = tuple(s // 2 for s in volume.shape)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Axial
    axes[0].imshow(volume[:, :, slice_indices[2]], cmap='gray')
    axes[0].set_title(f'Axial (z={slice_indices[2]})')
    axes[0].axis('off')
    
    # Sagittal
    axes[1].imshow(volume[slice_indices[0], :, :], cmap='gray')
    axes[1].set_title(f'Sagittal (x={slice_indices[0]})')
    axes[1].axis('off')
    
    # Coronal
    axes[2].imshow(volume[:, slice_indices[1], :], cmap='gray')
    axes[2].set_title(f'Coronal (y={slice_indices[1]})')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved orthogonal slices to {save_path}")
    else:
        plt.show()
    
    plt.close()
