"""
Visualization of segmentation overlays on MRI.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import logging

logger = logging.getLogger(__name__)


def plot_segmentation_overlay(imaging, mask, prediction, slice_idx=None, save_path=None):
    """
    Plot MRI with ground truth and predicted masks.
    
    Args:
        imaging (np.ndarray): MRI volume [C, H, W, D]
        mask (np.ndarray): Ground truth mask [1, H, W, D]
        prediction (np.ndarray): Predicted mask [1, H, W, D]
        slice_idx (int, optional): Slice index. If None, use middle slice
        save_path (str, optional): Path to save figure
    """
    if slice_idx is None:
        slice_idx = imaging.shape[-1] // 2
    
    # Get slices
    img_slice = imaging[0, :, :, slice_idx]  # First modality (FLAIR)
    mask_slice = mask[0, :, :, slice_idx]
    pred_slice = prediction[0, :, :, slice_idx]
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original MRI
    axes[0].imshow(img_slice, cmap='gray')
    axes[0].set_title('MRI (FLAIR)')
    axes[0].axis('off')
    
    # Ground truth overlay
    axes[1].imshow(img_slice, cmap='gray')
    axes[1].imshow(mask_slice, cmap='Reds', alpha=0.5)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Prediction overlay
    axes[2].imshow(img_slice, cmap='gray')
    axes[2].imshow(pred_slice, cmap='Blues', alpha=0.5)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    # Comparison
    axes[3].imshow(img_slice, cmap='gray')
    axes[3].imshow(mask_slice, cmap='Reds', alpha=0.3, label='Ground Truth')
    axes[3].imshow(pred_slice, cmap='Blues', alpha=0.3, label='Prediction')
    axes[3].set_title('Overlay')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()
