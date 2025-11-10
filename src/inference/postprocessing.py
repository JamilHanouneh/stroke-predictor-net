"""
Post-processing for segmentation masks.
"""

import numpy as np
from scipy.ndimage import label, binary_fill_holes
import logging

logger = logging.getLogger(__name__)


def remove_small_components(mask, min_size=100):
    """
    Remove small connected components.
    
    Args:
        mask (np.ndarray): Binary mask
        min_size (int): Minimum component size (voxels)
    
    Returns:
        np.ndarray: Cleaned mask
    """
    labeled, num_features = label(mask)
    
    if num_features == 0:
        return mask
    
    # Compute component sizes
    sizes = np.bincount(labeled.ravel())[1:]  # Exclude background
    
    # Remove small components
    for i, size in enumerate(sizes, start=1):
        if size < min_size:
            mask[labeled == i] = 0
    
    return mask


def fill_holes(mask):
    """
    Fill holes in binary mask.
    
    Args:
        mask (np.ndarray): Binary mask
    
    Returns:
        np.ndarray: Filled mask
    """
    return binary_fill_holes(mask).astype(mask.dtype)


def postprocess_prediction(prediction, min_component_size=100, fill_holes_flag=True):
    """
    Apply post-processing to prediction.
    
    Args:
        prediction (np.ndarray): Predicted mask
        min_component_size (int): Minimum component size
        fill_holes_flag (bool): Fill holes
    
    Returns:
        np.ndarray: Post-processed mask
    """
    # Binarize
    binary_mask = (prediction > 0.5).astype(np.uint8)
    
    # Remove small components
    if min_component_size > 0:
        binary_mask = remove_small_components(binary_mask, min_component_size)
    
    # Fill holes
    if fill_holes_flag:
        binary_mask = fill_holes(binary_mask)
    
    return binary_mask
