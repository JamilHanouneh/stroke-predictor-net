"""
Preprocessing utilities for medical images.
"""

import numpy as np
from scipy.ndimage import zoom, gaussian_filter
import logging

logger = logging.getLogger(__name__)


def resample_volume(volume, current_spacing, target_spacing, target_shape, order=1):
    """
    Resample volume to target spacing and shape.
    
    Args:
        volume (np.ndarray): Input volume
        current_spacing (tuple): Current voxel spacing (mm)
        target_spacing (tuple): Target voxel spacing (mm)
        target_shape (tuple): Target shape (H, W, D)
        order (int): Interpolation order (0=nearest, 1=linear, 3=cubic)
    
    Returns:
        np.ndarray: Resampled volume
    """
    # Calculate zoom factors
    zoom_factors = tuple(
        (c * cs) / (t * ts)
        for c, cs, t, ts in zip(volume.shape, current_spacing, target_shape, target_spacing)
    )
    
    # Resample
    resampled = zoom(volume, zoom_factors, order=order)
    
    # Ensure exact target shape (crop or pad)
    result = np.zeros(target_shape, dtype=volume.dtype)
    
    slices = tuple(
        slice(0, min(r, t))
        for r, t in zip(resampled.shape, target_shape)
    )
    
    result[slices] = resampled[slices]
    
    return result


def normalize_intensity(volume, method='z_score', percentiles=(1, 99)):
    """
    Normalize intensity values.
    
    Args:
        volume (np.ndarray): Input volume
        method (str): Normalization method
        percentiles (tuple): Percentiles for clipping
    
    Returns:
        np.ndarray: Normalized volume
    """
    # Create mask (exclude background)
    mask = volume > 0
    
    if mask.sum() == 0:
        return volume
    
    if method == 'z_score':
        # Z-score normalization
        mean = volume[mask].mean()
        std = volume[mask].std()
        
        if std > 0:
            normalized = np.where(mask, (volume - mean) / std, 0)
        else:
            normalized = volume
    
    elif method == 'min_max':
        # Min-max normalization
        vmin, vmax = volume[mask].min(), volume[mask].max()
        
        if vmax > vmin:
            normalized = np.where(mask, (volume - vmin) / (vmax - vmin), 0)
        else:
            normalized = volume
    
    elif method == 'percentile':
        # Percentile clipping + normalization
        p_low, p_high = np.percentile(volume[mask], percentiles)
        clipped = np.clip(volume, p_low, p_high)
        
        normalized = np.where(mask, (clipped - p_low) / (p_high - p_low), 0)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized


def apply_gaussian_smoothing(volume, sigma=1.0):
    """
    Apply Gaussian smoothing.
    
    Args:
        volume (np.ndarray): Input volume
        sigma (float): Gaussian kernel standard deviation
    
    Returns:
        np.ndarray: Smoothed volume
    """
    return gaussian_filter(volume, sigma=sigma)


def remove_background(volume, threshold=0):
    """
    Remove background voxels.
    
    Args:
        volume (np.ndarray): Input volume
        threshold (float): Background threshold
    
    Returns:
        np.ndarray: Volume with background removed
    """
    return np.where(volume > threshold, volume, 0)
