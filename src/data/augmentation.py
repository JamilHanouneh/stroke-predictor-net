"""
Medical image augmentation for training.
Spatial and intensity transformations.
"""

import numpy as np
import torch
from scipy.ndimage import rotate, affine_transform, gaussian_filter
import logging

logger = logging.getLogger(__name__)


class Compose:
    """Compose multiple transforms"""
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, mask):
        for transform in self.transforms:
            image, mask = transform(image, mask)
        return image, mask


class RandomFlip:
    """Random flip along axes"""
    
    def __init__(self, probability=0.5, axes=(0, 1, 2)):
        self.probability = probability
        self.axes = axes
    
    def __call__(self, image, mask):
        if np.random.random() < self.probability:
            axis = np.random.choice(self.axes)
            image = np.flip(image, axis=axis + 1).copy()  # +1 for channel dim
            mask = np.flip(mask, axis=axis + 1).copy()
        return image, mask


class RandomRotation:
    """Random 3D rotation"""
    
    def __init__(self, probability=0.3, rotation_range=(-10, 10)):
        self.probability = probability
        self.rotation_range = rotation_range
    
    def __call__(self, image, mask):
        if np.random.random() < self.probability:
            angle = np.random.uniform(*self.rotation_range)
            axes = np.random.choice([0, 1, 2], size=2, replace=False)
            
            # Rotate each channel
            for c in range(image.shape[0]):
                image[c] = rotate(image[c], angle, axes=axes, reshape=False, order=1)
            
            mask[0] = rotate(mask[0], angle, axes=axes, reshape=False, order=0)
        
        return image, mask


class RandomGamma:
    """Random gamma correction"""
    
    def __init__(self, probability=0.3, gamma_range=(0.7, 1.5)):
        self.probability = probability
        self.gamma_range = gamma_range
    
    def __call__(self, image, mask):
        if np.random.random() < self.probability:
            gamma = np.random.uniform(*self.gamma_range)
            
            for c in range(image.shape[0]):
                # Normalize to [0, 1]
                img_min, img_max = image[c].min(), image[c].max()
                if img_max > img_min:
                    img_norm = (image[c] - img_min) / (img_max - img_min)
                    img_gamma = np.power(img_norm, gamma)
                    image[c] = img_gamma * (img_max - img_min) + img_min
        
        return image, mask


class RandomNoise:
    """Add random Gaussian noise"""
    
    def __init__(self, probability=0.2, noise_std=0.1):
        self.probability = probability
        self.noise_std = noise_std
    
    def __call__(self, image, mask):
        if np.random.random() < self.probability:
            noise = np.random.normal(0, self.noise_std, image.shape)
            image = image + noise
        
        return image, mask


class RandomBlur:
    """Random Gaussian blur"""
    
    def __init__(self, probability=0.2, sigma_range=(0.5, 1.5)):
        self.probability = probability
        self.sigma_range = sigma_range
    
    def __call__(self, image, mask):
        if np.random.random() < self.probability:
            sigma = np.random.uniform(*self.sigma_range)
            
            for c in range(image.shape[0]):
                image[c] = gaussian_filter(image[c], sigma=sigma)
        
        return image, mask


def get_training_augmentation(config):
    """
    Get training augmentation pipeline from config.
    
    Args:
        config (dict): Augmentation configuration
    
    Returns:
        Compose: Composed augmentation pipeline
    """
    transforms = []
    
    if not config.get('enabled', True):
        return None
    
    aug_config = config.get('augmentation', {})
    spatial = aug_config.get('spatial', {})
    intensity = aug_config.get('intensity', {})
    
    # Spatial augmentations
    if spatial.get('random_flip', {}).get('enabled', True):
        transforms.append(RandomFlip(
            probability=spatial['random_flip'].get('probability', 0.5),
            axes=spatial['random_flip'].get('axes', [0, 1, 2])
        ))
    
    if spatial.get('random_rotation', {}).get('enabled', True):
        transforms.append(RandomRotation(
            probability=spatial['random_rotation'].get('probability', 0.3),
            rotation_range=spatial['random_rotation'].get('rotation_range', [-10, 10])
        ))
    
    # Intensity augmentations
    if intensity.get('random_gamma', {}).get('enabled', True):
        transforms.append(RandomGamma(
            probability=intensity['random_gamma'].get('probability', 0.3),
            gamma_range=intensity['random_gamma'].get('gamma_range', [0.7, 1.5])
        ))
    
    if intensity.get('random_noise', {}).get('enabled', True):
        transforms.append(RandomNoise(
            probability=intensity['random_noise'].get('probability', 0.2),
            noise_std=intensity['random_noise'].get('noise_std', 0.1)
        ))
    
    if intensity.get('random_blur', {}).get('enabled', True):
        transforms.append(RandomBlur(
            probability=intensity['random_blur'].get('probability', 0.2),
            sigma_range=intensity['random_blur'].get('sigma_range', [0.5, 1.5])
        ))
    
    if transforms:
        logger.info(f"✓ Loaded {len(transforms)} augmentation transforms")
        return Compose(transforms)
    else:
        return None
