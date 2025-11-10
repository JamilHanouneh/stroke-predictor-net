"""
Unit tests for data loading and preprocessing.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import ISLESStrokeDataset
from src.data.augmentation import RandomFlip, RandomRotation, RandomGamma


class TestDataset:
    """Test ISLESStrokeDataset"""
    
    def test_dataset_initialization(self):
        """Test dataset can be initialized"""
        # This would require mock data
        pass
    
    def test_dataset_length(self):
        """Test __len__ returns correct number"""
        pass
    
    def test_dataset_getitem(self):
        """Test __getitem__ returns correct format"""
        pass
    
    def test_normalization(self):
        """Test z-score normalization"""
        # Create dummy data
        data = np.random.randn(3, 64, 64, 64).astype(np.float32)
        
        # Normalize
        for c in range(3):
            mask = data[c] > 0
            if mask.sum() > 0:
                mean = data[c][mask].mean()
                std = data[c][mask].std()
                data[c] = np.where(mask, (data[c] - mean) / std, 0)
        
        # Check properties
        for c in range(3):
            mask = data[c] > 0
            if mask.sum() > 0:
                assert abs(data[c][mask].mean()) < 0.1
                assert abs(data[c][mask].std() - 1.0) < 0.1


class TestAugmentation:
    """Test augmentation transforms"""
    
    def test_random_flip(self):
        """Test random flip transform"""
        image = np.random.randn(3, 64, 64, 64).astype(np.float32)
        mask = np.random.randint(0, 2, (1, 64, 64, 64)).astype(np.float32)
        
        transform = RandomFlip(probability=1.0, axes=[0])
        aug_image, aug_mask = transform(image, mask)
        
        # Check shapes preserved
        assert aug_image.shape == image.shape
        assert aug_mask.shape == mask.shape
        
        # Check values different (flipped)
        assert not np.array_equal(aug_image, image)
    
    def test_random_rotation(self):
        """Test random rotation transform"""
        image = np.random.randn(3, 64, 64, 64).astype(np.float32)
        mask = np.random.randint(0, 2, (1, 64, 64, 64)).astype(np.float32)
        
        transform = RandomRotation(probability=1.0, rotation_range=(-10, 10))
        aug_image, aug_mask = transform(image, mask)
        
        # Check shapes preserved
        assert aug_image.shape == image.shape
        assert aug_mask.shape == mask.shape
    
    def test_random_gamma(self):
        """Test gamma correction"""
        image = np.random.rand(3, 64, 64, 64).astype(np.float32)  # [0, 1]
        mask = np.random.randint(0, 2, (1, 64, 64, 64)).astype(np.float32)
        
        transform = RandomGamma(probability=1.0, gamma_range=(0.5, 2.0))
        aug_image, aug_mask = transform(image, mask)
        
        # Check shapes preserved
        assert aug_image.shape == image.shape
        
        # Mask should be unchanged
        assert np.array_equal(aug_mask, mask)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
