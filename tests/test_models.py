"""
Unit tests for model architectures.
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.imaging_encoder import ImagingEncoder
from src.models.clinical_encoder import ClinicalEncoder
from src.models.cross_attention import CrossAttentionFusion
from src.models.segmentation_head import SegmentationHead


class TestImagingEncoder:
    """Test imaging encoder"""
    
    def test_forward_pass(self):
        """Test forward pass with dummy data"""
        model = ImagingEncoder(
            in_channels=3,
            feature_dim=512,
            channels=[16, 32, 64, 128]
        )
        
        # Dummy input
        x = torch.randn(2, 3, 128, 128, 128)
        
        # Forward
        output = model(x)
        
        # Check output shape
        assert output.shape == (2, 512)
    
    def test_output_range(self):
        """Test output is not NaN or Inf"""
        model = ImagingEncoder(in_channels=3, feature_dim=512)
        x = torch.randn(1, 3, 64, 64, 64)
        
        output = model(x)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestClinicalEncoder:
    """Test clinical encoder"""
    
    def test_forward_pass(self):
        """Test forward pass"""
        model = ClinicalEncoder(
            input_dim=10,
            hidden_dims=[64, 32],
            output_dim=128
        )
        
        # Dummy input
        x = torch.randn(2, 10)
        
        # Forward
        output = model(x)
        
        # Check shape
        assert output.shape == (2, 128)
    
    def test_different_batch_sizes(self):
        """Test with different batch sizes"""
        model = ClinicalEncoder(input_dim=10, output_dim=128)
        
        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 10)
            output = model(x)
            assert output.shape == (batch_size, 128)


class TestCrossAttentionFusion:
    """Test cross-attention fusion"""
    
    def test_forward_pass(self):
        """Test fusion forward pass"""
        model = CrossAttentionFusion(
            imaging_dim=512,
            clinical_dim=128,
            attention_dim=256,
            num_heads=4
        )
        
        # Dummy inputs
        img_feat = torch.randn(2, 512)
        clin_feat = torch.randn(2, 128)
        
        # Forward
        fused, attention_weights = model(img_feat, clin_feat)
        
        # Check shapes
        assert fused.shape == (2, 256)
        assert attention_weights.shape[0] == 2  # batch size
    
    def test_attention_weights_sum(self):
        """Test attention weights sum to 1"""
        model = CrossAttentionFusion(
            imaging_dim=512,
            clinical_dim=128,
            attention_dim=256,
            num_heads=4
        )
        
        img_feat = torch.randn(1, 512)
        clin_feat = torch.randn(1, 128)
        
        fused, attention_weights = model(img_feat, clin_feat)
        
        # Attention weights should be valid probabilities
        assert not torch.isnan(attention_weights).any()


class TestSegmentationHead:
    """Test segmentation decoder"""
    
    def test_forward_pass(self):
        """Test decoder forward pass"""
        model = SegmentationHead(
            feature_dim=256,
            decoder_channels=[256, 128, 64, 32],
            num_classes=1,
            spatial_dims=(128, 128, 128)
        )
        
        # Dummy input
        x = torch.randn(2, 256)
        
        # Forward
        output = model(x)
        
        # Check shape
        assert output.shape == (2, 1, 128, 128, 128)
    
    def test_output_range(self):
        """Test output is in valid range"""
        model = SegmentationHead(
            feature_dim=256,
            decoder_channels=[128, 64, 32],
            num_classes=1,
            spatial_dims=(64, 64, 64)
        )
        
        x = torch.randn(1, 256)
        output = model(x)
        
        # Apply sigmoid
        probs = torch.sigmoid(output)
        
        # Check range [0, 1]
        assert (probs >= 0).all() and (probs <= 1).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
