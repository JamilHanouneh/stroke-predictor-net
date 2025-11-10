"""
Unit tests for training utilities.
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.loss import DiceLoss, CombinedLoss
from src.models.metrics import SegmentationMetrics


class TestLossFunctions:
    """Test loss functions"""
    
    def test_dice_loss_perfect_match(self):
        """Test Dice loss with perfect prediction"""
        criterion = DiceLoss()
        
        # Perfect prediction
        pred = torch.ones(1, 1, 32, 32, 32)
        target = torch.ones(1, 1, 32, 32, 32)
        
        loss = criterion(pred, target)
        
        # Loss should be close to 0
        assert loss.item() < 0.01
    
    def test_dice_loss_no_overlap(self):
        """Test Dice loss with no overlap"""
        criterion = DiceLoss()
        
        # No overlap
        pred = torch.ones(1, 1, 32, 32, 32)
        target = torch.zeros(1, 1, 32, 32, 32)
        
        loss = criterion(pred, target)
        
        # Loss should be close to 1
        assert loss.item() > 0.9
    
    def test_combined_loss(self):
        """Test combined Dice + CE loss"""
        criterion = CombinedLoss(dice_weight=0.7, ce_weight=0.3)
        
        logits = torch.randn(2, 1, 32, 32, 32)
        targets = torch.randint(0, 2, (2, 1, 32, 32, 32)).float()
        
        loss_dict = criterion(logits, targets)
        
        # Check loss components exist
        assert 'total' in loss_dict
        assert 'dice' in loss_dict
        assert 'ce' in loss_dict
        
        # Check loss is positive
        assert loss_dict['total'].item() >= 0


class TestMetrics:
    """Test evaluation metrics"""
    
    def test_dice_score_perfect(self):
        """Test Dice score with perfect prediction"""
        metrics = SegmentationMetrics(threshold=0.5)
        
        pred = torch.ones(1, 1, 32, 32, 32)
        target = torch.ones(1, 1, 32, 32, 32)
        
        dice = metrics.dice_score(pred, target)
        
        assert dice == 1.0
    
    def test_dice_score_no_overlap(self):
        """Test Dice score with no overlap"""
        metrics = SegmentationMetrics(threshold=0.5)
        
        pred = torch.ones(1, 1, 32, 32, 32)
        target = torch.zeros(1, 1, 32, 32, 32)
        
        dice = metrics.dice_score(pred, target)
        
        assert dice == 0.0
    
    def test_iou_score(self):
        """Test IoU score"""
        metrics = SegmentationMetrics(threshold=0.5)
        
        pred = torch.ones(1, 1, 32, 32, 32)
        pred[:, :, 16:, :, :] = 0  # Half zeros
        
        target = torch.ones(1, 1, 32, 32, 32)
        target[:, :, :16, :, :] = 0  # Other half zeros
        
        iou = metrics.iou_score(pred, target)
        
        # No overlap, so IoU should be 0
        assert iou == 0.0
    
    def test_sensitivity_specificity(self):
        """Test sensitivity and specificity"""
        metrics = SegmentationMetrics(threshold=0.5)
        
        # All positive predictions
        pred = torch.ones(1, 1, 32, 32, 32)
        target = torch.ones(1, 1, 32, 32, 32)
        
        sens = metrics.sensitivity(pred, target)
        spec = metrics.specificity(pred, target)
        
        # Perfect sensitivity
        assert sens == 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
