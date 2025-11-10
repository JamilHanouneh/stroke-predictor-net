"""
Loss functions for stroke segmentation.
Combined Dice loss and Cross-Entropy loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice loss for segmentation.
    Handles class imbalance well (common in medical imaging).
    """
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        """
        Compute Dice loss.
        
        Args:
            predictions (Tensor): Predicted probabilities [batch, 1, H, W, D]
            targets (Tensor): Ground truth masks [batch, 1, H, W, D]
        
        Returns:
            Tensor: Dice loss (scalar)
        """
        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Compute Dice coefficient
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum()
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return Dice loss (1 - Dice)
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """
    Combined Dice + Cross-Entropy loss.
    
    Benefits:
    - Dice handles class imbalance
    - CE provides stable gradients
    """
    
    def __init__(self, dice_weight=0.7, ce_weight=0.3, smooth=1.0):
        super().__init__()
        
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        
        self.dice_loss = DiceLoss(smooth=smooth)
        self.ce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, logits, targets):
        """
        Compute combined loss.
        
        Args:
            logits (Tensor): Raw model outputs [batch, 1, H, W, D]
            targets (Tensor): Ground truth masks [batch, 1, H, W, D]
        
        Returns:
            dict: Loss components
        """
        # Sigmoid for Dice loss
        probs = torch.sigmoid(logits)
        
        # Compute losses
        dice = self.dice_loss(probs, targets)
        ce = self.ce_loss(logits, targets)
        
        # Combined loss
        total_loss = self.dice_weight * dice + self.ce_weight * ce
        
        return {
            'total': total_loss,
            'dice': dice,
            'ce': ce
        }
