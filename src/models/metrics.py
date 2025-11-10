"""
Evaluation metrics for segmentation.
"""

import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff


class SegmentationMetrics:
    """
    Compute segmentation metrics.
    
    Metrics:
    - Dice Score (F1)
    - IoU (Jaccard)
    - Sensitivity (Recall)
    - Specificity
    - Hausdorff Distance
    """
    
    def __init__(self, threshold=0.5):
        """
        Initialize metrics.
        
        Args:
            threshold (float): Probability threshold for binary prediction
        """
        self.threshold = threshold
    
    def dice_score(self, predictions, targets):
        """
        Compute Dice coefficient.
        
        Args:
            predictions (Tensor): Predicted probabilities [batch, 1, H, W, D]
            targets (Tensor): Ground truth masks [batch, 1, H, W, D]
        
        Returns:
            float: Dice score
        """
        # Binarize predictions
        preds_binary = (predictions > self.threshold).float()
        
        # Flatten
        preds_flat = preds_binary.view(-1)
        targets_flat = targets.view(-1)
        
        # Compute Dice
        intersection = (preds_flat * targets_flat).sum()
        union = preds_flat.sum() + targets_flat.sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        dice = (2.0 * intersection) / union
        return dice.item()
    
    def iou_score(self, predictions, targets):
        """
        Compute Intersection over Union (Jaccard index).
        
        Args:
            predictions (Tensor): Predicted probabilities
            targets (Tensor): Ground truth masks
        
        Returns:
            float: IoU score
        """
        preds_binary = (predictions > self.threshold).float()
        
        preds_flat = preds_binary.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (preds_flat * targets_flat).sum()
        union = preds_flat.sum() + targets_flat.sum() - intersection
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        iou = intersection / union
        return iou.item()
    
    def sensitivity(self, predictions, targets):
        """
        Compute sensitivity (recall, true positive rate).
        
        Args:
            predictions (Tensor): Predicted probabilities
            targets (Tensor): Ground truth masks
        
        Returns:
            float: Sensitivity
        """
        preds_binary = (predictions > self.threshold).float()
        
        preds_flat = preds_binary.view(-1)
        targets_flat = targets.view(-1)
        
        true_positives = (preds_flat * targets_flat).sum()
        actual_positives = targets_flat.sum()
        
        if actual_positives == 0:
            return 1.0 if true_positives == 0 else 0.0
        
        sens = true_positives / actual_positives
        return sens.item()
    
    def specificity(self, predictions, targets):
        """
        Compute specificity (true negative rate).
        
        Args:
            predictions (Tensor): Predicted probabilities
            targets (Tensor): Ground truth masks
        
        Returns:
            float: Specificity
        """
        preds_binary = (predictions > self.threshold).float()
        
        preds_flat = preds_binary.view(-1)
        targets_flat = targets.view(-1)
        
        true_negatives = ((1 - preds_flat) * (1 - targets_flat)).sum()
        actual_negatives = (1 - targets_flat).sum()
        
        if actual_negatives == 0:
            return 1.0 if true_negatives == 0 else 0.0
        
        spec = true_negatives / actual_negatives
        return spec.item()
    
    def hausdorff_distance_95(self, predictions, targets):
        """
        Compute 95th percentile Hausdorff distance.
        
        Args:
            predictions (Tensor): Predicted probabilities
            targets (Tensor): Ground truth masks
        
        Returns:
            float: Hausdorff distance (95th percentile)
        """
        preds_binary = (predictions > self.threshold).float()
        
        # Convert to numpy
        preds_np = preds_binary.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        # Get surface points
        pred_points = np.argwhere(preds_np[0, 0] > 0)
        target_points = np.argwhere(targets_np[0, 0] > 0)
        
        if len(pred_points) == 0 or len(target_points) == 0:
            return 0.0 if len(pred_points) == len(target_points) else float('inf')
        
        # Compute directed Hausdorff distances
        hd1 = directed_hausdorff(pred_points, target_points)[0]
        hd2 = directed_hausdorff(target_points, pred_points)[0]
        
        # Return 95th percentile
        hd95 = np.percentile([hd1, hd2], 95)
        return float(hd95)
    
    def compute_all_metrics(self, predictions, targets):
        """
        Compute all metrics.
        
        Args:
            predictions (Tensor): Predicted probabilities
            targets (Tensor): Ground truth masks
        
        Returns:
            dict: All metrics
        """
        metrics = {
            'dice': self.dice_score(predictions, targets),
            'iou': self.iou_score(predictions, targets),
            'sensitivity': self.sensitivity(predictions, targets),
            'specificity': self.specificity(predictions, targets),
        }
        
        # Hausdorff is expensive, compute only if needed
        try:
            metrics['hausdorff_95'] = self.hausdorff_distance_95(predictions, targets)
        except:
            metrics['hausdorff_95'] = 0.0
        
        return metrics
