"""
Validation loop manager.
"""

import torch
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class Validator:
    """
    Validation loop manager.
    
    Handles validation and metric computation.
    """
    
    def __init__(self, model, criterion, metrics_calculator, device_mgr):
        """
        Initialize validator.
        
        Args:
            model: Neural network
            criterion: Loss function
            metrics_calculator: Metrics calculator
            device_mgr: Device manager
        """
        self.model = model
        self.criterion = criterion
        self.metrics_calculator = metrics_calculator
        self.device_mgr = device_mgr
    
    def validate(self, val_loader):
        """
        Run validation.
        
        Args:
            val_loader: Validation DataLoader
        
        Returns:
            dict: Validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        all_dice = []
        all_iou = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                # Move to device
                imaging = self.device_mgr.to_device(batch['imaging'])
                clinical = self.device_mgr.to_device(batch['clinical'])
                masks = self.device_mgr.to_device(batch['mask'])
                
                # Forward pass
                outputs = self.model(imaging, clinical)
                loss_dict = self.criterion(outputs['segmentation'], masks)
                
                # Accumulate loss
                total_loss += loss_dict['total'].item()
                
                # Compute metrics
                probs = torch.sigmoid(outputs['segmentation'])
                metrics = self.metrics_calculator.compute_all_metrics(probs, masks)
                
                all_dice.append(metrics['dice'])
                all_iou.append(metrics['iou'])
        
        # Average metrics
        results = {
            'loss': total_loss / len(val_loader),
            'dice': sum(all_dice) / len(all_dice),
            'iou': sum(all_iou) / len(all_iou)
        }
        
        return results
