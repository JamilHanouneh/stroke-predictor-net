"""
Predictor class for model inference.
Encapsulates prediction logic.
"""

import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


class StrokePredictor:
    """
    Stroke outcome predictor.
    
    Handles:
    - Model loading
    - Data preprocessing
    - Inference
    - Post-processing
    """
    
    def __init__(self, model, device_mgr, threshold=0.5):
        """
        Initialize predictor.
        
        Args:
            model: Trained model
            device_mgr: Device manager
            threshold (float): Binary prediction threshold
        """
        self.model = model
        self.device_mgr = device_mgr
        self.threshold = threshold
        
        self.model.eval()
    
    def predict(self, imaging, clinical):
        """
        Make prediction.
        
        Args:
            imaging (torch.Tensor): MRI data [batch, C, H, W, D]
            clinical (torch.Tensor): Clinical features [batch, num_features]
        
        Returns:
            dict: Predictions with probabilities and binary masks
        """
        # Move to device
        imaging = self.device_mgr.to_device(imaging)
        clinical = self.device_mgr.to_device(clinical)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(imaging, clinical)
            logits = outputs['segmentation']
            probs = torch.sigmoid(logits)
        
        # Binary prediction
        binary = (probs > self.threshold).float()
        
        return {
            'probabilities': probs.cpu().numpy(),
            'binary': binary.cpu().numpy(),
            'attention_weights': outputs['attention_weights'].cpu().numpy()
        }
    
    def predict_with_uncertainty(self, imaging, clinical, num_samples=10):
        """
        Predict with uncertainty using MC Dropout.
        
        Args:
            imaging: MRI data
            clinical: Clinical features
            num_samples (int): Number of MC samples
        
        Returns:
            dict: Predictions with uncertainty
        """
        # Enable dropout during inference
        self.model.train()
        
        imaging = self.device_mgr.to_device(imaging)
        clinical = self.device_mgr.to_device(clinical)
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                outputs = self.model(imaging, clinical)
                probs = torch.sigmoid(outputs['segmentation'])
                predictions.append(probs.cpu().numpy())
        
        # Stack predictions
        predictions = np.stack(predictions, axis=0)  # [num_samples, batch, 1, H, W, D]
        
        # Compute mean and std
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        
        # Restore eval mode
        self.model.eval()
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'samples': predictions
        }
