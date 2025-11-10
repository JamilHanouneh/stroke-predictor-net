"""
Training loop abstraction.
Encapsulates training logic for cleaner code.
"""

import torch
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class Trainer:
    """
    Training loop manager.
    
    Handles:
    - Forward/backward pass
    - Gradient accumulation
    - Mixed precision training
    - Metrics computation
    """
    
    def __init__(self, model, criterion, optimizer, device_mgr, scaler=None, grad_accum_steps=1):
        """
        Initialize trainer.
        
        Args:
            model: Neural network model
            criterion: Loss function
            optimizer: Optimizer
            device_mgr: Device manager
            scaler: GradScaler for mixed precision
            grad_accum_steps: Gradient accumulation steps
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device_mgr = device_mgr
        self.scaler = scaler
        self.grad_accum_steps = grad_accum_steps
    
    def train_epoch(self, train_loader, epoch):
        """
        Train for one epoch.
        
        Args:
            train_loader: Training dataloader
            epoch (int): Current epoch number
        
        Returns:
            dict: Training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            imaging = self.device_mgr.to_device(batch['imaging'])
            clinical = self.device_mgr.to_device(batch['clinical'])
            masks = self.device_mgr.to_device(batch['mask'])
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.device_mgr.use_amp):
                outputs = self.model(imaging, clinical)
                loss_dict = self.criterion(outputs['segmentation'], masks)
                loss = loss_dict['total'] / self.grad_accum_steps
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step with gradient accumulation
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Accumulate metrics
            total_loss += loss_dict['total'].item()
            num_batches += 1
            
            # Update progress
            progress_bar.set_postfix({'loss': f"{loss_dict['total'].item():.4f}"})
        
        return {'loss': total_loss / num_batches}
