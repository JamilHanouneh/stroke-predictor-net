"""
Clinical feature encoder.
MLP that processes patient metadata.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class ClinicalEncoder(nn.Module):
    """
    Multi-layer perceptron for clinical features.
    
    Input: [batch, num_features] (age, NIHSS, BP, etc.)
    Output: [batch, output_dim] (clinical embeddings)
    """
    
    def __init__(self,
                 input_dim=10,
                 hidden_dims=[64, 32],
                 output_dim=128,
                 dropout=0.2,
                 batch_norm=True):
        """
        Initialize clinical encoder.
        
        Args:
            input_dim (int): Number of clinical features
            hidden_dims (list): Hidden layer dimensions
            output_dim (int): Output embedding dimension
            dropout (float): Dropout probability
            batch_norm (bool): Use batch normalization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build MLP
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            
            in_dim = hidden_dim
        
        # Final projection
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        logger.info(f"Initialized ClinicalEncoder: {input_dim} → {output_dim}")
        logger.info(f"  Hidden dims: {hidden_dims}")
        logger.info(f"  Parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (Tensor): Clinical features [batch, num_features]
        
        Returns:
            Tensor: Clinical embeddings [batch, output_dim]
        """
        return self.mlp(x)
