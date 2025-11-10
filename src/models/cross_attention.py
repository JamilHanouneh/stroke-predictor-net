"""
Cross-attention fusion module.
Allows imaging and clinical features to attend to each other.
"""

import torch
import torch.nn as nn
import math
import logging

logger = logging.getLogger(__name__)


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention between imaging and clinical features.
    
    Key innovation: Instead of simple concatenation, learns how
    imaging patterns relate to clinical context through attention.
    """
    
    def __init__(self,
                 imaging_dim=512,
                 clinical_dim=128,
                 attention_dim=256,
                 num_heads=4,
                 dropout=0.1):
        """
        Initialize cross-attention fusion.
        
        Args:
            imaging_dim (int): Imaging feature dimension
            clinical_dim (int): Clinical feature dimension
            attention_dim (int): Attention space dimension
            num_heads (int): Number of attention heads
            dropout (float): Dropout probability
        """
        super().__init__()
        
        self.imaging_dim = imaging_dim
        self.clinical_dim = clinical_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        
        # Project features to common dimension
        self.imaging_proj = nn.Linear(imaging_dim, attention_dim)
        self.clinical_proj = nn.Linear(clinical_dim, attention_dim)
        
        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.out_proj = nn.Linear(attention_dim * 2, attention_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(attention_dim)
        
        logger.info(f"Initialized CrossAttentionFusion:")
        logger.info(f"  Imaging: {imaging_dim} → {attention_dim}")
        logger.info(f"  Clinical: {clinical_dim} → {attention_dim}")
        logger.info(f"  Heads: {num_heads}")
        logger.info(f"  Parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, imaging_feat, clinical_feat):
        """
        Forward pass with cross-attention.
        
        Args:
            imaging_feat (Tensor): Imaging features [batch, imaging_dim]
            clinical_feat (Tensor): Clinical features [batch, clinical_dim]
        
        Returns:
            Tensor: Fused features [batch, attention_dim]
            Tensor: Attention weights [batch, num_heads, 1, 1]
        """
        batch_size = imaging_feat.size(0)
        
        # Project to attention space
        img_proj = self.imaging_proj(imaging_feat)  # [batch, attention_dim]
        clin_proj = self.clinical_proj(clinical_feat)  # [batch, attention_dim]
        
        # Add sequence dimension for attention
        # [batch, 1, attention_dim]
        img_proj = img_proj.unsqueeze(1)
        clin_proj = clin_proj.unsqueeze(1)
        
        # Cross-attention: imaging attends to clinical
        img_attn, img_weights = self.multihead_attn(
            query=img_proj,
            key=clin_proj,
            value=clin_proj
        )
        
        # Cross-attention: clinical attends to imaging
        clin_attn, clin_weights = self.multihead_attn(
            query=clin_proj,
            key=img_proj,
            value=img_proj
        )
        
        # Concatenate attended features
        fused = torch.cat([
            img_attn.squeeze(1),
            clin_attn.squeeze(1)
        ], dim=1)  # [batch, attention_dim * 2]
        
        # Output projection
        fused = self.out_proj(fused)  # [batch, attention_dim]
        fused = self.dropout(fused)
        fused = self.layer_norm(fused)
        
        return fused, img_weights
