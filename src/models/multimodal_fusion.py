"""
Complete multimodal fusion network.
Combines imaging encoder, clinical encoder, cross-attention, and segmentation head.
"""

import torch
import torch.nn as nn
import logging

from .imaging_encoder import ImagingEncoder
from .clinical_encoder import ClinicalEncoder
from .cross_attention import CrossAttentionFusion
from .segmentation_head import SegmentationHead

logger = logging.getLogger(__name__)


class MultimodalStrokeNet(nn.Module):
    """
    Complete multimodal network for stroke outcome prediction.
    
    Architecture:
    1. Imaging Encoder: 3D CNN (ResNet3D) for MRI
    2. Clinical Encoder: MLP for patient metadata
    3. Cross-Attention Fusion: Learn multimodal interactions
    4. Segmentation Head: U-Net decoder for lesion prediction
    """
    
    def __init__(self, config):
        """
        Initialize complete model.
        
        Args:
            config (dict): Model configuration
        """
        super().__init__()
        
        self.config = config
        
        # Extract config
        img_config = config['model']['imaging_encoder']
        clin_config = config['model']['clinical_encoder']
        fusion_config = config['model']['fusion']
        seg_config = config['model']['segmentation_head']
        
        # Build submodules
        self.imaging_encoder = ImagingEncoder(
            in_channels=len(config['data']['modalities']),
            feature_dim=img_config['feature_dim'],
            channels=[16, 32, 64, 128],  # Adjust based on config
            dropout=img_config['dropout']
        )
        
        self.clinical_encoder = ClinicalEncoder(
            input_dim=clin_config['input_dim'],
            hidden_dims=clin_config['hidden_dims'],
            output_dim=clin_config['output_dim'],
            dropout=clin_config['dropout'],
            batch_norm=clin_config['batch_norm']
        )
        
        self.fusion = CrossAttentionFusion(
            imaging_dim=img_config['feature_dim'],
            clinical_dim=clin_config['output_dim'],
            attention_dim=fusion_config['attention_dim'],
            num_heads=fusion_config['attention_heads'],
            dropout=fusion_config['dropout']
        )
        
        self.segmentation_head = SegmentationHead(
            feature_dim=fusion_config['attention_dim'],
            decoder_channels=seg_config['decoder_channels'],
            num_classes=seg_config['num_classes'],
            spatial_dims=tuple(config['data']['preprocessing']['target_shape'])
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info("=" * 70)
        logger.info("MultimodalStrokeNet Architecture")
        logger.info("=" * 70)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Model size: ~{total_params * 4 / 1e6:.2f} MB (float32)")
        logger.info("=" * 70)
    
    def forward(self, imaging, clinical):
        """
        Forward pass.
        
        Args:
            imaging (Tensor): MRI data [batch, 3, H, W, D]
            clinical (Tensor): Clinical features [batch, num_features]
        
        Returns:
            dict: Output dictionary containing:
                - segmentation: Lesion predictions [batch, 1, H, W, D]
                - attention_weights: Attention weights for visualization
        """
        # Encode imaging
        img_features = self.imaging_encoder(imaging)  # [batch, img_dim]
        
        # Encode clinical
        clin_features = self.clinical_encoder(clinical)  # [batch, clin_dim]
        
        # Fuse with cross-attention
        fused_features, attention_weights = self.fusion(img_features, clin_features)
        
        # Generate segmentation
        segmentation = self.segmentation_head(fused_features)
        
        return {
            'segmentation': segmentation,
            'attention_weights': attention_weights,
            'img_features': img_features,
            'clin_features': clin_features
        }
    
    def get_num_parameters(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
