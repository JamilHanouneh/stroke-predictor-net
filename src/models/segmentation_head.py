"""
3D U-Net style decoder for lesion segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class UpBlock(nn.Module):
    """3D upsampling block"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.up = nn.ConvTranspose3d(in_channels, out_channels,
                                     kernel_size=2, stride=2)
        self.conv1 = nn.Conv3d(out_channels, out_channels,
                              kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels,
                              kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.up(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class SegmentationHead(nn.Module):
    """
    U-Net style decoder for 3D segmentation.
    
    Takes fused features and produces lesion segmentation.
    """
    
    def __init__(self,
                 feature_dim=256,
                 decoder_channels=[256, 128, 64, 32],
                 num_classes=1,
                 spatial_dims=(128, 128, 128)):
        """
        Initialize segmentation head.
        
        Args:
            feature_dim (int): Input feature dimension from fusion
            decoder_channels (list): Decoder channel progression
            num_classes (int): Number of output classes (1 for binary)
            spatial_dims (tuple): Target spatial dimensions (H, W, D)
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.spatial_dims = spatial_dims
        
        # Initial projection to spatial features
        # From [batch, feature_dim] to [batch, C, H, W, D]
        self.initial_spatial_dim = (8, 8, 8)  # Small spatial size
        self.initial_channels = decoder_channels[0]
        
        self.fc_to_spatial = nn.Linear(
            feature_dim,
            self.initial_channels * np.prod(self.initial_spatial_dim)
        )
        
        # Decoder blocks (upsampling)
        self.decoder = nn.ModuleList()
        in_ch = decoder_channels[0]
        
        for out_ch in decoder_channels[1:]:
            self.decoder.append(UpBlock(in_ch, out_ch))
            in_ch = out_ch
        
        # Final upsampling to match input size
        # Calculate required upsampling factor
        current_size = self.initial_spatial_dim[0] * (2 ** len(decoder_channels[1:]))
        remaining_factor = spatial_dims[0] // current_size
        
        if remaining_factor > 1:
            self.final_up = nn.Upsample(
                scale_factor=remaining_factor,
                mode='trilinear',
                align_corners=True
            )
        else:
            self.final_up = nn.Identity()
        
        # Final convolution
        self.final_conv = nn.Conv3d(
            decoder_channels[-1],
            num_classes,
            kernel_size=1
        )
        
        logger.info(f"Initialized SegmentationHead:")
        logger.info(f"  Feature dim: {feature_dim}")
        logger.info(f"  Decoder channels: {decoder_channels}")
        logger.info(f"  Output shape: {num_classes} x {spatial_dims}")
        logger.info(f"  Parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, fused_features):
        """
        Forward pass.
        
        Args:
            fused_features (Tensor): Fused features [batch, feature_dim]
        
        Returns:
            Tensor: Segmentation logits [batch, num_classes, H, W, D]
        """
        batch_size = fused_features.size(0)
        
        # Project to spatial features
        x = self.fc_to_spatial(fused_features)  # [batch, C*H*W*D]
        
        # Reshape to spatial
        x = x.view(
            batch_size,
            self.initial_channels,
            *self.initial_spatial_dim
        )  # [batch, C, H, W, D]
        
        # Decode (upsample)
        for decoder_block in self.decoder:
            x = decoder_block(x)
        
        # Final upsampling
        x = self.final_up(x)
        
        # Final convolution
        x = self.final_conv(x)
        
        return x


import numpy as np
