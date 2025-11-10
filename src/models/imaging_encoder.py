"""
3D imaging encoder for multi-sequence MRI.
Uses ResNet3D backbone to extract spatial features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class ResBlock3D(nn.Module):
    """3D Residual block"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class ImagingEncoder(nn.Module):
    """
    3D ResNet-style encoder for multi-sequence MRI.
    
    Input: [batch, 3, H, W, D] (FLAIR, DWI, ADC)
    Output: [batch, feature_dim] (global features)
    """
    
    def __init__(self, 
                 in_channels=3,
                 feature_dim=512,
                 channels=[16, 32, 64, 128],
                 dropout=0.3):
        """
        Initialize imaging encoder.
        
        Args:
            in_channels (int): Number of input modalities (3 for FLAIR/DWI/ADC)
            feature_dim (int): Output feature dimension
            channels (list): Channel progression [C1, C2, C3, C4]
            dropout (float): Dropout probability
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        
        # Initial convolution
        self.conv1 = nn.Conv3d(in_channels, channels[0], kernel_size=7,
                              stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(channels[0], channels[0], num_blocks=2, stride=1)
        self.layer2 = self._make_layer(channels[0], channels[1], num_blocks=2, stride=2)
        self.layer3 = self._make_layer(channels[1], channels[2], num_blocks=2, stride=2)
        self.layer4 = self._make_layer(channels[2], channels[3], num_blocks=2, stride=2)
        
        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Feature projection
        self.fc = nn.Linear(channels[3], feature_dim)
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"Initialized ImagingEncoder: {in_channels} → {feature_dim}")
        logger.info(f"  Channels: {channels}")
        logger.info(f"  Parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """Create a layer with multiple residual blocks"""
        layers = []
        
        # First block (with stride for downsampling)
        layers.append(ResBlock3D(in_channels, out_channels, stride))
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(ResBlock3D(out_channels, out_channels, stride=1))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (Tensor): Input [batch, 3, H, W, D]
        
        Returns:
            Tensor: Features [batch, feature_dim]
        """
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Feature projection
        x = self.fc(x)
        x = self.dropout(x)
        
        return x
