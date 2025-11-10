"""
Clinical metadata loader.
Handles loading and preprocessing of clinical features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ClinicalDataLoader:
    """
    Loader for clinical metadata.
    
    Handles:
    - CSV loading
    - Feature normalization
    - Missing value handling
    """
    
    def __init__(self, csv_path, feature_config=None):
        """
        Initialize clinical data loader.
        
        Args:
            csv_path (str or Path): Path to clinical CSV
            feature_config (dict, optional): Feature configuration
        """
        self.csv_path = Path(csv_path)
        self.feature_config = feature_config or {}
        
        # Load data
        self.df = pd.read_csv(csv_path)
        logger.info(f"Loaded clinical data: {len(self.df)} subjects")
        
        # Set index
        if 'subject_id' in self.df.columns:
            self.df = self.df.set_index('subject_id')
    
    def get_features(self, subject_id):
        """
        Get clinical features for a subject.
        
        Args:
            subject_id (str): Subject ID
        
        Returns:
            np.ndarray: Clinical feature vector
        """
        if subject_id not in self.df.index:
            logger.warning(f"Subject {subject_id} not found in clinical data")
            # Return zeros
            return np.zeros(len(self.df.columns))
        
        features = self.df.loc[subject_id].values.astype(np.float32)
        return features
    
    def normalize_features(self, method='z_score'):
        """
        Normalize clinical features.
        
        Args:
            method (str): Normalization method ('z_score', 'min_max')
        """
        if method == 'z_score':
            self.df = (self.df - self.df.mean()) / self.df.std()
        elif method == 'min_max':
            self.df = (self.df - self.df.min()) / (self.df.max() - self.df.min())
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        logger.info(f"✓ Normalized features using {method}")
    
    def handle_missing_values(self, strategy='mean'):
        """
        Handle missing values.
        
        Args:
            strategy (str): Strategy ('mean', 'median', 'zero')
        """
        if strategy == 'mean':
            self.df = self.df.fillna(self.df.mean())
        elif strategy == 'median':
            self.df = self.df.fillna(self.df.median())
        elif strategy == 'zero':
            self.df = self.df.fillna(0)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        logger.info(f"✓ Handled missing values using {strategy}")
    
    def get_feature_names(self):
        """Get list of feature names"""
        return list(self.df.columns)
    
    def get_feature_statistics(self):
        """Get feature statistics"""
        stats = {
            'mean': self.df.mean().to_dict(),
            'std': self.df.std().to_dict(),
            'min': self.df.min().to_dict(),
            'max': self.df.max().to_dict()
        }
        return stats
