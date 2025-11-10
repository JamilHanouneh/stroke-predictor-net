"""
PyTorch Dataset for ISLES 2022 stroke segmentation.
Loads MRI modalities and clinical features.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ISLESStrokeDataset(Dataset):
    """
    PyTorch Dataset for ISLES 2022 stroke data.
    
    Loads:
    - Multi-sequence MRI (FLAIR, DWI, ADC)
    - Clinical metadata
    - Lesion segmentation masks
    """
    
    def __init__(self, 
                 imaging_h5_path,
                 clinical_csv_path,
                 masks_h5_path,
                 subject_ids,
                 transform=None,
                 normalize=True):
        """
        Initialize dataset.
        
        Args:
            imaging_h5_path (str): Path to HDF5 file with imaging data
            clinical_csv_path (str): Path to CSV with clinical features
            masks_h5_path (str): Path to HDF5 file with lesion masks
            subject_ids (list): List of subject IDs to include
            transform (callable, optional): Transform to apply
            normalize (bool): Apply z-score normalization
        """
        self.imaging_h5_path = Path(imaging_h5_path)
        self.clinical_csv_path = Path(clinical_csv_path)
        self.masks_h5_path = Path(masks_h5_path)
        self.subject_ids = subject_ids
        self.transform = transform
        self.normalize = normalize
        
        # Load clinical features
        self.clinical_df = pd.read_csv(clinical_csv_path)
        self.clinical_df = self.clinical_df[self.clinical_df['subject_id'].isin(subject_ids)]
        self.clinical_df = self.clinical_df.set_index('subject_id')
        
        logger.info(f"Initialized ISLES dataset: {len(self)} subjects")
    
    def __len__(self):
        """Return dataset size"""
        return len(self.subject_ids)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Args:
            idx (int): Sample index
        
        Returns:
            dict: Sample containing:
                - imaging: Tensor [C, H, W, D] (3 modalities)
                - clinical: Tensor [num_features]
                - mask: Tensor [1, H, W, D]
                - subject_id: str
        """
        subject_id = self.subject_ids[idx]
        
        # Load imaging data
        with h5py.File(self.imaging_h5_path, 'r') as f:
            imaging = f[subject_id][()]  # Shape: [3, H, W, D]
        
        # Load mask
        with h5py.File(self.masks_h5_path, 'r') as f:
            mask = f[subject_id][()]  # Shape: [1, H, W, D]
        
        # Get clinical features
        clinical = self.clinical_df.loc[subject_id].values.astype(np.float32)
        
        # Normalize imaging
        if self.normalize:
            imaging = self._normalize(imaging)
        
        # Apply transforms
        if self.transform is not None:
            imaging, mask = self.transform(imaging, mask)
        
        # Convert to tensors
        imaging = torch.from_numpy(imaging).float()
        clinical = torch.from_numpy(clinical).float()
        mask = torch.from_numpy(mask).float()
        
        return {
            'imaging': imaging,
            'clinical': clinical,
            'mask': mask,
            'subject_id': subject_id
        }
    
    def _normalize(self, imaging):
        """
        Apply z-score normalization per modality.
        
        Args:
            imaging (np.ndarray): Imaging array [C, H, W, D]
        
        Returns:
            np.ndarray: Normalized imaging
        """
        normalized = np.zeros_like(imaging)
        
        for c in range(imaging.shape[0]):
            modality = imaging[c]
            
            # Only normalize non-zero voxels (exclude background)
            mask = modality > 0
            if mask.sum() > 0:
                mean = modality[mask].mean()
                std = modality[mask].std()
                
                if std > 0:
                    normalized[c] = np.where(mask, (modality - mean) / std, 0)
                else:
                    normalized[c] = modality
            else:
                normalized[c] = modality
        
        return normalized
