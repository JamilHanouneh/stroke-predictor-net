"""
ISLES 2022 dataset loader.
Handles NIfTI file loading and organization.
"""

import nibabel as nib
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ISLESLoader:
    """
    Loader for ISLES 2022 dataset.
    
    Handles:
    - Subject discovery
    - NIfTI file loading
    - Metadata extraction
    """
    
    def __init__(self, dataset_root):
        """
        Initialize loader.
        
        Args:
            dataset_root (str or Path): Root directory of ISLES dataset
        """
        self.dataset_root = Path(dataset_root)
        self.rawdata_dir = self.dataset_root / 'ISLES-2022' / 'rawdata'
        self.derivatives_dir = self.dataset_root / 'ISLES-2022' / 'derivatives'
        
        if not self.rawdata_dir.exists():
            raise ValueError(f"Dataset not found at {self.rawdata_dir}")
        
        # Discover subjects
        self.subject_dirs = sorted(self.rawdata_dir.glob('sub-*'))
        logger.info(f"Found {len(self.subject_dirs)} subjects in ISLES dataset")
    
    def get_subject_ids(self):
        """
        Get list of subject IDs.
        
        Returns:
            list: Subject IDs
        """
        return [d.name for d in self.subject_dirs]
    
    def load_modality(self, subject_id, modality):
        """
        Load single modality for a subject.
        
        Args:
            subject_id (str): Subject ID (e.g., 'sub-001')
            modality (str): Modality name (e.g., 'FLAIR', 'DWI', 'ADC')
        
        Returns:
            tuple: (data, affine) numpy array and affine matrix
        """
        subject_dir = self.rawdata_dir / subject_id / 'anat'
        
        # Find modality file
        modality_files = list(subject_dir.glob(f'*_{modality}.nii.gz'))
        
        if not modality_files:
            raise FileNotFoundError(f"Modality {modality} not found for {subject_id}")
        
        # Load NIfTI
        nii = nib.load(modality_files[0])
        data = nii.get_fdata()
        affine = nii.affine
        
        return data, affine
    
    def load_mask(self, subject_id):
        """
        Load lesion segmentation mask.
        
        Args:
            subject_id (str): Subject ID
        
        Returns:
            tuple: (mask, affine)
        """
        mask_dir = self.derivatives_dir / subject_id
        
        # Find mask file
        mask_files = list(mask_dir.glob('*_lesion-mask.nii.gz'))
        
        if not mask_files:
            raise FileNotFoundError(f"Mask not found for {subject_id}")
        
        # Load NIfTI
        nii = nib.load(mask_files[0])
        mask = nii.get_fdata()
        affine = nii.affine
        
        # Binarize
        mask = (mask > 0.5).astype(np.float32)
        
        return mask, affine
    
    def load_subject(self, subject_id, modalities):
        """
        Load all modalities and mask for a subject.
        
        Args:
            subject_id (str): Subject ID
            modalities (list): List of modality names
        
        Returns:
            dict: Dictionary with imaging and mask
        """
        # Load modalities
        imaging_list = []
        affines = []
        
        for modality in modalities:
            data, affine = self.load_modality(subject_id, modality)
            imaging_list.append(data)
            affines.append(affine)
        
        # Stack modalities
        imaging = np.stack(imaging_list, axis=0)  # [C, H, W, D]
        
        # Load mask
        mask, mask_affine = self.load_mask(subject_id)
        mask = mask[np.newaxis, ...]  # [1, H, W, D]
        
        return {
            'imaging': imaging,
            'mask': mask,
            'affine': affines[0],  # Use first modality's affine
            'subject_id': subject_id
        }
