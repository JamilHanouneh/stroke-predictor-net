#!/usr/bin/env python3
"""
Preprocess ISLES 2022 dataset for training.
Converts NIfTI files to HDF5 format with preprocessing.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import h5py
from tqdm import tqdm
import json
import logging
from scipy.ndimage import zoom

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_parser import load_config

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_nifti(file_path):
    """Load NIfTI file and return numpy array"""
    nii = nib.load(file_path)
    data = nii.get_fdata()
    return data, nii.affine


def resample_volume(volume, current_spacing, target_spacing, target_shape):
    """
    Resample volume to target spacing and shape.
    
    Args:
        volume (np.ndarray): Input volume
        current_spacing (tuple): Current voxel spacing
        target_spacing (tuple): Target voxel spacing
        target_shape (tuple): Target shape
    
    Returns:
        np.ndarray: Resampled volume
    """
    # Calculate zoom factors
    zoom_factors = [
        (c * s) / (t * ts) 
        for c, s, t, ts in zip(volume.shape, current_spacing, target_shape, target_spacing)
    ]
    
    # Resample
    resampled = zoom(volume, zoom_factors, order=1)
    
    # Ensure exact target shape
    if resampled.shape != target_shape:
        # Crop or pad
        result = np.zeros(target_shape)
        
        slices = tuple(
            slice(0, min(r, t))
            for r, t in zip(resampled.shape, target_shape)
        )
        
        result[slices] = resampled[slices]
        return result
    
    return resampled

def preprocess_subject(subject_dir, derivatives_dir, modalities, config):
    """
    Preprocess single subject - handles case-insensitive modality matching.
    
    Args:
        subject_dir (Path): Subject directory with raw data
        derivatives_dir (Path): Derivatives directory with masks
        modalities (list): List of modalities to load
        config (dict): Configuration
    
    Returns:
        tuple: (imaging_array, mask_array) or (None, None) if failed
    """
    subject_id = subject_dir.name
    target_shape = tuple(config['data']['preprocessing']['target_shape'])
    target_spacing = tuple(config['data']['preprocessing']['target_spacing'])
    
    try:
        # Load modalities
        imaging_list = []
        
        for modality in modalities:
            # Determine search directory based on modality type
            if modality.upper() == 'FLAIR':
                search_dir = subject_dir / 'ses-0001' / 'anat'
            else:  # DWI, ADC, dwi, adc, etc.
                search_dir = subject_dir / 'ses-0001' / 'dwi'
            
            if not search_dir.exists():
                logger.warning(f"Directory not found: {search_dir}")
                return None, None
            
            # Try both uppercase and lowercase versions
            modality_file = None
            for variant in [modality, modality.upper(), modality.lower()]:
                pattern = f'*_{variant}.nii.gz'
                matches = list(search_dir.glob(pattern))
                if matches:
                    modality_file = matches[0]
                    break
            
            if not modality_file:
                logger.warning(f"Missing {modality} for {subject_id} in {search_dir}")
                return None, None
            
            # Load NIfTI
            data, affine = load_nifti(modality_file)
            
            # Get current spacing from affine
            current_spacing = np.abs(np.diag(affine)[:3])
            
            # Resample
            resampled = resample_volume(data, current_spacing, target_spacing, target_shape)
            
            # Intensity clipping
            if config['data']['preprocessing']['normalize']:
                mask = resampled > 0
                if mask.sum() > 0:
                    percentiles = config['data']['preprocessing']['intensity_clip_percentiles']
                    p_low, p_high = np.percentile(resampled[mask], percentiles)
                    resampled = np.clip(resampled, p_low, p_high)
            
            imaging_list.append(resampled)
        
        # Stack modalities
        imaging_array = np.stack(imaging_list, axis=0)  # [C, H, W, D]
        
        # Load mask
        mask_dir = derivatives_dir / subject_id / 'ses-0001'
        
        if not mask_dir.exists():
            logger.warning(f"Mask directory not found: {mask_dir}")
            return None, None
        
        mask_files = list(mask_dir.glob('*_msk.nii.gz'))
        
        if not mask_files:
            logger.warning(f"Missing mask for {subject_id}")
            return None, None
        
        mask_data, mask_affine = load_nifti(mask_files[0])
        mask_spacing = np.abs(np.diag(mask_affine)[:3])
        
        # Resample mask (nearest neighbor for binary)
        mask_resampled = resample_volume(mask_data, mask_spacing, target_spacing, target_shape)
        mask_array = (mask_resampled > 0.5).astype(np.float32)[np.newaxis, ...]
        
        return imaging_array, mask_array
        
    except Exception as e:
        logger.error(f"Failed to process {subject_id}: {e}")
        return None, None



def create_splits(subject_ids, config, output_dir):
    """
    Create train/val/test splits.
    
    Args:
        subject_ids (list): List of subject IDs
        config (dict): Configuration
        output_dir (Path): Output directory
    
    Returns:
        dict: Split dictionaries
    """
    np.random.seed(config['system']['seed'])
    np.random.shuffle(subject_ids)
    
    train_ratio = config['data']['splits']['train']
    val_ratio = config['data']['splits']['val']
    
    num_subjects = len(subject_ids)
    num_train = int(num_subjects * train_ratio)
    num_val = int(num_subjects * val_ratio)
    
    splits = {
        'train': subject_ids[:num_train],
        'val': subject_ids[num_train:num_train + num_val],
        'test': subject_ids[num_train + num_val:]
    }
    
    # Save splits
    splits_dir = output_dir.parent / 'splits'
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, ids in splits.items():
        split_file = splits_dir / f'{split_name}_subjects.txt'
        with open(split_file, 'w') as f:
            f.write('\n'.join(ids))
        logger.info(f"Saved {split_name} split: {len(ids)} subjects")
    
    return splits


def main():
    parser = argparse.ArgumentParser(description='Preprocess ISLES 2022 dataset')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config.yaml')
    parser.add_argument('--input', type=str, default=None,
                       help='ISLES dataset directory (overrides config)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (overrides config)')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Get paths
    if args.input:
        input_dir = Path(args.input)
    else:
        input_dir = Path(config['data']['paths']['raw_data'])
    
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(config['data']['paths']['processed_data'])
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("ISLES 2022 Data Preprocessing")
    logger.info("=" * 70)
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Target shape: {config['data']['preprocessing']['target_shape']}")
    logger.info(f"Target spacing: {config['data']['preprocessing']['target_spacing']} mm")
    logger.info(f"Modalities: {', '.join(config['data']['modalities'])}")
    logger.info("=" * 70)
    
    # Get subject directories
    # Get paths (fixed for your directory structure)
    rawdata_dir = input_dir / 'rawdata'
    derivatives_dir = input_dir / 'derivatives'

    
    if not rawdata_dir.exists():
        logger.error(f"Dataset not found at {rawdata_dir}")
        logger.info("Please run: python scripts/download_isles.py")
        return
    
    subject_dirs = sorted(rawdata_dir.glob('sub-*'))
    logger.info(f"Found {len(subject_dirs)} subjects")
    
    # Create HDF5 files
    imaging_h5 = h5py.File(output_dir / 'imaging_tensors.h5', 'w')
    masks_h5 = h5py.File(output_dir / 'lesion_masks.h5', 'w')
    
    # Process subjects
    processed_subjects = []
    failed_subjects = []
    
    logger.info("\nProcessing subjects...")
    for subject_dir in tqdm(subject_dirs, desc='Preprocessing'):
        subject_id = subject_dir.name
        
        imaging, mask = preprocess_subject(
            subject_dir,
            derivatives_dir,
            config['data']['modalities'],
            config
        )
        
        if imaging is not None and mask is not None:
            # Save to HDF5
            imaging_h5.create_dataset(subject_id, data=imaging, compression='gzip')
            masks_h5.create_dataset(subject_id, data=mask, compression='gzip')
            processed_subjects.append(subject_id)
        else:
            failed_subjects.append(subject_id)
    
    imaging_h5.close()
    masks_h5.close()
    
    logger.info(f"\n✓ Successfully processed: {len(processed_subjects)} subjects")
    if failed_subjects:
        logger.warning(f"⚠ Failed: {len(failed_subjects)} subjects")
        logger.warning(f"Failed subjects: {', '.join(failed_subjects[:10])}")
    
    # Create splits
    logger.info("\nCreating train/val/test splits...")
    splits = create_splits(processed_subjects, config, output_dir)
    
    # Save metadata
    metadata = {
        'num_subjects': len(processed_subjects),
        'modalities': config['data']['modalities'],
        'target_shape': config['data']['preprocessing']['target_shape'],
        'target_spacing': config['data']['preprocessing']['target_spacing'],
        'splits': {k: len(v) for k, v in splits.items()},
        'failed_subjects': failed_subjects
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("=" * 70)
    logger.info("✓ Preprocessing complete!")
    logger.info("=" * 70)
    logger.info("\nNext step:")
    logger.info("  python scripts/train.py --config config/config.yaml --device cpu")


if __name__ == '__main__':
    main()
