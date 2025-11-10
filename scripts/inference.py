#!/usr/bin/env python3
"""
Run inference on new data.
Load trained model and predict on unseen subjects.
"""

import sys
import argparse
from pathlib import Path
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_parser import load_config
from src.utils.logger import setup_logger
from src.utils.device_manager import get_device_manager
from src.models.multimodal_fusion import MultimodalStrokeNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_subject_data(subject_dir, modalities):
    """
    Load MRI data for a single subject.
    
    Args:
        subject_dir (Path): Subject directory
        modalities (list): List of modalities to load
    
    Returns:
        np.ndarray: Imaging array [C, H, W, D]
    """
    imaging_list = []
    
    for modality in modalities:
        # Find modality file
        modality_files = list(subject_dir.glob(f'*{modality}.nii.gz'))
        
        if not modality_files:
            raise FileNotFoundError(f"Missing {modality} for {subject_dir.name}")
        
        # Load NIfTI
        nii = nib.load(modality_files[0])
        data = nii.get_fdata()
        
        imaging_list.append(data)
    
    # Stack modalities
    imaging = np.stack(imaging_list, axis=0)
    
    return imaging


def preprocess_imaging(imaging, config):
    """
    Preprocess imaging data.
    
    Args:
        imaging (np.ndarray): Raw imaging [C, H, W, D]
        config (dict): Configuration
    
    Returns:
        torch.Tensor: Preprocessed imaging
    """
    # Normalize
    normalized = np.zeros_like(imaging)
    
    for c in range(imaging.shape[0]):
        modality = imaging[c]
        mask = modality > 0
        
        if mask.sum() > 0:
            mean = modality[mask].mean()
            std = modality[mask].std()
            
            if std > 0:
                normalized[c] = np.where(mask, (modality - mean) / std, 0)
            else:
                normalized[c] = modality
    
    # Convert to tensor
    tensor = torch.from_numpy(normalized).float()
    
    return tensor


def main():
    parser = argparse.ArgumentParser(description='Run inference')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Input subject directory or file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--clinical', type=str, default=None,
                       help='Clinical features (comma-separated)')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Device manager
    device_mgr = get_device_manager(config)
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = MultimodalStrokeNet(config)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = device_mgr.to_device(model)
    model.eval()
    
    logger.info("✓ Model loaded successfully")
    
    # Load input data
    input_path = Path(args.input)
    
    if input_path.is_dir():
        # Subject directory
        logger.info(f"Loading data from {input_path}")
        imaging = load_subject_data(input_path, config['data']['modalities'])
    else:
        # Single NIfTI file
        logger.info(f"Loading {input_path}")
        nii = nib.load(input_path)
        imaging = nii.get_fdata()
        imaging = imaging[np.newaxis, ...]  # Add channel dim
    
    # Preprocess
    logger.info("Preprocessing...")
    imaging_tensor = preprocess_imaging(imaging, config)
    imaging_tensor = imaging_tensor.unsqueeze(0)  # Add batch dim
    imaging_tensor = device_mgr.to_device(imaging_tensor)
    
    # Clinical features
    if args.clinical:
        clinical_values = [float(x) for x in args.clinical.split(',')]
        clinical_tensor = torch.tensor(clinical_values).float().unsqueeze(0)
        clinical_tensor = device_mgr.to_device(clinical_tensor)
    else:
        # Use dummy clinical features
        logger.warning("No clinical features provided, using zeros")
        clinical_tensor = torch.zeros(1, config['model']['clinical_encoder']['input_dim'])
        clinical_tensor = device_mgr.to_device(clinical_tensor)
    
    # Run inference
    logger.info("Running inference...")
    with torch.no_grad():
        outputs = model(imaging_tensor, clinical_tensor)
        segmentation = outputs['segmentation']
        probs = torch.sigmoid(segmentation)
    
    # Convert to numpy
    pred_mask = probs.cpu().numpy()[0, 0]
    
    # Save prediction
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'prediction.nii.gz'
    nii_output = nib.Nifti1Image(pred_mask, affine=np.eye(4))
    nib.save(nii_output, output_file)
    
    logger.info(f"✓ Prediction saved to {output_file}")
    
    # Summary statistics
    lesion_volume = (pred_mask > 0.5).sum()
    logger.info(f"Predicted lesion volume: {lesion_volume} voxels")


if __name__ == '__main__':
    main()
