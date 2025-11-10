#!/usr/bin/env python3
"""
Evaluation script for StrokePredictorNet.
Evaluates model on test set and saves predictions.
"""

import sys
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_parser import load_config
from src.utils.logger import setup_logger
from src.utils.device_manager import get_device_manager
from src.data.dataset import ISLESStrokeDataset
from src.models.multimodal_fusion import MultimodalStrokeNet
from src.models.metrics import SegmentationMetrics


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate StrokePredictorNet')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='outputs/evaluation')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    logger = setup_logger(config)
    
    # Device manager
    device_mgr = get_device_manager(config)
    
    # Load test subjects
    with open(Path(config['data']['paths']['splits']) / 'test_subjects.txt', 'r') as f:
        test_ids = [line.strip() for line in f]
    
    logger.info(f"Evaluating on {len(test_ids)} test subjects")
    
    # Create test dataset
    processed_dir = Path(config['data']['paths']['processed_data'])
    test_dataset = ISLESStrokeDataset(
        imaging_h5_path=processed_dir / 'imaging_tensors.h5',
        clinical_csv_path=config['data']['paths']['clinical_metadata'],
        masks_h5_path=processed_dir / 'lesion_masks.h5',
        subject_ids=test_ids,
        normalize=True
    )
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Load model
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model = MultimodalStrokeNet(config)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = device_mgr.to_device(model)
    model.eval()
    
    # Evaluate
    metrics_calculator = SegmentationMetrics()
    all_results = []
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting evaluation...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            imaging = device_mgr.to_device(batch['imaging'])
            clinical = device_mgr.to_device(batch['clinical'])
            masks = device_mgr.to_device(batch['mask'])
            subject_id = batch['subject_id'][0]
            
            # Forward pass
            outputs = model(imaging, clinical)
            segmentation = outputs['segmentation']
            probs = torch.sigmoid(segmentation)
            
            # Compute metrics
            metrics = metrics_calculator.compute_all_metrics(probs, masks)
            metrics['subject_id'] = subject_id
            all_results.append(metrics)
            
            # Save prediction
            pred_np = probs.cpu().numpy()[0, 0]
            np.save(output_dir / f'{subject_id}_prediction.npy', pred_np)
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / 'test_results.csv', index=False)
    
    # Log summary
    logger.info("=" * 70)
    logger.info("Test Results Summary")
    logger.info("=" * 70)
    for metric in ['dice', 'iou', 'sensitivity', 'specificity']:
        mean_val = results_df[metric].mean()
        std_val = results_df[metric].std()
        logger.info(f"{metric.capitalize()}: {mean_val:.4f} ± {std_val:.4f}")
    logger.info("=" * 70)
    
    logger.info(f"Results saved to {output_dir}")


if __name__ == '__main__':
    main()
