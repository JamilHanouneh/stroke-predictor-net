#!/usr/bin/env python3
"""
Generate synthetic clinical metadata for ISLES 2022 dataset.
Since ISLES 2022 has limited clinical data, we generate realistic synthetic features.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def generate_clinical_features(subject_ids, seed=42):
    """
    Generate realistic synthetic clinical features.
    
    Args:
        subject_ids (list): List of subject IDs
        seed (int): Random seed
    
    Returns:
        pd.DataFrame: Clinical features dataframe
    """
    np.random.seed(seed)
    
    num_subjects = len(subject_ids)
    
    clinical_data = {
        'subject_id': subject_ids,
        
        # Demographics
        'age': np.random.normal(65, 15, num_subjects).clip(18, 95).astype(int),
        'sex': np.random.choice([0, 1], num_subjects),  # 0=Female, 1=Male
        
        # Stroke severity (NIHSS: 0-42, higher = more severe)
        'nihss_score': np.random.gamma(2, 5, num_subjects).clip(0, 42).astype(int),
        
        # Time to scan (minutes since symptom onset)
        'time_to_scan': np.random.exponential(180, num_subjects).clip(30, 720).astype(int),
        
        # Vital signs
        'systolic_bp': np.random.normal(145, 20, num_subjects).clip(90, 220).astype(int),
        'diastolic_bp': np.random.normal(85, 15, num_subjects).clip(60, 130).astype(int),
        
        # Lab values
        'glucose': np.random.normal(120, 40, num_subjects).clip(60, 300).astype(int),
        
        # Medical history (binary)
        'prior_stroke': np.random.binomial(1, 0.2, num_subjects),
        'diabetes': np.random.binomial(1, 0.3, num_subjects),
        'hypertension': np.random.binomial(1, 0.5, num_subjects),
    }
    
    df = pd.DataFrame(clinical_data)
    
    # Add correlation between features
    # Higher NIHSS correlates with worse outcomes
    df.loc[df['nihss_score'] > 20, 'prior_stroke'] = np.random.binomial(1, 0.4, sum(df['nihss_score'] > 20))
    
    # Older patients more likely to have comorbidities
    df.loc[df['age'] > 70, 'hypertension'] = np.random.binomial(1, 0.7, sum(df['age'] > 70))
    df.loc[df['age'] > 70, 'diabetes'] = np.random.binomial(1, 0.4, sum(df['age'] > 70))
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic clinical metadata')
    parser.add_argument('--input', type=str, default='data/raw/ISLES2022',
                       help='ISLES 2022 dataset directory')
    parser.add_argument('--output', type=str, default='data/synthetic/synthetic_clinical.csv',
                       help='Output CSV file')
    parser.add_argument('--num-subjects', type=int, default=250,
                       help='Number of subjects (default: 250 for ISLES 2022)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("Synthetic Clinical Metadata Generator")
    logger.info("=" * 70)
    
    # Get subject IDs from dataset
    input_dir = Path(args.input)
    rawdata_dir = input_dir / 'ISLES-2022' / 'rawdata'
    
    if rawdata_dir.exists():
        subject_dirs = sorted(rawdata_dir.glob('sub-*'))
        subject_ids = [d.name for d in subject_dirs]
        logger.info(f"Found {len(subject_ids)} subjects in dataset")
    else:
        # Generate generic subject IDs
        logger.warning(f"Dataset not found at {rawdata_dir}")
        logger.info(f"Generating {args.num_subjects} generic subject IDs")
        subject_ids = [f'sub-{i:03d}' for i in range(1, args.num_subjects + 1)]
    
    # Generate clinical features
    logger.info("Generating synthetic clinical features...")
    clinical_df = generate_clinical_features(subject_ids, seed=args.seed)
    
    # Display summary statistics
    logger.info("\nSynthetic Clinical Data Summary:")
    logger.info("-" * 70)
    logger.info(f"Number of subjects: {len(clinical_df)}")
    logger.info(f"\nAge: {clinical_df['age'].mean():.1f} ± {clinical_df['age'].std():.1f} years")
    logger.info(f"Sex (Male): {clinical_df['sex'].mean():.1%}")
    logger.info(f"\nNIHSS Score: {clinical_df['nihss_score'].mean():.1f} ± {clinical_df['nihss_score'].std():.1f}")
    logger.info(f"Time to Scan: {clinical_df['time_to_scan'].mean():.0f} ± {clinical_df['time_to_scan'].std():.0f} min")
    logger.info(f"\nComorbidities:")
    logger.info(f"  Prior Stroke: {clinical_df['prior_stroke'].mean():.1%}")
    logger.info(f"  Diabetes: {clinical_df['diabetes'].mean():.1%}")
    logger.info(f"  Hypertension: {clinical_df['hypertension'].mean():.1%}")
    
    # Save to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clinical_df.to_csv(output_path, index=False)
    
    logger.info("-" * 70)
    logger.info(f"✓ Saved to {output_path}")
    logger.info("=" * 70)
    
    # Display first few rows
    logger.info("\nSample data (first 5 subjects):")
    print(clinical_df.head().to_string())


if __name__ == '__main__':
    main()
