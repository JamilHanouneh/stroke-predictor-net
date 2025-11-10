#!/usr/bin/env python3
"""
Download ISLES 2022 dataset from Zenodo.
"""

import argparse
import requests
from pathlib import Path
from tqdm import tqdm
import zipfile
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def download_file(url, output_path):
    """
    Download file with progress bar.
    
    Args:
        url (str): Download URL
        output_path (Path): Output file path
    """
    logger.info(f"Downloading from {url}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading') as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    logger.info(f"✓ Downloaded to {output_path}")


def extract_zip(zip_path, extract_to):
    """
    Extract zip file.
    
    Args:
        zip_path (Path): Path to zip file
        extract_to (Path): Extraction directory
    """
    logger.info(f"Extracting {zip_path.name}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    logger.info(f"✓ Extracted to {extract_to}")


def main():
    parser = argparse.ArgumentParser(description='Download ISLES 2022 dataset')
    parser.add_argument('--output', type=str, default='data/raw/ISLES2022',
                       help='Output directory')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip download if file exists')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ISLES 2022 Zenodo URL
    ZENODO_URL = "https://zenodo.org/records/7153326/files/ISLES-2022.zip"
    zip_path = output_dir / "ISLES-2022.zip"
    
    logger.info("=" * 70)
    logger.info("ISLES 2022 Dataset Download")
    logger.info("=" * 70)
    logger.info(f"Dataset: Ischemic Stroke Lesion Segmentation Challenge 2022")
    logger.info(f"Source: Zenodo (https://zenodo.org/records/7153326)")
    logger.info(f"Size: ~15-20 GB")
    logger.info(f"License: CC BY 4.0")
    logger.info("=" * 70)
    
    # Check if already downloaded
    if zip_path.exists() and args.skip_download:
        logger.info(f"Dataset already downloaded: {zip_path}")
    else:
        # Download
        try:
            download_file(ZENODO_URL, zip_path)
        except Exception as e:
            logger.error(f"Download failed: {e}")
            logger.info("\nManual download instructions:")
            logger.info(f"1. Visit: https://zenodo.org/records/7153326")
            logger.info(f"2. Download ISLES-2022.zip (~15-20 GB)")
            logger.info(f"3. Place the file at: {zip_path}")
            logger.info(f"4. Run this script again with --skip-download")
            return
    
    # Extract
    logger.info("\nExtracting dataset...")
    try:
        extract_zip(zip_path, output_dir)
        logger.info("✓ Extraction complete")
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return
    
    # Verify structure
    logger.info("\nVerifying dataset structure...")
    expected_subdirs = ['rawdata', 'derivatives']
    
    all_found = True
    for subdir in expected_subdirs:
        subdir_path = output_dir / 'ISLES-2022' / subdir
        if subdir_path.exists():
            num_subjects = len(list(subdir_path.glob('sub-*')))
            logger.info(f"✓ Found {subdir}/ with {num_subjects} subjects")
        else:
            logger.warning(f"⚠ Missing {subdir}/")
            all_found = False
    
    if all_found:
        logger.info("\n" + "=" * 70)
        logger.info("✓ Dataset downloaded and verified successfully!")
        logger.info("=" * 70)
        logger.info("\nNext steps:")
        logger.info("1. Create synthetic clinical data:")
        logger.info("   python scripts/create_synthetic_clinical.py")
        logger.info("\n2. Preprocess data:")
        logger.info("   python scripts/preprocess_data.py --config config/config.yaml")
    else:
        logger.error("\n✗ Dataset verification failed")


if __name__ == '__main__':
    main()
