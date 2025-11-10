"""
Miscellaneous helper functions.
"""

import os
import json
import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def ensure_dir(directory):
    """
    Create directory if it doesn't exist.
    
    Args:
        directory (str or Path): Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_json(data, filepath):
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        filepath (str or Path): Output path
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"✓ Saved JSON to {filepath}")


def load_json(filepath):
    """
    Load data from JSON file.
    
    Args:
        filepath (str or Path): JSON file path
    
    Returns:
        Data from JSON
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def count_parameters(model):
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
    
    Returns:
        tuple: (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_time(seconds):
    """
    Format seconds to readable time string.
    
    Args:
        seconds (float): Time in seconds
    
    Returns:
        str: Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"
