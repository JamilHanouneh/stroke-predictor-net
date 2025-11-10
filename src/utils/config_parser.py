"""
Configuration parsing utilities.
Loads and validates YAML configuration files.
"""

import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str or Path): Path to config.yaml
    
    Returns:
        dict: Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate configuration
    validate_config(config)
    
    return config


def validate_config(config):
    """
    Validate configuration dictionary.
    
    Args:
        config (dict): Configuration to validate
    
    Raises:
        ValueError: If configuration is invalid
    """
    required_sections = ['system', 'data', 'model', 'training', 'paths']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate device
    valid_devices = ['auto', 'cpu', 'cuda']
    if config['system']['device'] not in valid_devices:
        raise ValueError(f"Invalid device: {config['system']['device']}. "
                        f"Must be one of {valid_devices}")
    
    # Validate batch sizes
    if config['training']['batch_size']['cpu'] < 1:
        raise ValueError("CPU batch size must be >= 1")
    if config['training']['batch_size']['gpu'] < 1:
        raise ValueError("GPU batch size must be >= 1")
    
    # Validate learning rate
    if config['training']['learning_rate'] <= 0:
        raise ValueError("Learning rate must be > 0")
    
    logger.info("✓ Configuration validated successfully")


def save_config(config, save_path):
    """
    Save configuration to YAML file.
    
    Args:
        config (dict): Configuration to save
        save_path (str or Path): Output path
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    logger.info(f"✓ Configuration saved to {save_path}")
