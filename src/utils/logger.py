"""
Logging configuration for StrokePredictorNet.
Based on your proven logging style from BrainGraphNet.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(config):
    """
    Setup comprehensive logging system.
    
    Args:
        config (dict): Configuration dictionary
    
    Returns:
        logging.Logger: Configured logger
    """
    # Create logs directory
    log_dir = Path(config['logging']['log_file']).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('StrokePredictorNet')
    logger.setLevel(getattr(logging, config['logging']['level']))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = ColoredFormatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (detailed logs)
    file_handler = logging.FileHandler(config['logging']['log_file'])
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Log startup info
    logger.info("=" * 70)
    logger.info("StrokePredictorNet - Multimodal Stroke Outcome Prediction")
    logger.info("=" * 70)
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log level: {config['logging']['level']}")
    logger.info(f"Log file: {config['logging']['log_file']}")
    
    return logger


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for console output"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        """Format log record with colors"""
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        
        return super().format(record)


def log_config(logger, config):
    """
    Log configuration details.
    
    Args:
        logger: Logger instance
        config (dict): Configuration dictionary
    """
    logger.info("Configuration:")
    logger.info("-" * 70)
    
    # System
    logger.info(f"Device: {config['system']['device']}")
    logger.info(f"Mixed Precision: {config['system']['mixed_precision']}")
    logger.info(f"Random Seed: {config['system']['seed']}")
    
    # Data
    logger.info(f"Dataset: {config['data']['dataset']}")
    logger.info(f"Modalities: {', '.join(config['data']['modalities'])}")
    logger.info(f"Target Shape: {config['data']['preprocessing']['target_shape']}")
    
    # Model
    logger.info(f"Architecture: {config['model']['architecture']}")
    logger.info(f"Imaging Encoder: {config['model']['imaging_encoder']['backbone']}")
    logger.info(f"Fusion Method: {config['model']['fusion']['method']}")
    
    # Training
    logger.info(f"Epochs: {config['training']['num_epochs']}")
    logger.info(f"Learning Rate: {config['training']['learning_rate']}")
    logger.info(f"Optimizer: {config['training']['optimizer']['type']}")
    
    logger.info("-" * 70)
