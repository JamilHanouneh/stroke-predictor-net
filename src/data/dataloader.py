"""
DataLoader factory with CPU/GPU optimization.
"""

from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)


def create_dataloader(dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True, **kwargs):
    """
    Create DataLoader with appropriate settings.
    
    Args:
        dataset: PyTorch Dataset
        batch_size (int): Batch size
        shuffle (bool): Shuffle data
        num_workers (int): Number of worker processes
        pin_memory (bool): Pin memory for GPU
        **kwargs: Additional DataLoader arguments
    
    Returns:
        DataLoader: Configured DataLoader
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        **kwargs
    )
    
    logger.info(f"Created DataLoader: batch_size={batch_size}, num_workers={num_workers}")
    
    return dataloader


def create_train_val_loaders(train_dataset, val_dataset, config, device_mgr):
    """
    Create train and validation DataLoaders.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config (dict): Configuration
        device_mgr: Device manager
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    batch_size = device_mgr.get_batch_size(config)
    num_workers = device_mgr.get_num_workers(config)
    pin_memory = device_mgr.device.type == 'cuda'
    
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader
