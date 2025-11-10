"""
Device management for CPU/GPU compatibility.
Automatically detects available device and adjusts settings.
"""

import torch
import logging

logger = logging.getLogger(__name__)


class DeviceManager:
    """
    Manages device selection and configuration for CPU/GPU training.
    
    Features:
    - Automatic CUDA detection
    - Memory-aware batch size adjustment
    - Worker count optimization
    - Mixed precision settings
    """
    
    def __init__(self, prefer_gpu=True, gpu_id=0):
        """
        Initialize device manager.
        
        Args:
            prefer_gpu (bool): Prefer GPU if available
            gpu_id (int): GPU device ID if multiple GPUs
        """
        self.prefer_gpu = prefer_gpu
        self.gpu_id = gpu_id
        self.device = self._detect_device()
        self.use_amp = self._should_use_amp()
        
        self._log_device_info()
    
    def _detect_device(self):
        """Detect and return appropriate device"""
        if self.prefer_gpu and torch.cuda.is_available():
            device = torch.device(f'cuda:{self.gpu_id}')
            logger.info(f"✓ Using GPU: {torch.cuda.get_device_name(self.gpu_id)}")
        else:
            device = torch.device('cpu')
            if self.prefer_gpu and not torch.cuda.is_available():
                logger.warning("⚠ CUDA not available, falling back to CPU")
            else:
                logger.info("✓ Using CPU (as configured)")
        
        return device
    
    def _should_use_amp(self):
        """Check if automatic mixed precision should be used"""
        # AMP only beneficial on GPU with Tensor Cores
        if self.device.type == 'cuda':
            # Check GPU compute capability (Tensor Cores require >=7.0)
            capability = torch.cuda.get_device_capability(self.gpu_id)
            has_tensor_cores = capability[0] >= 7
            
            if has_tensor_cores:
                logger.info("✓ Mixed precision training enabled (Tensor Cores available)")
                return True
            else:
                logger.info("ℹ Mixed precision disabled (Tensor Cores not available)")
                return False
        
        return False
    
    def _log_device_info(self):
        """Log detailed device information"""
        if self.device.type == 'cuda':
            props = torch.cuda.get_device_properties(self.gpu_id)
            logger.info(f"  GPU Memory: {props.total_memory / 1e9:.2f} GB")
            logger.info(f"  CUDA Version: {torch.version.cuda}")
            logger.info(f"  Compute Capability: {props.major}.{props.minor}")
        else:
            import psutil
            logger.info(f"  CPU Cores: {psutil.cpu_count(logical=False)} physical, "
                       f"{psutil.cpu_count(logical=True)} logical")
            logger.info(f"  RAM: {psutil.virtual_memory().total / 1e9:.2f} GB")
    
    def get_batch_size(self, config):
        """
        Get appropriate batch size based on device.
        
        Args:
            config (dict): Configuration with batch_size settings
        
        Returns:
            int: Batch size
        """
        if self.device.type == 'cuda':
            batch_size = config['training']['batch_size']['gpu']
        else:
            batch_size = config['training']['batch_size']['cpu']
        
        logger.info(f"Batch size: {batch_size} ({self.device.type.upper()})")
        return batch_size
    
    def get_num_workers(self, config):
        """
        Get optimal number of DataLoader workers.
        
        Args:
            config (dict): Configuration
        
        Returns:
            int: Number of workers
        """
        if config['system']['num_workers'] == 'auto':
            # CPU: More workers for data loading
            # GPU: Fewer workers to not bottleneck GPU
            if self.device.type == 'cpu':
                import os
                num_workers = min(os.cpu_count() // 2, 8)
            else:
                num_workers = 4
        else:
            num_workers = config['system']['num_workers']
        
        logger.info(f"DataLoader workers: {num_workers}")
        return num_workers
    
    def get_gradient_accumulation_steps(self, config):
        """
        Get gradient accumulation steps.
        CPU uses more accumulation to simulate larger batches.
        
        Args:
            config (dict): Configuration
        
        Returns:
            int: Accumulation steps
        """
        if self.device.type == 'cuda':
            steps = config['training']['gradient_accumulation_steps']['gpu']
        else:
            steps = config['training']['gradient_accumulation_steps']['cpu']
        
        if steps > 1:
            logger.info(f"Gradient accumulation steps: {steps}")
        
        return steps
    
    def to_device(self, *tensors):
        """
        Move tensors to device.
        
        Args:
            *tensors: Variable number of tensors or models
        
        Returns:
            Tuple of tensors on device (or single tensor if only one input)
        """
        moved = [t.to(self.device) if t is not None else None for t in tensors]
        return moved[0] if len(moved) == 1 else tuple(moved)
    
    def empty_cache(self):
        """Clear GPU cache if using CUDA"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
    
    def get_memory_usage(self):
        """
        Get current memory usage.
        
        Returns:
            dict: Memory statistics
        """
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.gpu_id) / 1e9
            cached = torch.cuda.memory_reserved(self.gpu_id) / 1e9
            total = torch.cuda.get_device_properties(self.gpu_id).total_memory / 1e9
            
            return {
                'allocated_gb': allocated,
                'cached_gb': cached,
                'total_gb': total,
                'utilization_percent': (allocated / total) * 100
            }
        else:
            import psutil
            memory = psutil.virtual_memory()
            return {
                'used_gb': memory.used / 1e9,
                'total_gb': memory.total / 1e9,
                'utilization_percent': memory.percent
            }


# Global device manager instance
_device_manager = None

def get_device_manager(config=None):
    """
    Get or create global device manager instance.
    
    Args:
        config (dict, optional): Configuration dict
    
    Returns:
        DeviceManager: Global device manager
    """
    global _device_manager
    
    if _device_manager is None:
        if config is None:
            raise ValueError("Must provide config for first initialization")
        
        prefer_gpu = config['system']['device'] in ['auto', 'cuda']
        _device_manager = DeviceManager(prefer_gpu=prefer_gpu)
    
    return _device_manager
