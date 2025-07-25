"""
GPU Memory Optimizer for Document Search System
Implements model caching, memory management, and batch processing optimizations
"""
import torch
import gc
import logging
from typing import Dict, Any, Optional
from functools import lru_cache
import threading
import time

logger = logging.getLogger(__name__)

class GPUOptimizer:
    """Manages GPU memory and model loading for optimal performance"""
    
    def __init__(self):
        self.model_cache = {}
        self.cache_lock = threading.Lock()
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5 minutes
        
    def get_gpu_memory_info(self):
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            memory_free = torch.cuda.get_device_properties(0).total_memory / 1024**3 - memory_reserved
            
            return {
                "allocated": memory_allocated,
                "reserved": memory_reserved,
                "free": memory_free,
                "total": torch.cuda.get_device_properties(0).total_memory / 1024**3
            }
        return None
    
    def cleanup_gpu_memory(self, force=False):
        """Clean up GPU memory"""
        current_time = time.time()
        
        if force or (current_time - self.last_cleanup) > self.cleanup_interval:
            logger.info("Performing GPU memory cleanup...")
            
            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Force garbage collection
            gc.collect()
            
            self.last_cleanup = current_time
            
            # Log memory status
            memory_info = self.get_gpu_memory_info()
            if memory_info:
                logger.info(f"GPU Memory after cleanup - Allocated: {memory_info['allocated']:.2f}GB, "
                           f"Reserved: {memory_info['reserved']:.2f}GB, Free: {memory_info['free']:.2f}GB")
    
    def get_or_create_model(self, model_type: str, model_path: str, create_func, *args, **kwargs):
        """Get model from cache or create new one with memory optimization"""
        cache_key = f"{model_type}_{model_path}"
        
        with self.cache_lock:
            if cache_key in self.model_cache:
                logger.info(f"Using cached {model_type} model")
                return self.model_cache[cache_key]
            
            # Check memory before loading
            memory_info = self.get_gpu_memory_info()
            if memory_info and memory_info['free'] < 2.0:  # Less than 2GB free
                logger.warning("Low GPU memory, performing cleanup before model loading")
                self.cleanup_gpu_memory(force=True)
            
            # Create new model
            logger.info(f"Creating new {model_type} model")
            model = create_func(*args, **kwargs)
            
            # Cache the model
            self.model_cache[cache_key] = model
            
            logger.info(f"Cached {model_type} model successfully")
            return model
    
    def optimize_batch_size(self, base_batch_size: int = 32) -> int:
        """Dynamically adjust batch size based on available GPU memory"""
        memory_info = self.get_gpu_memory_info()
        
        if not memory_info:
            return base_batch_size
        
        free_memory_gb = memory_info['free']
        
        # Adjust batch size based on available memory
        if free_memory_gb > 10:
            return min(base_batch_size * 2, 128)
        elif free_memory_gb > 5:
            return base_batch_size
        elif free_memory_gb > 2:
            return max(base_batch_size // 2, 8)
        else:
            return max(base_batch_size // 4, 4)
    
    def enable_memory_efficient_attention(self):
        """Enable memory efficient attention mechanisms"""
        if torch.cuda.is_available():
            try:
                # Enable memory efficient attention
                torch.backends.cuda.enable_math_sdp(False)
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                logger.info("Enabled memory efficient attention mechanisms")
            except Exception as e:
                logger.warning(f"Could not enable memory efficient attention: {e}")
    
    def setup_mixed_precision(self):
        """Setup mixed precision training for memory efficiency"""
        if torch.cuda.is_available():
            try:
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("Enabled mixed precision optimizations")
            except Exception as e:
                logger.warning(f"Could not enable mixed precision: {e}")
    
    def monitor_memory_usage(self, threshold_gb: float = 20.0):
        """Monitor memory usage and trigger cleanup if needed"""
        memory_info = self.get_gpu_memory_info()
        
        if memory_info and memory_info['allocated'] > threshold_gb:
            logger.warning(f"High GPU memory usage: {memory_info['allocated']:.2f}GB allocated")
            self.cleanup_gpu_memory(force=True)
            return True
        return False

# Global optimizer instance
gpu_optimizer = GPUOptimizer()