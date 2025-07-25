"""
Singleton Service Manager to prevent multiple model instances
"""
import logging
from typing import Optional, Dict, Any
import threading
from gpu_optimizer import gpu_optimizer

logger = logging.getLogger(__name__)

class ServiceManager:
    """Singleton manager for all services to prevent multiple instances"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ServiceManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.services = {}
            self.config = None
            self._initialized = True
            logger.info("ServiceManager initialized")
    
    def set_config(self, config):
        """Set the configuration for all services"""
        self.config = config
    
    def get_service(self, service_name: str, service_class, *args, **kwargs):
        """Get or create a service instance with GPU memory optimization"""
        if service_name not in self.services:
            logger.info(f"Creating new {service_name} service")
            
            # Check GPU memory before creating memory-intensive services
            gpu_memory_before = None
            if service_name in ['embedding_service', 'llm_service', 'document_analyzer']:
                gpu_memory_before = gpu_optimizer.get_gpu_memory_info()
                
                # Cleanup if memory is low
                if gpu_memory_before and gpu_memory_before['free'] < 3.0:  # Less than 3GB free
                    logger.warning(f"Low GPU memory ({gpu_memory_before['free']:.1f}GB), cleaning up before creating {service_name}")
                    gpu_optimizer.cleanup_gpu_memory(force=True)
                
                # Enable optimizations for GPU services
                gpu_optimizer.enable_memory_efficient_attention()
                gpu_optimizer.setup_mixed_precision()
            
            # Create the service
            self.services[service_name] = service_class(*args, **kwargs)
            
            # Log memory usage after creation
            if gpu_memory_before:
                gpu_memory_after = gpu_optimizer.get_gpu_memory_info()
                if gpu_memory_after:
                    memory_used = gpu_memory_after['allocated'] - gpu_memory_before['allocated']
                    logger.info(f"{service_name} created, using {memory_used:.2f}GB GPU memory")
            
            logger.info(f"{service_name} service created successfully")
        else:
            logger.info(f"Using cached {service_name} service")
        
        return self.services[service_name]
    
    def clear_service(self, service_name: str):
        """Clear a specific service"""
        if service_name in self.services:
            del self.services[service_name]
            gpu_optimizer.cleanup_gpu_memory(force=True)
            logger.info(f"{service_name} service cleared")
    
    def clear_all_services(self):
        """Clear all services"""
        self.services.clear()
        gpu_optimizer.cleanup_gpu_memory(force=True)
        logger.info("All services cleared")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        stats = {
            "active_services": list(self.services.keys()),
            "service_count": len(self.services)
        }
        
        gpu_stats = gpu_optimizer.get_gpu_memory_info()
        if gpu_stats:
            stats.update(gpu_stats)
        
        return stats

# Global service manager instance
service_manager = ServiceManager()