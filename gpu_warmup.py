"""
GPU Warmup and Model Preloader for Document Search System
Ensures models are loaded and warmed up efficiently on startup
"""
import logging
import torch
import time
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class GPUWarmup:
    """Handles model preloading and GPU warmup for optimal performance"""
    
    def __init__(self, config):
        self.config = config
        self.warmup_stats = {}
        
    def warmup_gpu(self) -> Dict[str, Any]:
        """Perform comprehensive GPU warmup"""
        logger.info("Starting GPU warmup sequence...")
        start_time = time.time()
        
        stats = {
            'gpu_available': torch.cuda.is_available(),
            'warmup_time': 0,
            'models_warmed': [],
            'errors': []
        }
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, skipping GPU warmup")
            return stats
        
        try:
            # Basic GPU warmup
            self._basic_gpu_warmup()
            
            # Warmup embedding model
            embedding_stats = self._warmup_embedding_model()
            if embedding_stats:
                stats['models_warmed'].append('embedding_model')
                stats.update(embedding_stats)
            
            # Warmup LLM model if available
            llm_stats = self._warmup_llm_model()
            if llm_stats:
                stats['models_warmed'].append('llm_model')
                stats.update(llm_stats)
            
        except Exception as e:
            error_msg = f"GPU warmup error: {str(e)}"
            logger.error(error_msg)
            stats['errors'].append(error_msg)
        
        stats['warmup_time'] = time.time() - start_time
        logger.info(f"GPU warmup completed in {stats['warmup_time']:.2f}s")
        
        return stats
    
    def _basic_gpu_warmup(self):
        """Perform basic GPU operations to warm up"""
        logger.info("Performing basic GPU warmup...")
        
        # Create tensors and perform operations
        device = torch.device('cuda')
        
        # Matrix operations to warm up GPU
        for size in [512, 1024, 2048]:
            a = torch.randn(size, size, device=device, dtype=torch.float16)
            b = torch.randn(size, size, device=device, dtype=torch.float16)
            
            # Matrix multiplication
            c = torch.matmul(a, b)
            
            # Synchronize to ensure operations complete
            torch.cuda.synchronize()
        
        # Clear warmup tensors
        del a, b, c
        torch.cuda.empty_cache()
        
        logger.info("Basic GPU warmup completed")
    
    def _warmup_embedding_model(self) -> Optional[Dict[str, Any]]:
        """Warmup embedding model"""
        logger.info("Warming up embedding model...")
        
        try:
            from sentence_transformers import SentenceTransformer
            
            start_time = time.time()
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Load model
            model = SentenceTransformer(self.config.EMBEDDING_MODEL, device=device)
            
            if device == 'cuda':
                model = model.half()  # Use half precision
            
            # Warmup with dummy text
            warmup_texts = [
                "This is a warmup sentence for the embedding model.",
                "GPU warmup process for document search system.",
                "Testing model performance and memory allocation."
            ]
            
            # Generate embeddings
            embeddings = model.encode(
                warmup_texts,
                convert_to_tensor=False,
                normalize_embeddings=True,
                batch_size=len(warmup_texts)
            )
            
            # Cleanup
            del model, embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            warmup_time = time.time() - start_time
            logger.info(f"Embedding model warmed up in {warmup_time:.2f}s")
            
            return {
                'embedding_warmup_time': warmup_time,
                'embedding_model': self.config.EMBEDDING_MODEL
            }
            
        except Exception as e:
            logger.error(f"Embedding model warmup failed: {e}")
            return None
    
    def _warmup_llm_model(self) -> Optional[Dict[str, Any]]:
        """Warmup LLM model if available"""
        logger.info("Checking LLM model for warmup...")
        
        model_path = self.config.LOCAL_LLM_MODEL_PATH
        if not model_path.exists():
            logger.info("LLM model not found, skipping warmup")
            return None
        
        try:
            from llama_cpp import Llama
            
            start_time = time.time()
            
            # Load model with minimal configuration for warmup
            llm = Llama(
                model_path=str(model_path),
                n_ctx=512,  # Small context for warmup
                n_gpu_layers=self.config.N_GPU_LAYERS,
                verbose=False,
                n_threads=4,
                n_batch=256,
                use_mmap=True,
                use_mlock=False  # Don't lock memory for warmup
            )
            
            # Generate a small test response
            response = llm(
                "Test warmup prompt",
                max_tokens=5,
                temperature=0.1,
                echo=False
            )
            
            # Cleanup
            del llm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            warmup_time = time.time() - start_time
            logger.info(f"LLM model warmed up in {warmup_time:.2f}s")
            
            return {
                'llm_warmup_time': warmup_time,
                'llm_model_path': str(model_path)
            }
            
        except Exception as e:
            logger.error(f"LLM model warmup failed: {e}")
            return None
    
    def preload_models(self) -> Dict[str, Any]:
        """Preload models into GPU memory for immediate use"""
        logger.info("Preloading models for immediate use...")
        
        stats = {
            'preloaded_models': [],
            'total_gpu_memory_gb': 0,
            'errors': []
        }
        
        try:
            # Get initial GPU memory
            if torch.cuda.is_available():
                initial_memory = torch.cuda.memory_allocated() / 1024**3
                
                # Warmup GPU first
                warmup_stats = self.warmup_gpu()
                stats.update(warmup_stats)
                
                # Get final GPU memory
                final_memory = torch.cuda.memory_allocated() / 1024**3
                stats['total_gpu_memory_gb'] = final_memory - initial_memory
                
                logger.info(f"Models preloaded, using {stats['total_gpu_memory_gb']:.2f}GB GPU memory")
            
        except Exception as e:
            error_msg = f"Model preloading error: {str(e)}"
            logger.error(error_msg)
            stats['errors'].append(error_msg)
        
        return stats
    
    def optimize_for_inference(self):
        """Optimize GPU settings for inference workloads"""
        if not torch.cuda.is_available():
            return
        
        logger.info("Optimizing GPU for inference...")
        
        try:
            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Memory efficient attention
            torch.backends.cuda.enable_math_sdp(False)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            
            logger.info("GPU optimizations enabled")
            
        except Exception as e:
            logger.warning(f"Could not enable all GPU optimizations: {e}")

def warmup_system(config) -> Dict[str, Any]:
    """Lightweight system optimization - skip heavy warmup for faster startup"""
    warmer = GPUWarmup(config)
    
    # Only optimize for inference - skip model preloading for speed
    warmer.optimize_for_inference()
    
    # Return minimal stats
    return {
        'gpu_available': torch.cuda.is_available(),
        'optimizations_enabled': True,
        'heavy_warmup_skipped': True,
        'message': 'GPU optimized, models will warm up on first use'
    }