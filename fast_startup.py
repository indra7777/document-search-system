"""
Fast Startup Mode - Optimized for minimal loading time
Skips heavy models and warmup operations for faster Streamlit startup
"""
import logging
import os

logger = logging.getLogger(__name__)

# Environment variable to control startup mode
FAST_STARTUP = os.getenv('FAST_STARTUP', 'true').lower() == 'true'

def should_skip_heavy_operations():
    """Check if we should skip heavy operations for fast startup"""
    return FAST_STARTUP

def skip_if_fast_startup(operation_name, heavy_func, fallback_func=None):
    """Decorator to skip heavy operations in fast startup mode"""
    if should_skip_heavy_operations():
        logger.info(f"⚡ Fast startup: Skipping {operation_name}")
        return fallback_func() if fallback_func else None
    else:
        return heavy_func()

def log_fast_startup_status():
    """Log the current startup mode"""
    if FAST_STARTUP:
        logger.info("⚡ FAST STARTUP MODE ENABLED - Heavy models will load on demand")
    else:
        logger.info("🔥 FULL STARTUP MODE - All models will preload")

class FastStartupConfig:
    """Configuration for fast startup optimizations"""
    
    # Skip these heavy operations during startup
    SKIP_MODEL_WARMUP = FAST_STARTUP
    SKIP_COMPREHENSIVE_WARMUP = FAST_STARTUP
    SKIP_HEAVY_MODEL_PRELOAD = FAST_STARTUP
    
    # Lazy load these services
    LAZY_LOAD_DOCUMENT_ANALYZER = FAST_STARTUP  # TrOCR model
    LAZY_LOAD_LLM_SERVICE = FAST_STARTUP        # GGUF model
    
    # Optimization settings
    USE_MINIMAL_WARMUP = FAST_STARTUP
    ENABLE_GPU_OPTIMIZATIONS = True  # Always enable GPU opts
    
    @classmethod
    def get_startup_message(cls):
        if FAST_STARTUP:
            return "⚡ Fast startup enabled - models load on demand for 3-5x faster page loads"
        else:
            return "🔥 Full preload mode - all models loaded upfront"