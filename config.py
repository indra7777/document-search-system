import os
from pathlib import Path

class Config:
    # Base paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    EMBEDDINGS_DIR = BASE_DIR / "embeddings"
    MODELS_DIR = BASE_DIR / "models"
    
    # Document processing
    SUPPORTED_FORMATS = ['.pdf', '.docx', '.txt', '.html', '.md']
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Embedding model configuration - optimized for RTX 4090
    EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"  # High-accuracy BGE model
    EMBEDDING_DIMENSION = 1024
    
    # GPU optimization settings for RTX 4090
    CUDA_BATCH_SIZE = 128  # Large batch size for RTX 4090
    USE_MIXED_PRECISION = True
    GPU_MEMORY_FRACTION = 0.9  # Use 90% of GPU memory
    ENABLE_GPU_OPTIMIZATIONS = True
    
    # Vector database
    VECTOR_DB_PATH = BASE_DIR / "vector_db"
    
    # Local LLM configuration - optimized for RTX 4090
    LOCAL_LLM_MODEL_PATH = BASE_DIR / "models" / "llama-model.gguf"  # Path to GGUF model
    LLM_TEMPERATURE = 0.1
    MAX_TOKENS = 2048
    N_CTX = 8192  # Larger context window for RTX 4090
    N_GPU_LAYERS = -1  # Use all GPU layers with RTX 4090
    
    # Advanced GPU settings
    TENSOR_SPLIT = None  # Auto-distribute across GPU
    ROPE_SCALING_TYPE = 0  # Default rope scaling
    MUL_MAT_Q = True  # Enable quantized matrix multiplication
    SPLIT_MODE = 1  # GPU split mode
    
    # Search configuration
    TOP_K_RESULTS = 5
    SIMILARITY_THRESHOLD = 0.4  # Lowered for better recall
    
    # Web interface
    HOST = "0.0.0.0"  # Bind to all interfaces for external access
    PORT = 8501
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        for dir_path in [cls.DATA_DIR, cls.EMBEDDINGS_DIR, cls.MODELS_DIR, cls.VECTOR_DB_PATH]:
            dir_path.mkdir(parents=True, exist_ok=True)