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
    
    # GPU optimization settings
    CUDA_BATCH_SIZE = 128  # Large batch size for RTX 4090
    USE_MIXED_PRECISION = True
    
    # Vector database
    VECTOR_DB_PATH = BASE_DIR / "vector_db"
    
    # Local LLM configuration
    LOCAL_LLM_MODEL = "phi3:mini"  # Ollama model name
    LLM_TEMPERATURE = 0.1
    MAX_TOKENS = 2048
    
    # Search configuration
    TOP_K_RESULTS = 5
    SIMILARITY_THRESHOLD = 0.7
    
    # Web interface
    HOST = "localhost"
    PORT = 8501
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        for dir_path in [cls.DATA_DIR, cls.EMBEDDINGS_DIR, cls.MODELS_DIR, cls.VECTOR_DB_PATH]:
            dir_path.mkdir(parents=True, exist_ok=True)