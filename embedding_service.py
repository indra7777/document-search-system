import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import logging
from pathlib import Path
import pickle

class EmbeddingService:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.logger = logging.getLogger(__name__)
        self._query_cache = {}  # Cache for query embeddings
        self._load_model()
    
    def _load_model(self):
        """Load the BGE embedding model with optimized GPU utilization"""
        try:
            self.logger.info(f"Loading embedding model: {self.config.EMBEDDING_MODEL}")
            
            # Check for CUDA availability and use RTX 4090 optimally
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.logger.info(f"Using device: {device}")
            
            if device == 'cuda':
                self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
                # GPU optimizations for RTX 4090
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                # Enable memory efficient attention
                torch.backends.cuda.enable_math_sdp(False)
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
            
            # Load model directly on target device for efficiency
            try:
                self.model = SentenceTransformer(self.config.EMBEDDING_MODEL, device=device)
                
                if device == 'cuda':
                    # Optimize model for RTX 4090
                    self.model.max_seq_length = min(8192, self.model.max_seq_length)
                    # Set model to half precision for memory efficiency
                    self.model = self.model.half()
                    
                self.logger.info(f"Embedding model loaded successfully on {device}")
                
            except Exception as direct_load_error:
                self.logger.warning(f"Direct GPU loading failed: {direct_load_error}")
                # Fallback: Load on CPU then move
                self.model = SentenceTransformer(self.config.EMBEDDING_MODEL, device='cpu')
                if device == 'cuda':
                    self.model = self.model.to(device).half()
                    
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {str(e)}")
            # Fallback to a smaller model if BGE fails
            self.logger.info("Falling back to all-MiniLM-L6-v2")
            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
                if device == 'cuda':
                    self.model = self.model.half()
                self.logger.info(f"Fallback model loaded successfully on {device}")
            except Exception as fallback_error:
                self.logger.error(f"Fallback model also failed: {fallback_error}")
                # Last resort - CPU only
                self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
                self.logger.info("Using CPU-only fallback model")
    
    def embed_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for document chunks
        
        Args:
            documents: List of document chunks
            
        Returns:
            List of documents with embeddings added
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        texts = [doc['content'] for doc in documents]
        
        # Properly truncate texts using the model's tokenizer
        # BGE model has max sequence length of 512 tokens
        truncated_texts = []
        tokenizer = self.model.tokenizer
        
        for text in texts:
            # Tokenize and truncate properly
            tokens = tokenizer.encode(text, add_special_tokens=True, max_length=510, truncation=True)
            # Decode back to text (removing special tokens)
            truncated_text = tokenizer.decode(tokens, skip_special_tokens=True)
            truncated_texts.append(truncated_text)
        
        self.logger.info(f"Generating embeddings for {len(truncated_texts)} documents")
        
        # Dynamic batch size optimization for RTX 4090
        from gpu_optimizer import gpu_optimizer
        base_batch_size = 64 if torch.cuda.is_available() else 16
        batch_size = gpu_optimizer.optimize_batch_size(base_batch_size)
        
        self.logger.info(f"Using optimized batch size: {batch_size}")
        embeddings = []
        
        for i in range(0, len(truncated_texts), batch_size):
            batch_texts = truncated_texts[i:i + batch_size]
            
            # Use mixed precision for faster inference on RTX 4090
            if torch.cuda.is_available():
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        convert_to_tensor=False,
                        show_progress_bar=(i == 0),  # Only show progress for first batch
                        normalize_embeddings=True,
                        batch_size=batch_size
                    )
            else:
                with torch.no_grad():
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        convert_to_tensor=False,
                        show_progress_bar=(i == 0),
                        normalize_embeddings=True,
                        batch_size=batch_size
                    )
            embeddings.extend(batch_embeddings)
            
            # Monitor GPU memory and cleanup if needed
            if torch.cuda.is_available() and i % (batch_size * 4) == 0:
                gpu_optimizer.monitor_memory_usage()
        
        # Add embeddings to documents
        for doc, embedding in zip(documents, embeddings):
            doc['embedding'] = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            doc['embedding_model'] = self.config.EMBEDDING_MODEL
        
        self.logger.info("Embeddings generated successfully")
        return documents
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query with caching and GPU optimization
        
        Args:
            query: Search query string
            
        Returns:
            Query embedding as numpy array
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        # Check cache first for instant results
        query_hash = hash(query.strip().lower())
        if query_hash in self._query_cache:
            self.logger.debug("Using cached embedding for query")
            return self._query_cache[query_hash]
        
        # Generate embedding with GPU optimization
        if torch.cuda.is_available():
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                embedding = self.model.encode(
                    query,
                    convert_to_tensor=False,
                    normalize_embeddings=True,
                    show_progress_bar=False,  # Disable for single queries
                    batch_size=1
                )
        else:
            with torch.no_grad():
                embedding = self.model.encode(
                    query,
                    convert_to_tensor=False,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=1
                )
        
        # Cache the result (limit cache size to prevent memory issues)
        if len(self._query_cache) < 100:  # Keep last 100 queries
            self._query_cache[query_hash] = embedding
        elif len(self._query_cache) >= 100:
            # Remove oldest entry (simple cleanup)
            self._query_cache.pop(next(iter(self._query_cache)))
            self._query_cache[query_hash] = embedding
        
        return embedding
    
    def save_embeddings(self, documents: List[Dict[str, Any]], file_path: str):
        """Save embeddings to disk"""
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(documents, f)
            self.logger.info(f"Embeddings saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save embeddings: {str(e)}")
    
    def load_embeddings(self, file_path: str) -> List[Dict[str, Any]]:
        """Load embeddings from disk"""
        try:
            with open(file_path, 'rb') as f:
                documents = pickle.load(f)
            self.logger.info(f"Embeddings loaded from {file_path}")
            return documents
        except Exception as e:
            self.logger.error(f"Failed to load embeddings: {str(e)}")
            return []
    
    def compute_similarity(self, query_embedding: np.ndarray, doc_embeddings: List[np.ndarray]) -> List[float]:
        """
        Compute cosine similarity between query and document embeddings
        
        Args:
            query_embedding: Query embedding vector
            doc_embeddings: List of document embedding vectors
            
        Returns:
            List of similarity scores
        """
        similarities = []
        
        for doc_embedding in doc_embeddings:
            # Ensure embeddings are numpy arrays
            query_vec = np.array(query_embedding)
            doc_vec = np.array(doc_embedding)
            
            # Compute cosine similarity
            similarity = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
            similarities.append(float(similarity))
        
        return similarities