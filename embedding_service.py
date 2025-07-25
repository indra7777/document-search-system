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
        self._load_model()
    
    def _load_model(self):
        """Load the BGE embedding model with CUDA optimization"""
        try:
            self.logger.info(f"Loading embedding model: {self.config.EMBEDDING_MODEL}")
            
            # Check for CUDA availability and use RTX 4090 optimally
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.logger.info(f"Using device: {device}")
            
            if device == 'cuda':
                self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
                # Enable mixed precision for RTX 4090
                torch.backends.cudnn.benchmark = True
            
            # Load model on CPU first, then move to device to avoid meta tensor issues
            self.model = SentenceTransformer(self.config.EMBEDDING_MODEL, device='cpu')
            
            # Move to target device using to_empty() method for meta tensors
            if device == 'cuda':
                try:
                    # Move model to CUDA
                    self.model = self.model.to(device)
                    # Optimize for RTX 4090 with larger batch sizes
                    self.model.max_seq_length = 8192  # Utilize more memory
                except Exception as device_error:
                    self.logger.warning(f"Failed to move model to CUDA: {device_error}")
                    # Keep on CPU if CUDA transfer fails
                    device = 'cpu'
                
            self.logger.info(f"Embedding model loaded successfully on {device}")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {str(e)}")
            # Fallback to a smaller model if BGE fails
            self.logger.info("Falling back to all-MiniLM-L6-v2")
            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
                if device == 'cuda':
                    self.model = self.model.to(device)
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
        
        # Generate embeddings in large batches for RTX 4090 optimization
        batch_size = 64 if torch.cuda.is_available() else 16  # Reduced batch size for safety
        embeddings = []
        
        for i in range(0, len(truncated_texts), batch_size):
            batch_texts = truncated_texts[i:i + batch_size]
            
            # Use mixed precision for faster inference on RTX 4090
            with torch.autocast(device_type='cuda', dtype=torch.float16) if torch.cuda.is_available() else torch.no_grad():
                batch_embeddings = self.model.encode(
                    batch_texts,
                    convert_to_tensor=False,
                    show_progress_bar=True,
                    normalize_embeddings=True,
                    batch_size=batch_size
                )
            embeddings.extend(batch_embeddings)
        
        # Add embeddings to documents
        for doc, embedding in zip(documents, embeddings):
            doc['embedding'] = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            doc['embedding_model'] = self.config.EMBEDDING_MODEL
        
        self.logger.info("Embeddings generated successfully")
        return documents
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query
        
        Args:
            query: Search query string
            
        Returns:
            Query embedding as numpy array
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        embedding = self.model.encode(
            query,
            convert_to_tensor=False,
            normalize_embeddings=True
        )
        
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