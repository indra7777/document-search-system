import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple
import logging
from pathlib import Path
import pickle
import torch

class VectorDatabase:
    def __init__(self, config):
        self.config = config
        self.index = None
        self.documents = []
        self.logger = logging.getLogger(__name__)
        self.db_path = config.VECTOR_DB_PATH / "faiss_index"
        self.metadata_path = config.VECTOR_DB_PATH / "metadata.pkl"
        self._gpu_enabled = False
        self._gpu_resources = None
        
        # Create vector DB directory
        config.VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
        
        # Initialize GPU resources if available
        self._init_gpu_resources()
    
    def _init_gpu_resources(self):
        """Initialize GPU resources for FAISS"""
        try:
            if torch.cuda.is_available() and faiss.get_num_gpus() > 0:
                self._gpu_resources = faiss.StandardGpuResources()
                # Set memory fraction for FAISS (leave room for other GPU operations)
                self._gpu_resources.setDefaultNullStreamAllDevices()
                self.logger.info(f"Initialized FAISS GPU resources on {torch.cuda.get_device_name(0)}")
                return True
        except Exception as e:
            self.logger.warning(f"Could not initialize FAISS GPU resources: {e}")
        return False
    
    def create_index(self, embedding_dimension: int, num_docs: int = 0):
        """Create a new GPU-optimized FAISS index"""
        try:
            # For small datasets (< 1000 docs), use simple flat index
            # For larger datasets, use IVF index with appropriate cluster count
            if num_docs < 1000:
                self.logger.info("Creating Flat index for small dataset")
                self.index = faiss.IndexFlatIP(embedding_dimension)  # Inner product for cosine similarity
            else:
                # Use IVF (Inverted File) index for better performance on large datasets
                # Number of clusters should be roughly sqrt(num_documents), but at least 4*sqrt(num_docs)
                nlist = min(1024, max(16, int(4 * (num_docs ** 0.5))))
                self.logger.info(f"Creating IVF index with {nlist} clusters")
                quantizer = faiss.IndexFlatIP(embedding_dimension)
                self.index = faiss.IndexIVFFlat(quantizer, embedding_dimension, nlist)
            
            # Automatically move to GPU if available
            self.move_to_gpu()
            
            self.logger.info(f"Created FAISS index with dimension {embedding_dimension}")
            
        except Exception as e:
            self.logger.error(f"Failed to create FAISS index: {str(e)}")
            # Fallback to simple flat index
            self.index = faiss.IndexFlatIP(embedding_dimension)
    
    def move_to_gpu(self):
        """Move FAISS index to GPU for faster operations"""
        try:
            if self.index is not None and self._gpu_resources is not None and not self._gpu_enabled:
                # Move index to GPU
                self.index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, self.index)
                self._gpu_enabled = True
                self.logger.info("🚀 FAISS index moved to GPU for accelerated search")
                return True
        except Exception as e:
            self.logger.warning(f"Could not move FAISS index to GPU: {e}")
        return False
    
    def move_to_cpu(self):
        """Move FAISS index back to CPU"""
        try:
            if self.index is not None and self._gpu_enabled:
                self.index = faiss.index_gpu_to_cpu(self.index)
                self._gpu_enabled = False
                self.logger.info("FAISS index moved back to CPU")
                return True
        except Exception as e:
            self.logger.warning(f"Could not move FAISS index to CPU: {e}")
        return False
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents with embeddings to the vector database"""
        if not documents:
            return
        
        embeddings = []
        for doc in documents:
            if 'embedding' in doc:
                embedding = np.array(doc['embedding'], dtype=np.float32)
                embeddings.append(embedding)
                self.documents.append(doc)
        
        if not embeddings:
            self.logger.warning("No embeddings found in documents")
            return
        
        embeddings_matrix = np.vstack(embeddings)
        
        # Train the index if it's an IVF index
        if hasattr(self.index, 'train') and not self.index.is_trained:
            self.logger.info("Training FAISS index...")
            self.index.train(embeddings_matrix)
        
        # Add embeddings to index
        self.index.add(embeddings_matrix)
        
        self.logger.info(f"Added {len(embeddings)} documents to vector database")
        self.logger.info(f"Total documents in database: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector similarity
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of search results with metadata
        """
        if self.index is None or self.index.ntotal == 0:
            self.logger.warning("Vector database is empty")
            return []
        
        # Ensure query embedding is 2D array
        query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        
        # Search for similar vectors
        similarities, indices = self.index.search(query_vector, top_k)
        
        results = []
        for i, (similarity, doc_idx) in enumerate(zip(similarities[0], indices[0])):
            if doc_idx >= 0 and similarity >= threshold:
                if doc_idx < len(self.documents):
                    result = self.documents[doc_idx].copy()
                    result['similarity_score'] = float(similarity)
                    result['rank'] = i + 1
                    results.append(result)
        
        return results
    
    def save_index(self):
        """Save the FAISS index and metadata to disk"""
        try:
            # Save FAISS index
            if self.index is not None:
                # Convert GPU index to CPU before saving if needed
                if self._gpu_enabled:
                    cpu_index = faiss.index_gpu_to_cpu(self.index)
                    faiss.write_index(cpu_index, str(self.db_path))
                    self.logger.info("Converted GPU index to CPU for saving")
                else:
                    faiss.write_index(self.index, str(self.db_path))
            
            # Save document metadata
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.documents, f)
            
            self.logger.info(f"Vector database saved to {self.config.VECTOR_DB_PATH}")
            
        except Exception as e:
            self.logger.error(f"Failed to save vector database: {str(e)}")
    
    def load_index(self, embedding_dimension: int):
        """Load the FAISS index and metadata from disk with automatic GPU loading"""
        try:
            if self.db_path.exists():
                # Load FAISS index from disk
                self.index = faiss.read_index(str(self.db_path))
                
                # Automatically move to GPU for accelerated operations
                gpu_success = self.move_to_gpu()
                if gpu_success:
                    self.logger.info("🚀 Vector index automatically loaded to GPU")
                
                # Load document metadata
                if self.metadata_path.exists():
                    with open(self.metadata_path, 'rb') as f:
                        self.documents = pickle.load(f)
                
                self.logger.info(f"Vector database loaded with {self.index.ntotal} documents")
                return True
            else:
                self.logger.info("No existing vector database found, creating new GPU-optimized index")
                self.create_index(embedding_dimension, num_docs=0)
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to load vector database: {str(e)}")
            # Create new index if loading fails
            self.create_index(embedding_dimension, num_docs=0)
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the vector database"""
        stats = {
            'total_documents': len(self.documents),
            'index_size': self.index.ntotal if self.index else 0,
            'is_trained': self.index.is_trained if hasattr(self.index, 'is_trained') else True,
            'database_path': str(self.config.VECTOR_DB_PATH),
            'gpu_enabled': self._gpu_enabled,
            'gpu_available': torch.cuda.is_available() and faiss.get_num_gpus() > 0
        }
        
        # Add GPU memory info if available
        if self._gpu_enabled and torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                stats['gpu_memory_allocated'] = f"{gpu_memory:.2f}GB"
            except:
                pass
                
        return stats
    
    def delete_index(self):
        """Delete the vector database files"""
        try:
            if self.db_path.exists():
                self.db_path.unlink()
            if self.metadata_path.exists():
                self.metadata_path.unlink()
            
            self.index = None
            self.documents = []
            
            self.logger.info("Vector database deleted")
            
        except Exception as e:
            self.logger.error(f"Failed to delete vector database: {str(e)}")