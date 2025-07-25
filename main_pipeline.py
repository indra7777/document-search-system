#!/usr/bin/env python3
"""
Single-step Document Search Pipeline
Run once to build the index, then use for searching without rebuilding
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
import json
import time

from config import Config
from document_processor import DocumentProcessor
from embedding_service import EmbeddingService
from vector_database import VectorDatabase
from llm_service_cpp import LLMServiceCPP as LLMService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentSearchPipeline:
    def __init__(self):
        self.config = Config()
        self.config.create_directories()
        
        # Initialize services
        self.doc_processor = DocumentProcessor(self.config)
        self.embedding_service = EmbeddingService(self.config)
        self.vector_db = VectorDatabase(self.config)
        self.llm_service = LLMService(self.config)
        
        # State tracking
        self.index_metadata_path = self.config.BASE_DIR / "index_metadata.json"
        
    def build_index(self, data_paths: List[str], force_rebuild: bool = False):
        """
        Build vector index from documents (one-time operation)
        
        Args:
            data_paths: List of file or directory paths to process
            force_rebuild: Force rebuild even if index exists
        """
        logger.info("Starting index building process...")
        
        # Check if index already exists and is up-to-date
        if not force_rebuild and self._is_index_current(data_paths):
            logger.info("Index is current, skipping rebuild")
            return
        
        start_time = time.time()
        
        # Step 1: Process documents
        logger.info("Step 1: Processing documents with Docling...")
        processed_docs = self.doc_processor.process_documents(data_paths)
        
        if not processed_docs:
            logger.warning("No documents were processed")
            return
        
        logger.info(f"Processed {len(processed_docs)} documents")
        
        # Step 2: Chunk documents
        logger.info("Step 2: Chunking documents...")
        all_chunks = []
        for doc in processed_docs:
            chunks = self.doc_processor.chunk_document(doc)
            all_chunks.extend(chunks)
        
        logger.info(f"Created {len(all_chunks)} document chunks")
        
        # Step 3: Generate embeddings
        logger.info("Step 3: Generating embeddings with BGE model...")
        embedded_chunks = self.embedding_service.embed_documents(all_chunks)
        
        # Step 4: Build vector index
        logger.info("Step 4: Building FAISS vector index...")
        
        # Create new index if it doesn't exist or force rebuild
        if force_rebuild or not self.vector_db.load_index(self.config.EMBEDDING_DIMENSION):
            self.vector_db.create_index(self.config.EMBEDDING_DIMENSION, len(embedded_chunks))
        
        # Add documents to index
        self.vector_db.add_documents(embedded_chunks)
        
        # Step 5: Save index and metadata
        logger.info("Step 5: Saving index to disk...")
        self.vector_db.save_index()
        
        # Save metadata about the build
        metadata = {
            'build_timestamp': time.time(),
            'data_paths': data_paths,
            'total_documents': len(processed_docs),
            'total_chunks': len(all_chunks),
            'embedding_model': self.config.EMBEDDING_MODEL,
            'llm_model': str(self.config.LOCAL_LLM_MODEL_PATH)
        }
        
        with open(self.index_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        build_time = time.time() - start_time
        logger.info(f"Index building completed in {build_time:.2f} seconds")
        logger.info(f"Indexed {len(processed_docs)} documents with {len(all_chunks)} chunks")
        
    def search(self, query: str, top_k: int = 5, threshold: float = 0.7, use_llm: bool = True) -> Dict[str, Any]:
        """
        Search the pre-built index (fast operation)
        
        Args:
            query: Search query
            top_k: Number of results to return
            threshold: Similarity threshold
            use_llm: Whether to generate LLM response
            
        Returns:
            Search results with optional LLM response
        """
        logger.info(f"Searching for: '{query}'")
        
        # Load index if not already loaded
        if not self._ensure_index_loaded():
            raise RuntimeError("No index found. Please build the index first.")
        
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = self.embedding_service.embed_query(query)
        
        # Search vector database
        search_results = self.vector_db.search(
            query_embedding,
            top_k=top_k,
            threshold=threshold
        )
        
        result = {
            'query': query,
            'search_results': search_results,
            'search_time': time.time() - start_time,
            'results_count': len(search_results)
        }
        
        # Generate LLM response if requested
        if use_llm and search_results:
            llm_start = time.time()
            llm_response = self.llm_service.generate_response(query, search_results)
            result['llm_response'] = llm_response
            result['llm_time'] = time.time() - llm_start
        
        logger.info(f"Search completed in {result['search_time']:.2f}s, found {len(search_results)} results")
        
        return result
    
    def _is_index_current(self, data_paths: List[str]) -> bool:
        """Check if the index is current and doesn't need rebuilding"""
        if not self.index_metadata_path.exists():
            return False
        
        if not self.vector_db.db_path.exists():
            return False
        
        try:
            with open(self.index_metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Check if data paths have changed
            if set(metadata.get('data_paths', [])) != set(data_paths):
                logger.info("Data paths changed, rebuild required")
                return False
            
            # Check if any source files are newer than the index
            index_time = metadata.get('build_timestamp', 0)
            
            for path in data_paths:
                path_obj = Path(path)
                if path_obj.is_file():
                    if path_obj.stat().st_mtime > index_time:
                        logger.info(f"File {path} modified since last build")
                        return False
                elif path_obj.is_dir():
                    for file_path in path_obj.rglob('*'):
                        if file_path.is_file() and file_path.stat().st_mtime > index_time:
                            logger.info(f"File {file_path} modified since last build")
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking index currency: {str(e)}")
            return False
    
    def _ensure_index_loaded(self) -> bool:
        """Ensure the vector index is loaded"""
        try:
            if self.vector_db.index is None or self.vector_db.index.ntotal == 0:
                return self.vector_db.load_index(self.config.EMBEDDING_DIMENSION)
            return True
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index"""
        stats = self.vector_db.get_stats()
        
        if self.index_metadata_path.exists():
            with open(self.index_metadata_path, 'r') as f:
                metadata = json.load(f)
            stats.update(metadata)
        
        return stats
    
    def interactive_search(self):
        """Start interactive search mode"""
        if not self._ensure_index_loaded():
            print("No index found. Please build the index first with --build")
            return
        
        print("\n🔍 Interactive Document Search")
        print("Type 'quit' to exit, 'stats' for index statistics")
        print("-" * 50)
        
        while True:
            try:
                query = input("\nEnter your search query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                elif query.lower() == 'stats':
                    stats = self.get_index_stats()
                    print(f"\nIndex Statistics:")
                    print(f"  Documents: {stats.get('total_documents', 0)}")
                    print(f"  Chunks: {stats.get('total_chunks', 0)}")
                    print(f"  Model: {stats.get('embedding_model', 'Unknown')}")
                    continue
                elif not query:
                    continue
                
                # Perform search
                results = self.search(query, top_k=3, use_llm=True)
                
                # Display results
                print(f"\n🤖 AI Response:")
                if 'llm_response' in results:
                    print(results['llm_response']['answer'])
                    print(f"\n📊 Found {results['results_count']} relevant sources:")
                    
                    for i, result in enumerate(results['search_results'], 1):
                        print(f"  {i}. {result['file_name']} (similarity: {result['similarity_score']:.3f})")
                else:
                    print("No relevant documents found.")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {str(e)}")
        
        print("\nGoodbye!")

def main():
    parser = argparse.ArgumentParser(description="Document Search Pipeline")
    parser.add_argument('--build', nargs='+', help='Build index from data paths')
    parser.add_argument('--force-rebuild', action='store_true',
                       help='Force rebuild even if index exists')
    parser.add_argument('--search', type=str, help='Search query')
    parser.add_argument('--interactive', action='store_true',
                       help='Start interactive search mode')
    parser.add_argument('--stats', action='store_true',
                       help='Show index statistics')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of results to return')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='Similarity threshold')
    parser.add_argument('--no-llm', action='store_true',
                       help='Disable LLM response generation')

    args = parser.parse_args()
    
    pipeline = DocumentSearchPipeline()
    
    if args.build:
        pipeline.build_index(args.build, force_rebuild=args.force_rebuild)
    
    elif args.search:
        try:
            results = pipeline.search(
                args.search,
                top_k=args.top_k,
                threshold=args.threshold,
                use_llm=not args.no_llm
            )
            
            print(f"\n🔍 Query: {results['query']}")
            print(f"⏱️  Search time: {results['search_time']:.2f}s")
            print(f"📊 Results: {results['results_count']}")
            
            if 'llm_response' in results:
                print(f"\n🤖 AI Response:")
                print("=" * 80)
                print(results['llm_response']['answer'])
                print("=" * 80)
                
                print(f"\n📋 Sources Used:")
                for i, result in enumerate(results['search_results'], 1):
                    print(f"  {i}. 📄 {result['file_name']}")
                    print(f"     📊 Similarity: {result['similarity_score']:.3f}")
                    print(f"     📁 Path: {result['file_path']}")
                    print(f"     🔗 Click to open: file://{os.path.abspath(result['file_path'])}")
                    print()
            else:
                print("❌ No relevant documents found.")
                print("💡 Try:")
                print("   - Using different keywords")
                print("   - Lowering the similarity threshold")
                print("   - Checking if your documents contain the information you're looking for")
            
        except Exception as e:
            print(f"Search error: {str(e)}")
    
    elif args.interactive:
        pipeline.interactive_search()
    
    elif args.stats:
        stats = pipeline.get_index_stats()
        print("📊 Index Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()