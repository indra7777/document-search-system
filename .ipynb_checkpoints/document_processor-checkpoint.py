import os
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
import hashlib

class DocumentProcessor:
    def __init__(self, config):
        self.config = config
        self.converter = DocumentConverter()
        self.logger = logging.getLogger(__name__)
        
    def process_documents(self, data_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Process documents from given paths using docling
        
        Args:
            data_paths: List of file or directory paths
            
        Returns:
            List of processed documents with metadata
        """
        processed_docs = []
        
        for path in data_paths:
            path_obj = Path(path)
            
            if path_obj.is_file():
                doc = self._process_single_file(path_obj)
                if doc:
                    processed_docs.append(doc)
            elif path_obj.is_dir():
                for file_path in self._get_supported_files(path_obj):
                    doc = self._process_single_file(file_path)
                    if doc:
                        processed_docs.append(doc)
                        
        return processed_docs
    
    def _process_single_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single file using docling"""
        try:
            if file_path.suffix.lower() not in self.config.SUPPORTED_FORMATS:
                self.logger.warning(f"Unsupported format: {file_path}")
                return None
                
            # Convert document using docling
            result = self.converter.convert(str(file_path))
            
            # Extract text content
            text_content = result.document.export_to_markdown()
            
            # Create document metadata
            doc_metadata = {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'file_size': file_path.stat().st_size,
                'file_type': file_path.suffix.lower(),
                'content': text_content,
                'doc_id': self._generate_doc_id(file_path),
                'page_count': getattr(result.document, 'page_count', 1),
                'processed_timestamp': pd.Timestamp.now().isoformat()
            }
            
            self.logger.info(f"Successfully processed: {file_path.name}")
            return doc_metadata
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {str(e)}")
            return None
    
    def _get_supported_files(self, directory: Path) -> List[Path]:
        """Get all supported files from directory recursively"""
        supported_files = []
        
        for ext in self.config.SUPPORTED_FORMATS:
            pattern = f"**/*{ext}"
            supported_files.extend(directory.glob(pattern))
            
        return supported_files
    
    def _generate_doc_id(self, file_path: Path) -> str:
        """Generate unique document ID based on file path and content"""
        file_info = f"{file_path}{file_path.stat().st_mtime}"
        return hashlib.md5(file_info.encode()).hexdigest()
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split document content into chunks for embedding
        
        Args:
            document: Document dictionary with content
            
        Returns:
            List of document chunks with metadata
        """
        content = document['content']
        chunks = []
        
        # Simple chunking by character count
        chunk_size = self.config.CHUNK_SIZE
        overlap = self.config.CHUNK_OVERLAP
        
        start = 0
        chunk_id = 0
        
        while start < len(content):
            end = start + chunk_size
            chunk_text = content[start:end]
            
            # Try to end at a sentence boundary
            if end < len(content):
                last_period = chunk_text.rfind('.')
                last_newline = chunk_text.rfind('\n')
                boundary = max(last_period, last_newline)
                
                if boundary > start + chunk_size // 2:
                    end = start + boundary + 1
                    chunk_text = content[start:end]
            
            chunk = {
                'chunk_id': f"{document['doc_id']}_{chunk_id}",
                'doc_id': document['doc_id'],
                'chunk_index': chunk_id,
                'content': chunk_text.strip(),
                'start_pos': start,
                'end_pos': end,
                'file_path': document['file_path'],
                'file_name': document['file_name'],
                'file_type': document['file_type']
            }
            
            chunks.append(chunk)
            start = end - overlap
            chunk_id += 1
            
        return chunks