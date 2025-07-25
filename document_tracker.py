"""
Document Tracking and Redundancy Detection System
Handles document versioning, duplicate detection, and processing history
"""
import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class DocumentTracker:
    """Tracks processed documents and handles redundancy detection"""
    
    def __init__(self, base_dir: str = "/workspace"):
        self.base_dir = Path(base_dir)
        self.tracking_file = self.base_dir / "document_tracking.json"
        self.processed_dir = self.base_dir / "processed_files"
        self.uploads_dir = self.base_dir / "uploads"
        
        # Create directories
        self.processed_dir.mkdir(exist_ok=True)
        self.uploads_dir.mkdir(exist_ok=True)
        
        # Load existing tracking data
        self.tracking_data = self._load_tracking_data()
    
    def _load_tracking_data(self) -> Dict[str, Any]:
        """Load document tracking data from JSON file"""
        if self.tracking_file.exists():
            try:
                with open(self.tracking_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading tracking data: {e}")
        
        return {
            "documents": {},  # file_hash -> document info
            "file_paths": {},  # original_path -> file_hash
            "processing_history": [],  # chronological processing log
            "stats": {
                "total_processed": 0,
                "duplicates_detected": 0,
                "last_cleanup": None
            }
        }
    
    def _save_tracking_data(self):
        """Save tracking data to JSON file"""
        try:
            with open(self.tracking_file, 'w', encoding='utf-8') as f:
                json.dump(self.tracking_data, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"Error saving tracking data: {e}")
    
    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file content"""
        hasher = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return None
    
    def is_document_processed(self, file_path: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check if document has been processed before
        
        Returns:
            (is_processed, document_info)
        """
        file_hash = self.calculate_file_hash(file_path)
        if not file_hash:
            return False, None
        
        # Check by content hash (most reliable)
        if file_hash in self.tracking_data["documents"]:
            doc_info = self.tracking_data["documents"][file_hash]
            logger.info(f"Document found by hash: {file_path} -> {doc_info['original_name']}")
            return True, doc_info
        
        # Check by file path (less reliable, for renamed files)
        file_path_str = str(Path(file_path).resolve())
        if file_path_str in self.tracking_data["file_paths"]:
            linked_hash = self.tracking_data["file_paths"][file_path_str]
            if linked_hash in self.tracking_data["documents"]:
                doc_info = self.tracking_data["documents"][linked_hash]
                logger.info(f"Document found by path: {file_path} -> {doc_info['original_name']}")
                return True, doc_info
        
        return False, None
    
    def register_document(self, file_path: str, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register a newly processed document
        
        Args:
            file_path: Path to the document
            analysis_result: Result from document analysis
            
        Returns:
            Updated document info with tracking metadata
        """
        file_hash = self.calculate_file_hash(file_path)
        if not file_hash:
            return analysis_result
        
        file_path_obj = Path(file_path)
        timestamp = datetime.now().isoformat()
        
        # Create document info
        doc_info = {
            "file_hash": file_hash,
            "original_name": file_path_obj.name,
            "original_path": str(file_path_obj.resolve()),
            "file_size": file_path_obj.stat().st_size,
            "processing_timestamp": timestamp,
            "analysis_file": analysis_result.get("analysis_file"),
            "content_summary": {
                "total_pages": len(analysis_result.get("pages", [])),
                "total_tables": len(analysis_result.get("tables", [])),
                "total_images": len(analysis_result.get("images", [])),
                "has_handwriting": len(analysis_result.get("handwriting_text", [])) > 0,
                "content_types": self._extract_content_types(analysis_result)
            },
            "processing_version": "1.0",  # Track which version of analyzer was used
            "vector_db_indexed": False,   # Track if added to vector database
            "last_accessed": timestamp
        }
        
        # Store in tracking data
        self.tracking_data["documents"][file_hash] = doc_info
        self.tracking_data["file_paths"][str(file_path_obj.resolve())] = file_hash
        
        # Add to processing history
        self.tracking_data["processing_history"].append({
            "timestamp": timestamp,
            "action": "processed",
            "file_hash": file_hash,
            "file_name": file_path_obj.name,
            "file_path": str(file_path_obj.resolve())
        })
        
        # Update stats
        self.tracking_data["stats"]["total_processed"] += 1
        
        # Save tracking data
        self._save_tracking_data()
        
        # Add tracking info to analysis result
        analysis_result["tracking_info"] = doc_info
        
        logger.info(f"Document registered: {file_path_obj.name} (hash: {file_hash[:8]}...)")
        return analysis_result
    
    def handle_duplicate_document(self, file_path: str, existing_doc_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle detection of duplicate document
        
        Args:
            file_path: Path to the duplicate document
            existing_doc_info: Info about the existing document
            
        Returns:
            Document info with duplicate handling
        """
        timestamp = datetime.now().isoformat()
        file_path_obj = Path(file_path)
        
        # Update last accessed time
        existing_doc_info["last_accessed"] = timestamp
        
        # Add to processing history
        self.tracking_data["processing_history"].append({
            "timestamp": timestamp,
            "action": "duplicate_detected",
            "file_hash": existing_doc_info["file_hash"],
            "file_name": file_path_obj.name,
            "file_path": str(file_path_obj.resolve()),
            "original_file": existing_doc_info["original_name"],
            "original_path": existing_doc_info["original_path"]
        })
        
        # Update stats
        self.tracking_data["stats"]["duplicates_detected"] += 1
        
        # Update file path mapping (in case file was moved/renamed)
        self.tracking_data["file_paths"][str(file_path_obj.resolve())] = existing_doc_info["file_hash"]
        
        # Save tracking data
        self._save_tracking_data()
        
        logger.info(f"Duplicate detected: {file_path_obj.name} -> {existing_doc_info['original_name']}")
        
        # Return existing analysis if available
        if existing_doc_info.get("analysis_file") and Path(existing_doc_info["analysis_file"]).exists():
            try:
                with open(existing_doc_info["analysis_file"], 'r', encoding='utf-8') as f:
                    analysis_result = json.load(f)
                analysis_result["tracking_info"] = existing_doc_info
                analysis_result["duplicate_of"] = existing_doc_info["original_name"]
                return analysis_result
            except Exception as e:
                logger.error(f"Error loading existing analysis: {e}")
        
        return {"tracking_info": existing_doc_info, "duplicate_of": existing_doc_info["original_name"]}
    
    def _extract_content_types(self, analysis_result: Dict[str, Any]) -> List[str]:
        """Extract content types from analysis result"""
        content_types = []
        
        if analysis_result.get("pages"):
            content_types.append("text")
        if analysis_result.get("tables"):
            content_types.append("tables")
        if analysis_result.get("images"):
            content_types.append("images")
        if analysis_result.get("handwriting_text"):
            content_types.append("handwriting")
        
        return content_types
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = self.tracking_data["stats"].copy()
        stats.update({
            "unique_documents": len(self.tracking_data["documents"]),
            "total_file_paths": len(self.tracking_data["file_paths"]),
            "recent_activity": self.tracking_data["processing_history"][-10:]  # Last 10 activities
        })
        return stats
    
    def get_document_history(self, file_path: str) -> List[Dict[str, Any]]:
        """Get processing history for a specific document"""
        file_hash = self.calculate_file_hash(file_path)
        if not file_hash:
            return []
        
        return [
            entry for entry in self.tracking_data["processing_history"]
            if entry.get("file_hash") == file_hash
        ]
    
    def list_processed_documents(self, limit: int = None) -> List[Dict[str, Any]]:
        """List all processed documents with their info"""
        documents = list(self.tracking_data["documents"].values())
        
        # Sort by processing timestamp (newest first)
        documents.sort(key=lambda x: x.get("processing_timestamp", ""), reverse=True)
        
        if limit:
            documents = documents[:limit]
        
        return documents
    
    def cleanup_orphaned_files(self) -> Dict[str, int]:
        """Clean up tracking data for files that no longer exist"""
        cleanup_stats = {"removed_documents": 0, "removed_paths": 0}
        
        # Check documents
        orphaned_hashes = []
        for file_hash, doc_info in self.tracking_data["documents"].items():
            if not Path(doc_info["original_path"]).exists():
                # Check if analysis file exists
                analysis_file = doc_info.get("analysis_file")
                if not analysis_file or not Path(analysis_file).exists():
                    orphaned_hashes.append(file_hash)
        
        # Remove orphaned documents
        for file_hash in orphaned_hashes:
            del self.tracking_data["documents"][file_hash]
            cleanup_stats["removed_documents"] += 1
        
        # Check file paths
        orphaned_paths = []
        for file_path, file_hash in self.tracking_data["file_paths"].items():
            if not Path(file_path).exists() and file_hash not in self.tracking_data["documents"]:
                orphaned_paths.append(file_path)
        
        # Remove orphaned paths
        for file_path in orphaned_paths:
            del self.tracking_data["file_paths"][file_path]
            cleanup_stats["removed_paths"] += 1
        
        # Update cleanup timestamp
        self.tracking_data["stats"]["last_cleanup"] = datetime.now().isoformat()
        
        # Save changes
        self._save_tracking_data()
        
        logger.info(f"Cleanup completed: {cleanup_stats}")
        return cleanup_stats

# Global document tracker instance
document_tracker = DocumentTracker()