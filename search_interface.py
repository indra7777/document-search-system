import streamlit as st
import logging
import os
import json
import webbrowser
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import pandas as pd
from PIL import Image

# Import our custom modules
from config import Config
from document_processor import DocumentProcessor
from embedding_service import EmbeddingService
from vector_database import VectorDatabase
from llm_service_cpp import LLMServiceCPP as LLMService
from document_analyzer import DocumentAnalyzer
from service_manager import service_manager
from gpu_optimizer import gpu_optimizer
from document_tracker import document_tracker
from fast_startup import FastStartupConfig, log_fast_startup_status

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource(show_spinner="⚡ Fast startup - Loading essential services...", hash_funcs={Config: lambda x: "config"})
def _initialize_services():
    """Initialize and cache services with fast startup optimization"""
    # Log startup mode
    log_fast_startup_status()
    
    config = Config()
    config.create_directories()
    
    # Always enable GPU optimizations (lightweight)
    gpu_optimizer.enable_memory_efficient_attention()
    gpu_optimizer.setup_mixed_precision()
    
    # Set config for service manager
    service_manager.set_config(config)
    
    # Initialize only essential services for fast startup
    services = {
        'config': config,
        'doc_processor': service_manager.get_service('document_processor', DocumentProcessor, config),
        'embedding_service': service_manager.get_service('embedding_service', EmbeddingService, config),
        'vector_db': service_manager.get_service('vector_database', VectorDatabase, config),
    }
    
    # Always lazy load heavy services for fast startup
    services['doc_analyzer'] = None  # 📊 TrOCR model - loads on document upload
    services['llm_service'] = None   # 🤖 LLM model - loads on first AI response, then cached
    
    # Load vector database index (lightweight)
    services['vector_db'].load_index(config.EMBEDDING_DIMENSION)
    
    # Fast startup complete
    startup_msg = FastStartupConfig.get_startup_message()
    logging.info(f"✅ {startup_msg}")
    
    return services

class DocumentSearchInterface:
    def __init__(self):
        # Get cached services to prevent reloading
        self.services = _initialize_services()
        self.config = self.services['config']
        self.doc_processor = self.services['doc_processor']
        self.embedding_service = self.services['embedding_service']
        self.vector_db = self.services['vector_db']
        self._doc_analyzer = None  # Lazy loading
        self._llm_service = None   # Lazy loading
        
        # Initialize session state
        if 'vector_db_loaded' not in st.session_state:
            st.session_state.vector_db_loaded = True  # Already loaded via cache
        if 'documents_processed' not in st.session_state:
            st.session_state.documents_processed = 0
    
    @property
    def doc_analyzer(self):
        """Lazy load document analyzer only when needed (heavy TrOCR model)"""
        if self._doc_analyzer is None:
            with st.spinner("📊 Loading document analyzer (TrOCR, table extraction)..."):
                self._doc_analyzer = service_manager.get_service('document_analyzer', DocumentAnalyzer, self.config)
        return self._doc_analyzer
    
    @property
    def llm_service(self):
        """Lazy load LLM service with GPU optimization"""
        if self._llm_service is None:
            self._llm_service = self._get_cached_llm_service()
        return self._llm_service
    
    @st.cache_resource(show_spinner="🚀 Loading LLM to GPU (one-time setup)...")
    def _get_cached_llm_service(_self):
        """Cache LLM service globally to prevent reloading"""
        return service_manager.get_service('llm_service', LLMService, _self.config)
    
    def _init_services(self):
        """Legacy method - services now loaded via cache"""
        pass  # Services are now loaded via @st.cache_resource
    
    def run(self):
        """Main application interface"""
        st.set_page_config(
            page_title="Document Search System",
            page_icon="🔍",
            layout="wide"
        )
        
        st.title("🔍 Intelligent Document Search System")
        st.markdown("*Powered by BGE embeddings, FAISS vector search, and local LLM*")
        
        # Sidebar for configuration and data management
        self._render_sidebar()
        
        # Main search interface
        self._render_search_interface()
    
    def _render_sidebar(self):
        """Render sidebar with data management options"""
        st.sidebar.header("📊 Data Management")
        
        # Database and memory stats
        if hasattr(self, 'vector_db'):
            stats = self.vector_db.get_stats()
            st.sidebar.metric("Documents Indexed", stats['total_documents'])
            st.sidebar.metric("Vector Index Size", stats['index_size'])
            
            # GPU memory stats
            memory_info = gpu_optimizer.get_gpu_memory_info()
            if memory_info:
                st.sidebar.metric("GPU Memory (GB)", f"{memory_info['allocated']:.1f}/{memory_info['total']:.1f}")
                
                # Memory warning
                if memory_info['allocated'] > memory_info['total'] * 0.8:
                    st.sidebar.warning("⚠️ High GPU memory usage")
                    if st.sidebar.button("🧹 Clean GPU Memory"):
                        gpu_optimizer.cleanup_gpu_memory(force=True)
                        st.sidebar.success("Memory cleaned!")
        
        # Enhanced file upload section
        st.sidebar.subheader("📁 Enhanced Document Upload")
        uploaded_files = st.sidebar.file_uploader(
            "Upload documents (supports tables, images, handwriting)",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt', 'md', 'png', 'jpg', 'jpeg', 'xlsx', 'xls', 'csv']
        )
        
        # Analysis options
        analyze_tables = st.sidebar.checkbox("Extract tables", value=True)
        analyze_images = st.sidebar.checkbox("Analyze images/handwriting", value=True)
        persist_files = st.sidebar.checkbox("Save files permanently", value=True)
        force_reprocess = st.sidebar.checkbox("🔄 Force reprocess duplicates", value=False, 
                                            help="Reprocess files even if they were already analyzed")
        
        # Process and remove buttons
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if uploaded_files and st.button("🔍 Analyze & Process"):
                self._process_uploaded_files_enhanced(uploaded_files, analyze_tables, analyze_images, persist_files, force_reprocess)
        with col2:
            if st.button("🗑️ Remove Uploaded"):
                self._remove_uploaded_files()
        
        # Show processed files
        if st.sidebar.button("📋 View Processed Files"):
            self._show_processed_files()
        
        # Document tracking dashboard
        if st.sidebar.button("📈 Document Tracking"):
            self._show_document_tracking()
        
        # Directory path input
        st.sidebar.subheader("📂 Process Directory")
        data_path = st.sidebar.text_input("Enter directory path:")
        
        if data_path and st.sidebar.button("Process Directory"):
            self._process_directory(data_path)
        
        # Clear database
        if st.sidebar.button("🗑️ Clear Database"):
            self._clear_database()
        
        # Memory management
        if st.sidebar.button("🧹 Optimize Memory"):
            self._optimize_memory()
        
        # Cleanup orphaned files
        if st.sidebar.button("🧹 Cleanup Tracking"):
            self._cleanup_tracking()
    
    def _render_search_interface(self):
        """Render main search interface"""
        # Search input
        query = st.text_input(
            "🔍 Search your documents:",
            placeholder="Enter your question or search query...",
            key="search_query"
        )
        
        # Search parameters
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            top_k = st.slider("Results to retrieve", 1, 20, 5)
        with col2:
            similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.7, 0.05)
        with col3:
            enable_llm = st.checkbox("Generate AI response", value=False, 
                                   help="🤖 Slower but provides AI summary")
        with col4:
            prioritize_visuals = st.checkbox("Prioritize charts/images", value=False, 
                                           help="📊 Show results with charts and images first")
        
        # Show helpful tip when AI is enabled
        if enable_llm:
            st.info("💡 **Tip:** AI responses use GPU-accelerated LLM for faster generation")
        
        # Search button and results
        if st.button("Search", type="primary") and query:
            self._perform_search(query, top_k, similarity_threshold, enable_llm, prioritize_visuals)
    
    def _perform_search(self, query: str, top_k: int, threshold: float, enable_llm: bool, prioritize_visuals: bool = False):
        """Perform optimized document search with minimal I/O"""
        try:
            # Step 1: Fast embedding generation (with caching)
            with st.spinner("🔍 Generating query embedding..."):
                start_time = time.time()
                query_embedding = self.embedding_service.embed_query(query)
                embedding_time = time.time() - start_time
                
            # Step 2: Fast vector search (GPU-accelerated FAISS)
            with st.spinner("⚡ Searching vector database..."):
                search_start = time.time()
                search_results = self.vector_db.search(
                    query_embedding, 
                    top_k=top_k, 
                    threshold=threshold
                )
                search_time = time.time() - search_start
                
                if not search_results:
                    st.warning("No relevant documents found. Try adjusting the similarity threshold.")
                    return
                
            # Step 3: Show results immediately with performance stats
            total_search_time = embedding_time + search_time
            
            # Performance feedback for user
            if total_search_time < 0.5:
                st.success(f"⚡ Found {len(search_results)} documents in {total_search_time:.3f}s (Lightning fast!)")
            elif total_search_time < 2.0:
                st.success(f"🚀 Found {len(search_results)} documents in {total_search_time:.2f}s (Fast)")
            else:
                st.success(f"✅ Found {len(search_results)} documents in {total_search_time:.2f}s")
            
            # Detailed timing in expander
            with st.expander("⏱️ Performance Details"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Query Embedding", f"{embedding_time:.3f}s")
                with col2:
                    st.metric("Vector Search", f"{search_time:.3f}s")
                with col3:
                    cache_status = "🎯 Cache Hit" if embedding_time < 0.01 else "💫 GPU Computed"
                    st.metric("Cache Status", cache_status)
            
            # Step 4: Generate LLM response first (if enabled) - users want AI answer first
            if enable_llm:
                # Show helpful loading message based on whether LLM is cached
                if self._llm_service is None:
                    loading_msg = "🚀 Loading LLM to GPU (first time) + generating response..."
                else:
                    loading_msg = "🤖 Generating AI response..."
                
                with st.spinner(loading_msg):
                    llm_start = time.time()
                    llm_response = self.llm_service.generate_response(query, search_results)
                    llm_time = time.time() - llm_start
                    
                    # Add timing info to response
                    llm_response['generation_time'] = llm_time
                    
                    # Display AI response at the top
                    self._display_llm_response(llm_response)
            
            # Step 5: Visual content sorting (optimized - no file I/O during search)
            if prioritize_visuals:
                with st.spinner("📊 Prioritizing visual content..."):
                    search_results = self._sort_results_by_visual_content_fast(search_results)
            
            # Step 6: Display source documents after AI response
            self._display_search_results(search_results)
                
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            logger.error(f"Search error: {str(e)}")
    
    def _display_llm_response(self, response: Dict[str, Any]):
        """Display LLM generated response"""
        st.subheader("🤖 AI Response")
        
        with st.container():
            st.markdown(response['answer'])
            
            # Display response metadata with performance info
            with st.expander("🔍 Response Details"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Model Used", response['model'])
                    st.metric("Context Documents", response['context_used'])
                with col2:
                    if 'generation_time' in response:
                        gen_time = response['generation_time']
                        if gen_time < 2:
                            st.metric("Generation Time", f"{gen_time:.2f}s ⚡")
                        elif gen_time < 5:
                            st.metric("Generation Time", f"{gen_time:.2f}s 🚀")
                        else:
                            st.metric("Generation Time", f"{gen_time:.2f}s")
                    
                    if 'tokens_generated' in response:
                        st.metric("Tokens Generated", response['tokens_generated'])
                
                with col3:
                    st.write("**Sources Used:**")
                    for i, source in enumerate(response['sources'], 1):
                        st.write(f"{i}. {source['file_name']} (similarity: {source['similarity_score']:.3f})")
    
    def _display_search_results(self, results: List[Dict[str, Any]]):
        """Display enhanced search results with source links and analysis info"""
        st.subheader("📋 Document Sources")
        
        for i, result in enumerate(results, 1):
            # Try to load analysis data for enhanced display
            analysis_data = self._load_analysis_for_file(result.get('file_path', ''))
            
            # Create visual indicators
            visual_indicators = []
            if analysis_data:
                if analysis_data.get('tables'):
                    visual_indicators.append("🗃️")
                if analysis_data.get('images'):
                    visual_indicators.append("🖼️")
                if analysis_data.get('handwriting_text'):
                    visual_indicators.append("✍️")
            
            # Show visual score if prioritization was used
            visual_score_text = ""
            if result.get('visual_score', 0) > 0:
                visual_score_text = f" [Visual: {result['visual_score']}]"
            
            indicator_text = " ".join(visual_indicators)
            title = f"{indicator_text} 📄 {result['file_name']} (Similarity: {result['similarity_score']:.3f}){visual_score_text}"
            
            with st.expander(title):
                
                # Enhanced document metadata
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.write(f"**File:** {result['file_name']}")
                    st.write(f"**Type:** {result.get('file_type', 'Unknown')}")
                with col2:
                    st.write(f"**Similarity:** {result['similarity_score']:.3f}")
                    st.write(f"**Rank:** {result.get('rank', i)}")
                with col3:
                    if analysis_data:
                        summary = self.doc_analyzer.get_document_summary(analysis_data)
                        st.write(f"**Pages:** {summary['total_pages']}")
                        st.write(f"**Tables:** {summary['total_tables']}")
                with col4:
                    # Enhanced source options
                    if st.button(f"📎 Open Source", key=f"source_{i}"):
                        self._open_source_file(result['file_path'])
                    if analysis_data and st.button(f"🔍 View Analysis", key=f"analysis_{i}"):
                        self._display_analysis_results([analysis_data])
                
                # Show content types and visual sources
                if analysis_data:
                    summary = self.doc_analyzer.get_document_summary(analysis_data)
                    if summary['content_types']:
                        # Add visual indicators to content types
                        content_with_icons = []
                        for content_type in summary['content_types']:
                            if content_type == 'tables':
                                content_with_icons.append('🗃️ tables')
                            elif content_type == 'images':
                                content_with_icons.append('🖼️ images')
                            elif content_type == 'handwriting':
                                content_with_icons.append('✍️ handwriting')
                            else:
                                content_with_icons.append(content_type)
                        
                        st.info(f"📊 Content types: {', '.join(content_with_icons)}")
                    
                    # Show relevant charts/images for this search result
                    self._display_visual_sources(analysis_data, result.get('content', ''))
                
                # Document content preview
                st.write("**Content Preview:**")
                content = result.get('content', '')
                if len(content) > 500:
                    content = content[:500] + "..."
                st.text_area("Content Preview", content, height=100, key=f"content_{i}", label_visibility="collapsed")
                
                # Show page/chunk information if available
                if result.get('page_number'):
                    st.caption(f"Source: Page {result['page_number']}, Chunk {result.get('chunk_id', 'N/A')}")
                
                # Quick access to related visual content - show inline without button
                if analysis_data and (analysis_data.get('images') or analysis_data.get('tables')):
                    with st.expander(f"🖼️ View Charts & Images from this Document"):
                        self._display_document_visuals_inline(analysis_data)
    
    def _load_analysis_for_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load analysis data for a specific file if it exists"""
        try:
            if not file_path:
                return None
            
            file_path = Path(file_path)
            analysis_file = self.doc_analyzer.processed_dir / f"{file_path.stem}_analysis.json"
            
            if analysis_file.exists():
                with open(analysis_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.debug(f"No analysis data found for {file_path}: {e}")
        
        return None
    
    def _open_source_file(self, file_path: str):
        """Display source file with download and preview options"""
        try:
            if os.path.exists(file_path):
                abs_path = os.path.abspath(file_path)
                file_name = os.path.basename(file_path)
                
                # Read PDF file for download
                with open(file_path, 'rb') as f:
                    pdf_bytes = f.read()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Download button - works everywhere
                    st.download_button(
                        label=f"📄 Download {file_name}",
                        data=pdf_bytes,
                        file_name=file_name,
                        mime="application/pdf",
                        help="Click to download the PDF file"
                    )
                
                with col2:
                    # Copy path button
                    if st.button(f"📋 Copy Path", key=f"copy_{file_name}"):
                        st.code(abs_path, language="text")
                        st.success("Path copied! You can paste it in your file explorer.")
                
                # File information
                file_size = len(pdf_bytes) / 1024 / 1024  # MB
                st.info(f"📁 **File**: {file_name} ({file_size:.1f} MB)\n📍 **Location**: `{abs_path}`")
                
                # PDF preview (first page only)
                try:
                    # Create an embedded PDF viewer that works in browsers
                    import base64
                    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                    
                    pdf_display = f'''
                    <iframe 
                        src="data:application/pdf;base64,{base64_pdf}" 
                        width="100%" 
                        height="400" 
                        style="border: 1px solid #ddd; border-radius: 5px;">
                    </iframe>
                    '''
                    
                    with st.expander("🔍 Preview PDF (Click to expand)"):
                        st.markdown(pdf_display, unsafe_allow_html=True)
                        st.caption("PDF preview - Download for full functionality")
                        
                except Exception as preview_error:
                    st.warning("Preview not available - use download button to view the full PDF")
                
            else:
                st.error("Source file not found", icon="❌")
        except Exception as e:
            st.error(f"Cannot access file: {str(e)}", icon="⚠️")
    
    def _process_uploaded_files_enhanced(self, uploaded_files, analyze_tables=True, analyze_images=True, persist_files=True, force_reprocess=False):
        """Enhanced processing with automatic duplicate detection and removal"""
        try:
            # Step 1: Detect and remove duplicates automatically
            unique_files = self._detect_and_remove_duplicates(uploaded_files)
            
            if len(unique_files) < len(uploaded_files):
                st.info(f"🔄 Removed {len(uploaded_files) - len(unique_files)} duplicate files automatically")
            
            if not unique_files:
                st.warning("⚠️ All uploaded files were duplicates. No new files to process.")
                return
            
            analysis_results = []
            
            with st.spinner("🔍 Analyzing unique files (tables, images, handwriting)..."):
                progress_bar = st.progress(0)
                
                for i, uploaded_file in enumerate(unique_files):
                    # Save file permanently if requested
                    if persist_files:
                        file_path = self.doc_analyzer.save_uploaded_file(uploaded_file)
                    else:
                        # Save temporarily
                        temp_path = self.config.DATA_DIR / uploaded_file.name
                        with open(temp_path, 'wb') as f:
                            f.write(uploaded_file.getbuffer())
                        file_path = str(temp_path)
                    
                    # Perform comprehensive analysis
                    analysis_result = self.doc_analyzer.analyze_document(file_path)
                    
                    # Check if this is a duplicate from previous sessions
                    if 'duplicate_of' in analysis_result and not force_reprocess:
                        st.info(f"⏭️ Skipping {uploaded_file.name} - already processed as {analysis_result['duplicate_of']}")
                        continue
                        
                    analysis_results.append(analysis_result)
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(unique_files))
                
                # Display analysis results
                if analysis_results:
                    self._display_analysis_results(analysis_results)
                
                # Step 2: Ensure all files are preprocessed and loaded into vector database
                self._ensure_vector_database_processing(analysis_results)
                
            st.success(f"✅ Successfully processed {len(analysis_results)} new files into vector database!")
            
        except Exception as e:
            st.error(f"❌ Error processing files: {str(e)}")
            logger.error(f"Enhanced file processing error: {str(e)}")
    
    def _detect_and_remove_duplicates(self, uploaded_files):
        """Detect and automatically remove duplicate files from upload batch"""
        unique_files = []
        seen_hashes = set()
        
        for uploaded_file in uploaded_files:
            # Calculate file hash
            uploaded_file.seek(0)  # Reset file pointer
            file_content = uploaded_file.read()
            file_hash = hashlib.md5(file_content).hexdigest()
            uploaded_file.seek(0)  # Reset again
            
            if file_hash not in seen_hashes:
                seen_hashes.add(file_hash)
                unique_files.append(uploaded_file)
            else:
                logger.info(f"Detected duplicate file: {uploaded_file.name}")
        
        return unique_files
    
    def _ensure_vector_database_processing(self, analysis_results):
        """Ensure all analyzed files are properly processed into vector database"""
        if not analysis_results:
            return
            
        # Extract file paths that need vector processing
        file_paths = []
        for result in analysis_results:
            if "file_path" in result and 'duplicate_of' not in result:
                file_paths.append(result["file_path"])
        
        if file_paths:
            with st.spinner("🚀 Loading documents into GPU-accelerated vector database..."):
                # Monitor GPU memory before processing
                gpu_optimizer.monitor_memory_usage()
                
                # Process documents into vector database
                self._process_documents(file_paths)
                
                # Ensure FAISS index is loaded on GPU
                self._ensure_gpu_vector_index()
                
                # Cleanup after processing
                gpu_optimizer.cleanup_gpu_memory()
                
                st.success(f"🎯 Loaded {len(file_paths)} documents into GPU vector database")
    
    def _ensure_gpu_vector_index(self):
        """Ensure FAISS vector index is properly loaded on GPU"""
        try:
            if self.vector_db.index is not None and hasattr(self.vector_db, '_gpu_enabled'):
                if not self.vector_db._gpu_enabled:
                    self.vector_db.move_to_gpu()
                    st.info("📍 Vector index moved to GPU for faster search")
        except Exception as e:
            logger.warning(f"Could not ensure GPU vector index: {e}")
    
    def _process_uploaded_files(self, uploaded_files):
        """Legacy method - calls enhanced version"""
        self._process_uploaded_files_enhanced(uploaded_files, True, True, True)
    
    def _process_directory(self, directory_path: str):
        """Process documents from directory"""
        try:
            if not os.path.exists(directory_path):
                st.error("Directory does not exist")
                return
            
            with st.spinner("Processing directory..."):
                self._process_documents([directory_path])
                
            st.success("Directory processed successfully!")
            
        except Exception as e:
            st.error(f"Error processing directory: {str(e)}")
    
    def _process_documents(self, paths: List[str]):
        """Process documents and add to vector database"""
        # Ensure vector database is properly initialized
        if self.vector_db is None:
            st.error("Vector database not initialized")
            return
            
        # Process documents
        processed_docs = self.doc_processor.process_documents(paths)
        
        if not processed_docs:
            st.warning("No documents were processed")
            return
        
        # Chunk documents
        all_chunks = []
        for doc in processed_docs:
            chunks = self.doc_processor.chunk_document(doc)
            all_chunks.extend(chunks)
        
        if not all_chunks:
            st.warning("No document chunks created")
            return
        
        # Generate embeddings
        embedded_chunks = self.embedding_service.embed_documents(all_chunks)
        
        if not embedded_chunks:
            st.error("Failed to generate embeddings")
            return
        
        # Add to vector database
        try:
            self.vector_db.add_documents(embedded_chunks)
            # Save database
            self.vector_db.save_index()
            # Update session state
            st.session_state.documents_processed += len(processed_docs)
            st.session_state.vector_db_loaded = True
            st.success(f"✅ Added {len(embedded_chunks)} chunks to vector database")
        except Exception as e:
            st.error(f"❌ Failed to add documents to vector database: {str(e)}")
            logger.error(f"Vector database error: {str(e)}")
    
    def _clear_database(self):
        """Clear the vector database"""
        try:
            self.vector_db.delete_index()
            self.vector_db = VectorDatabase(self.config)
            st.session_state.documents_processed = 0
            st.session_state.vector_db_loaded = False
            
            # Clean up GPU memory after clearing database
            gpu_optimizer.cleanup_gpu_memory(force=True)
            
            st.success("Database cleared successfully!")
        except Exception as e:
            st.error(f"Error clearing database: {str(e)}")
    
    def _optimize_memory(self):
        """Optimize system memory usage"""
        try:
            with st.spinner("Optimizing memory..."):
                # Get memory stats before
                before_stats = gpu_optimizer.get_gpu_memory_info()
                
                # Perform cleanup
                gpu_optimizer.cleanup_gpu_memory(force=True)
                
                # Get memory stats after
                after_stats = gpu_optimizer.get_gpu_memory_info()
                
                if before_stats and after_stats:
                    freed = before_stats['allocated'] - after_stats['allocated']
                    st.success(f"Memory optimized! Freed {freed:.2f}GB GPU memory")
                    
                    # Show detailed stats
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Before", f"{before_stats['allocated']:.2f}GB")
                    with col2:
                        st.metric("After", f"{after_stats['allocated']:.2f}GB")
                else:
                    st.success("Memory optimization completed!")
                    
        except Exception as e:
            st.error(f"Error during memory optimization: {str(e)}")
    
    def _display_analysis_results(self, analysis_results: List[Dict[str, Any]]):
        """Display comprehensive analysis results"""
        st.subheader("📋 Document Analysis Results")
        
        for result in analysis_results:
            if "error" in result:
                st.error(f"❌ Analysis failed for {result.get('file_path', 'unknown')}: {result['error']}")
                continue
            
            summary = self.doc_analyzer.get_document_summary(result)
            
            with st.expander(f"📄 {summary['file_name']} - Analysis Report", expanded=True):
                # Check if this is a duplicate or new document
                tracking_info = result.get('tracking_info')
                is_duplicate = 'duplicate_of' in result
                
                if is_duplicate:
                    if result.get('force_reprocess_requested'):
                        st.info(f"🔄 **Force Reprocess Requested** but file is identical to: {result['duplicate_of']}")
                        st.warning("⚠️ Document content is identical - no new analysis needed.")
                    else:
                        st.warning(f"🔁 **Duplicate Document Detected!** This file is identical to: {result['duplicate_of']}")
                        st.info("💡 **Tip:** Use 🔄 'Force reprocess duplicates' checkbox or 🗑️ Remove Uploaded button to clear old files.")
                elif tracking_info and not result.get('is_new_document', True):
                    st.info(f"📄 **Document Previously Processed** on {tracking_info.get('processing_timestamp', 'N/A')[:19]}")
                else:
                    st.success("✨ **New Document Analyzed**")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📄 Pages", summary['total_pages'])
                with col2:
                    st.metric("🗃️ Tables", summary['total_tables'])
                with col3:
                    st.metric("🖼️ Images", summary['total_images'])
                with col4:
                    st.metric("✍️ Handwriting", "Yes" if summary['has_handwriting'] else "No")
                
                # Show tracking information if available
                if tracking_info:
                    with st.expander("📈 Document Tracking Info"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**File Hash:** {tracking_info.get('file_hash', 'N/A')[:16]}...")
                            st.write(f"**Processing Time:** {tracking_info.get('processing_timestamp', 'N/A')[:19]}")
                            st.write(f"**File Size:** {tracking_info.get('file_size', 0) / 1024:.1f} KB")
                        with col2:
                            st.write(f"**Vector DB Indexed:** {'Yes' if tracking_info.get('vector_db_indexed') else 'No'}")
                            st.write(f"**Last Accessed:** {tracking_info.get('last_accessed', 'N/A')[:19]}")
                            st.write(f"**Processing Version:** {tracking_info.get('processing_version', 'N/A')}")
                
                # Content types
                if summary['content_types']:
                    st.write(f"**Content Types Found:** {', '.join(summary['content_types'])}")
                
                # Tables section
                if result.get('tables'):
                    st.write("### 🗃️ Extracted Tables")
                    for i, table in enumerate(result['tables']):
                        st.write(f"**Table {i+1}** (Page {table.get('page', 'N/A')}) - Method: {table.get('method', 'unknown')}")
                        
                        # Show table data
                        if 'data' in table and table['data']:
                            df = pd.DataFrame(table['data'])
                            st.dataframe(df.head(10))  # Show first 10 rows
                            
                            # Download link for CSV
                            if 'csv_path' in table:
                                with open(table['csv_path'], 'rb') as f:
                                    st.download_button(
                                        f"💾 Download Table {i+1} CSV",
                                        f.read(),
                                        file_name=f"table_{i+1}.csv",
                                        mime="text/csv",
                                        key=f"analysis_table_{result['file_name']}_{i}_{int(time.time())}"
                                    )
                
                # Handwriting section
                if result.get('handwriting_text'):
                    st.write("### ✍️ Handwriting Recognition")
                    for hw in result['handwriting_text']:
                        st.write(f"**Page {hw['page']}** (Confidence: {hw.get('confidence', 0):.1f}%)")
                        st.text_area("Recognized Text:", hw['text'], height=100, key=f"hw_{result['file_name']}_{hw['page']}")
                
                # Images section
                if result.get('images'):
                    st.write("### 🖼️ Extracted Images")
                    cols = st.columns(min(3, len(result['images'])))
                    for i, img in enumerate(result['images'][:6]):  # Show first 6 images
                        with cols[i % 3]:
                            if os.path.exists(img['path']):
                                image = Image.open(img['path'])
                                st.image(image, caption=f"Page {img['page']}", use_container_width=True)
                                
                                # Show OCR results if available
                                analysis = img.get('analysis', {})
                                if analysis.get('ocr_text') or analysis.get('handwriting_text'):
                                    with st.expander(f"Text in Image {i+1}"):
                                        if analysis.get('ocr_text'):
                                            st.write("**OCR Text:**")
                                            st.write(analysis['ocr_text'])
                                        if analysis.get('handwriting_text'):
                                            st.write("**Handwriting:**")
                                            st.write(analysis['handwriting_text'])
                
                # Processing errors
                if result.get('processing_errors'):
                    st.warning(f"⚠️ {len(result['processing_errors'])} processing warning(s)")
                    with st.expander("Show warnings"):
                        for error in result['processing_errors']:
                            st.write(f"- {error}")
                
                # Skip detailed analysis display for duplicates (already processed)
                if is_duplicate:
                    st.info("🔁 Detailed analysis skipped for duplicate document. See original analysis above.")
                    continue
    
    def _show_processed_files(self):
        """Show list of previously processed files"""
        try:
            processed_files = self.doc_analyzer.list_processed_files()
            
            if not processed_files:
                st.info("📁 No processed files found. Upload and analyze some documents first!")
                return
            
            st.subheader(f"📁 Processed Files ({len(processed_files)})")
            
            for file_info in processed_files:
                with st.expander(f"📄 {file_info['file_name']} ({file_info['file_type']})"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Pages", file_info['total_pages'])
                        st.metric("Tables", file_info['total_tables'])
                    with col2:
                        st.metric("Images", file_info['total_images'])
                        st.metric("Handwriting", "Yes" if file_info['has_handwriting'] else "No")
                    with col3:
                        st.metric("Errors", file_info['processing_errors'])
                        if file_info['content_types']:
                            st.write(f"**Types:** {', '.join(file_info['content_types'])}")
                    
                    # Load and display full analysis
                    if st.button(f"Load Full Analysis", key=f"load_{file_info['file_name']}"):
                        try:
                            with open(file_info['analysis_file'], 'r') as f:
                                analysis_result = json.load(f)
                            self._display_analysis_results([analysis_result])
                        except Exception as e:
                            st.error(f"Error loading analysis: {e}")
        
        except Exception as e:
            st.error(f"Error listing processed files: {e}")
            logger.error(f"Error listing processed files: {e}")
    
    def _show_document_tracking(self):
        """Show document tracking dashboard"""
        try:
            st.subheader("📈 Document Tracking Dashboard")
            
            # Get tracking stats
            stats = document_tracker.get_processing_stats()
            
            # Display stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📄 Total Processed", stats['total_processed'])
            with col2:
                st.metric("🔍 Unique Documents", stats['unique_documents'])
            with col3:
                st.metric("🔁 Duplicates Detected", stats['duplicates_detected'])
            with col4:
                st.metric("📁 File Paths Tracked", stats['total_file_paths'])
            
            # Show recent activity
            if stats.get('recent_activity'):
                st.subheader("🕰️ Recent Activity")
                activity_df = pd.DataFrame(stats['recent_activity'])
                activity_df['timestamp'] = pd.to_datetime(activity_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                st.dataframe(activity_df[['timestamp', 'action', 'file_name']], use_container_width=True)
            
            # List all tracked documents
            st.subheader("📄 Tracked Documents")
            documents = document_tracker.list_processed_documents(limit=50)
            
            if documents:
                for i, doc in enumerate(documents):
                    with st.expander(f"📄 {doc['original_name']} ({doc['processing_timestamp'][:19]})"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**File Hash:** {doc['file_hash'][:16]}...")
                            st.write(f"**Size:** {doc['file_size'] / 1024:.1f} KB")
                            st.write(f"**Last Accessed:** {doc['last_accessed'][:19]}")
                        
                        with col2:
                            summary = doc['content_summary']
                            st.write(f"**Pages:** {summary['total_pages']}")
                            st.write(f"**Tables:** {summary['total_tables']}")
                            st.write(f"**Images:** {summary['total_images']}")
                            st.write(f"**Content Types:** {', '.join(summary['content_types'])}")
                        
                        # Show processing history for this document
                        if st.button(f"Show History", key=f"history_{i}"):
                            history = document_tracker.get_document_history(doc['original_path'])
                            if history:
                                st.write("**Processing History:**")
                                for entry in history:
                                    st.write(f"- {entry['timestamp'][:19]}: {entry['action']} ({entry.get('file_name', 'N/A')})")
            else:
                st.info("📁 No documents tracked yet. Upload and analyze some documents first!")
                
        except Exception as e:
            st.error(f"Error showing document tracking: {e}")
            logger.error(f"Error showing document tracking: {e}")
    
    def _cleanup_tracking(self):
        """Clean up orphaned tracking data"""
        try:
            with st.spinner("Cleaning up tracking data..."):
                cleanup_stats = document_tracker.cleanup_orphaned_files()
                
            st.success(
                f"🧹 Cleanup completed!\n"
                f"Removed {cleanup_stats['removed_documents']} orphaned documents\n"
                f"Removed {cleanup_stats['removed_paths']} orphaned file paths"
            )
            
        except Exception as e:
            st.error(f"Error during cleanup: {e}")
    
    def _remove_uploaded_files(self):
        """Remove uploaded files from data directory and processed files"""
        try:
            with st.spinner("🗑️ Removing uploaded files..."):
                removed_count = 0
                
                # Remove files from uploads directory
                uploads_dir = self.config.DATA_DIR / "uploads"
                if uploads_dir.exists():
                    for file in uploads_dir.iterdir():
                        if file.is_file():
                            try:
                                file.unlink()
                                removed_count += 1
                            except Exception as e:
                                logger.warning(f"Could not remove {file}: {e}")
                
                # Remove files from data directory (temporary uploads)
                for file in self.config.DATA_DIR.iterdir():
                    if file.is_file() and file.suffix.lower() in ['.pdf', '.docx', '.txt', '.md', '.png', '.jpg', '.jpeg']:
                        try:
                            file.unlink()
                            removed_count += 1
                        except Exception as e:
                            logger.warning(f"Could not remove {file}: {e}")
                
                # Clean up processed files directory
                processed_dir = Path(self.config.BASE_DIR) / "processed_files"
                if processed_dir.exists():
                    for file in processed_dir.iterdir():
                        if file.is_file():
                            try:
                                file.unlink()
                                removed_count += 1
                            except Exception as e:
                                logger.warning(f"Could not remove processed file {file}: {e}")
                
                # Clear session state
                if 'uploaded_files' in st.session_state:
                    del st.session_state['uploaded_files']
                
                st.success(f"✅ Successfully removed {removed_count} files!")
                st.rerun()  # Refresh the page to update file list
                
        except Exception as e:
            st.error(f"❌ Error removing files: {str(e)}")
            logger.error(f"File removal error: {str(e)}")
    
    def _display_visual_sources(self, analysis_data: Dict[str, Any], search_content: str):
        """Display relevant charts and images as sources for the search result"""
        try:
            # Check if search content relates to visual elements
            visual_keywords = ['chart', 'graph', 'table', 'figure', 'diagram', 'image', 'data', 'trend', 'analysis']
            content_lower = search_content.lower()
            
            has_visual_relevance = any(keyword in content_lower for keyword in visual_keywords)
            
            if has_visual_relevance and (analysis_data.get('images') or analysis_data.get('tables')):
                with st.expander("🖼️ Visual Sources Related to This Result"):
                    
                    # Show related tables
                    tables = analysis_data.get('tables', [])
                    if tables:
                        st.write("**🗃️ Related Tables:**")
                        for i, table in enumerate(tables[:2]):  # Show first 2 tables
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"Table {i+1} from Page {table.get('page', 'N/A')} ({table.get('method', 'unknown')} extraction)")
                                if table.get('data') and len(table['data']) > 0:
                                    df = pd.DataFrame(table['data'])
                                    st.dataframe(df.head(3), use_container_width=True)  # Preview
                            with col2:
                                if table.get('csv_path') and Path(table['csv_path']).exists():
                                    with open(table['csv_path'], 'rb') as f:
                                        st.download_button(
                                            f"💾 CSV",
                                            f.read(),
                                            file_name=f"table_{i+1}.csv",
                                            mime="text/csv",
                                            key=f"visual_table_{analysis_data.get('file_name', 'unknown')}_{i}_{int(time.time())}"
                                        )
                    
                    # Show related images with chart analysis
                    images = analysis_data.get('images', [])
                    if images:
                        st.write("**🖼️ Related Charts & Images:**")
                        cols = st.columns(min(3, len(images)))
                        for i, img in enumerate(images[:3]):  # Show first 3 images
                            with cols[i % 3]:
                                if Path(img['path']).exists():
                                    image = Image.open(img['path'])
                                    st.image(image, caption=f"Page {img['page']}", use_container_width=True)
                                    
                                    # Show chart analysis if available
                                    img_analysis = img.get('analysis', {})
                                    chart_analysis = img_analysis.get('chart_analysis', {})
                                    
                                    if chart_analysis.get('chart_type'):
                                        st.caption(f"📈 {chart_analysis['chart_type'].replace('_', ' ').title()}")
                                        
                                        # Show insights
                                        insights = chart_analysis.get('insights', [])
                                        if insights:
                                            for insight in insights[:2]:  # Show first 2 insights
                                                st.caption(f"• {insight}")
        
        except Exception as e:
            logger.error(f"Error displaying visual sources: {e}")
    
    def _display_document_visuals(self, analysis_data: Dict[str, Any]):
        """Display all charts and images from a document"""
        try:
            st.subheader(f"🖼️ Visual Content from {analysis_data.get('file_name', 'Document')}")
            
            # Display all tables
            tables = analysis_data.get('tables', [])
            if tables:
                st.write("### 🗃️ Extracted Tables")
                for i, table in enumerate(tables):
                    with st.expander(f"Table {i+1} - Page {table.get('page', 'N/A')} ({table.get('method', 'unknown')})"):
                        if table.get('data'):
                            df = pd.DataFrame(table['data'])
                            st.dataframe(df, use_container_width=True)
                            
                            # Download option
                            if table.get('csv_path') and Path(table['csv_path']).exists():
                                with open(table['csv_path'], 'rb') as f:
                                    st.download_button(
                                        f"💾 Download Table {i+1} as CSV",
                                        f.read(),
                                        file_name=f"table_{i+1}.csv",
                                        mime="text/csv",
                                        key=f"full_table_{analysis_data.get('file_name', 'doc')}_{i}_{int(time.time())}"
                                    )
            
            # Display all images with analysis
            images = analysis_data.get('images', [])
            if images:
                st.write("### 🖼️ Charts & Images")
                
                for i, img in enumerate(images):
                    with st.expander(f"Image {i+1} - Page {img['page']}", expanded=True):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            if Path(img['path']).exists():
                                image = Image.open(img['path'])
                                st.image(image, use_container_width=True)
                        
                        with col2:
                            img_analysis = img.get('analysis', {})
                            chart_analysis = img_analysis.get('chart_analysis', {})
                            
                            # Chart type and basic info
                            if chart_analysis.get('chart_type'):
                                st.write(f"**Type:** {chart_analysis['chart_type'].replace('_', ' ').title()}")
                            
                            # OCR text
                            ocr_text = img_analysis.get('ocr_text', '')
                            if ocr_text:
                                st.write("**Text in Image:**")
                                st.text_area("", ocr_text[:200] + ("..." if len(ocr_text) > 200 else ""), 
                                           height=80, key=f"ocr_{i}", label_visibility="collapsed")
                            
                            # Handwriting
                            handwriting = img_analysis.get('handwriting_text', '')
                            if handwriting:
                                st.write("**Handwriting:**")
                                st.text_area("", handwriting[:100] + ("..." if len(handwriting) > 100 else ""), 
                                           height=60, key=f"hw_{i}", label_visibility="collapsed")
                            
                            # Chart insights
                            insights = chart_analysis.get('insights', [])
                            if insights:
                                st.write("**Insights:**")
                                for insight in insights:
                                    st.write(f"• {insight}")
                            
                            # Visual elements
                            visual_elements = chart_analysis.get('visual_elements', {})
                            if visual_elements.get('dominant_colors'):
                                st.write("**Colors:**")
                                colors = visual_elements['dominant_colors'][:3]
                                for color in colors:
                                    rgb = color['color_rgb']
                                    st.write(f"• RGB({rgb[0]}, {rgb[1]}, {rgb[2]}) - {color['percentage']:.1f}%")
        
        except Exception as e:
            st.error(f"Error displaying document visuals: {e}")
            logger.error(f"Error displaying document visuals: {e}")
    
    def _display_document_visuals_inline(self, analysis_data: Dict[str, Any]):
        """Display charts and images inline without causing page refresh"""
        try:
            # Display tables inline
            tables = analysis_data.get('tables', [])
            if tables:
                st.write("**🗃️ Tables in this Document:**")
                for i, table in enumerate(tables[:2]):  # Show first 2 tables
                    st.write(f"Table {i+1} - Page {table.get('page', 'N/A')}")
                    if table.get('data') and len(table['data']) > 0:
                        df = pd.DataFrame(table['data'])
                        st.dataframe(df.head(5), use_container_width=True)
                        
                        # Download button with unique key
                        if table.get('csv_path') and Path(table['csv_path']).exists():
                            with open(table['csv_path'], 'rb') as f:
                                st.download_button(
                                    f"💾 Download Table {i+1}",
                                    f.read(),
                                    file_name=f"table_{i+1}.csv",
                                    mime="text/csv",
                                    key=f"inline_table_{analysis_data.get('file_name', 'doc')}_{i}_{int(time.time())}"
                                )
            
            # Display images inline
            images = analysis_data.get('images', [])
            if images:
                st.write("**🖼️ Charts & Images in this Document:**")
                
                # Show images in a grid layout
                num_cols = min(3, len(images))
                if num_cols > 0:
                    cols = st.columns(num_cols)
                    
                    for i, img in enumerate(images[:6]):  # Show first 6 images
                        with cols[i % num_cols]:
                            if Path(img['path']).exists():
                                image = Image.open(img['path'])
                                st.image(image, caption=f"Page {img['page']}", use_container_width=True)
                                
                                # Show chart analysis if available
                                img_analysis = img.get('analysis', {})
                                chart_analysis = img_analysis.get('chart_analysis', {})
                                
                                if chart_analysis.get('chart_type'):
                                    st.caption(f"📈 {chart_analysis['chart_type'].replace('_', ' ').title()}")
                                
                                # Show brief OCR text if available
                                ocr_text = img_analysis.get('ocr_text', '')
                                if ocr_text and len(ocr_text.strip()) > 0:
                                    preview_text = ocr_text[:50] + ("..." if len(ocr_text) > 50 else "")
                                    st.caption(f"📝 {preview_text}")
                                
                                # Show insights
                                insights = chart_analysis.get('insights', [])
                                if insights:
                                    st.caption(f"💡 {insights[0][:50]}{'...' if len(insights[0]) > 50 else ''}")
            
            # Show handwriting if available
            handwriting_text = analysis_data.get('handwriting_text', [])
            if handwriting_text:
                st.write("**✍️ Handwriting Found:**")
                for hw in handwriting_text[:2]:  # Show first 2
                    st.caption(f"Page {hw['page']}: {hw['text'][:100]}{'...' if len(hw['text']) > 100 else ''}")
        
        except Exception as e:
            st.error(f"Error displaying inline visuals: {e}")
            logger.error(f"Error displaying inline visuals: {e}")
    
    def _sort_results_by_visual_content_fast(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fast visual content sorting without file I/O during search"""
        try:
            # Use content keywords to estimate visual richness (no file reading)
            for result in results:
                content = result.get('content', '').lower()
                visual_keywords = ['chart', 'graph', 'table', 'figure', 'diagram', 'data', 'trend']
                
                # Simple keyword-based scoring (fast)
                visual_score = sum(1 for keyword in visual_keywords if keyword in content)
                result['visual_score'] = visual_score
            
            # Sort by visual score, then similarity
            sorted_results = sorted(results, key=lambda x: (x.get('visual_score', 0), x['similarity_score']), reverse=True)
            
            visual_count = sum(1 for r in sorted_results if r.get('visual_score', 0) > 0)
            logger.info(f"Fast visual sorting: {visual_count} results with visual keywords")
            
            return sorted_results
            
        except Exception as e:
            logger.error(f"Error in fast visual sorting: {e}")
            return results  # Return original results if sorting fails
    
    def _sort_results_by_visual_content(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort search results to prioritize those with visual content (charts, images, tables)"""
        try:
            def visual_content_score(result):
                """Calculate visual content score for a search result"""
                score = 0
                
                # Try to load analysis data for this result
                analysis_data = self._load_analysis_for_file(result.get('file_path', ''))
                
                if analysis_data:
                    # Points for tables
                    tables = analysis_data.get('tables', [])
                    score += len(tables) * 3  # 3 points per table
                    
                    # Points for images/charts
                    images = analysis_data.get('images', [])
                    for img in images:
                        score += 2  # 2 points per image
                        
                        # Bonus points for charts
                        img_analysis = img.get('analysis', {})
                        chart_analysis = img_analysis.get('chart_analysis', {})
                        if chart_analysis.get('chart_type'):
                            score += 5  # 5 bonus points for identified charts
                    
                    # Points for handwriting
                    handwriting = analysis_data.get('handwriting_text', [])
                    score += len(handwriting) * 1  # 1 point per handwriting section
                
                # Consider content keywords that suggest visual elements
                content = result.get('content', '').lower()
                visual_keywords = ['chart', 'graph', 'table', 'figure', 'diagram', 'data', 'trend', 
                                 'bar chart', 'line chart', 'pie chart', 'scatter plot', 'histogram']
                
                for keyword in visual_keywords:
                    if keyword in content:
                        score += 1
                
                return score
            
            # Sort results by visual content score (descending) while preserving similarity order for ties
            visual_results = []
            non_visual_results = []
            
            for result in results:
                visual_score = visual_content_score(result)
                result['visual_score'] = visual_score
                
                if visual_score > 0:
                    visual_results.append(result)
                else:
                    non_visual_results.append(result)
            
            # Sort visual results by visual score (descending), then by similarity
            visual_results.sort(key=lambda x: (x['visual_score'], x['similarity_score']), reverse=True)
            
            # Combine: visual results first, then non-visual results
            sorted_results = visual_results + non_visual_results
            
            logger.info(f"Visual prioritization: {len(visual_results)} visual results, {len(non_visual_results)} text-only results")
            return sorted_results
            
        except Exception as e:
            logger.error(f"Error sorting by visual content: {e}")
            return results  # Return original results if sorting fails

# Run the application
if __name__ == "__main__":
    app = DocumentSearchInterface()
    app.run()