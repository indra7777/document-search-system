import streamlit as st
import logging
import os
import json
import webbrowser
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentSearchInterface:
    def __init__(self):
        self.config = Config()
        self.config.create_directories()
        
        # Initialize services
        self._init_services()
        
        # Initialize session state
        if 'vector_db_loaded' not in st.session_state:
            st.session_state.vector_db_loaded = False
        if 'documents_processed' not in st.session_state:
            st.session_state.documents_processed = 0
    
    def _init_services(self):
        """Initialize all services using singleton pattern to prevent memory leaks"""
        try:
            # Set config for service manager
            service_manager.set_config(self.config)
            
            # Get services using singleton pattern - prevents multiple instances
            self.doc_processor = service_manager.get_service(
                'document_processor', DocumentProcessor, self.config
            )
            self.embedding_service = service_manager.get_service(
                'embedding_service', EmbeddingService, self.config
            )
            self.vector_db = service_manager.get_service(
                'vector_database', VectorDatabase, self.config
            )
            self.llm_service = service_manager.get_service(
                'llm_service', LLMService, self.config
            )
            self.doc_analyzer = service_manager.get_service(
                'document_analyzer', DocumentAnalyzer, self.config
            )
            
            # Try to load existing vector database
            self.vector_db.load_index(self.config.EMBEDDING_DIMENSION)
            
            # Log memory status
            memory_stats = service_manager.get_memory_stats()
            logger.info(f"Services initialized. Memory stats: {memory_stats}")
            
        except Exception as e:
            st.error(f"Error initializing services: {str(e)}")
            logger.error(f"Service initialization error: {str(e)}")
    
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
        
        if uploaded_files and st.sidebar.button("🔍 Analyze & Process Files"):
            self._process_uploaded_files_enhanced(uploaded_files, analyze_tables, analyze_images, persist_files)
        
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
            enable_llm = st.checkbox("Generate AI response", value=True)
        with col4:
            prioritize_visuals = st.checkbox("Prioritize charts/images", value=False, 
                                           help="Show results with charts and images first")
        
        # Search button and results
        if st.button("Search", type="primary") and query:
            self._perform_search(query, top_k, similarity_threshold, enable_llm, prioritize_visuals)
    
    def _perform_search(self, query: str, top_k: int, threshold: float, enable_llm: bool, prioritize_visuals: bool = False):
        """Perform document search and display results"""
        try:
            with st.spinner("Searching documents..."):
                # Generate query embedding
                query_embedding = self.embedding_service.embed_query(query)
                
                # Search vector database
                search_results = self.vector_db.search(
                    query_embedding, 
                    top_k=top_k, 
                    threshold=threshold
                )
                
                if not search_results:
                    st.warning("No relevant documents found. Try adjusting the similarity threshold.")
                    return
                
                # Display results count
                st.success(f"Found {len(search_results)} relevant documents")
                
                # Generate LLM response if enabled
                if enable_llm:
                    with st.spinner("Generating AI response..."):
                        llm_response = self.llm_service.generate_response(query, search_results)
                        self._display_llm_response(llm_response)
                
                # Sort results to prioritize visual content if requested
                if prioritize_visuals:
                    search_results = self._sort_results_by_visual_content(search_results)
                
                # Display search results
                self._display_search_results(search_results)
                
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            logger.error(f"Search error: {str(e)}")
    
    def _display_llm_response(self, response: Dict[str, Any]):
        """Display LLM generated response"""
        st.subheader("🤖 AI Response")
        
        with st.container():
            st.markdown(response['answer'])
            
            # Display response metadata
            with st.expander("Response Details"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Model Used", response['model'])
                    st.metric("Context Documents", response['context_used'])
                with col2:
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
    
    def _process_uploaded_files_enhanced(self, uploaded_files, analyze_tables=True, analyze_images=True, persist_files=True):
        """Enhanced processing of uploaded files with advanced analysis"""
        try:
            analysis_results = []
            
            with st.spinner("🔍 Analyzing uploaded files (tables, images, handwriting)..."):
                progress_bar = st.progress(0)
                
                for i, uploaded_file in enumerate(uploaded_files):
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
                    analysis_results.append(analysis_result)
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Display analysis results with tracking info
                self._display_analysis_results(analysis_results)
                
                # Process documents for vector database (existing functionality)
                file_paths = [result.get("file_path") for result in analysis_results if "file_path" in result]
                if file_paths:
                    # Monitor memory before processing
                    gpu_optimizer.monitor_memory_usage()
                    self._process_documents(file_paths)
                    # Cleanup after processing
                    gpu_optimizer.cleanup_gpu_memory()
                
            st.success(f"✅ Successfully analyzed and processed {len(uploaded_files)} files!")
            
        except Exception as e:
            st.error(f"❌ Error processing files: {str(e)}")
            logger.error(f"Enhanced file processing error: {str(e)}")
    
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
        
        # Generate embeddings
        embedded_chunks = self.embedding_service.embed_documents(all_chunks)
        
        # Add to vector database
        self.vector_db.add_documents(embedded_chunks)
        
        # Save database
        self.vector_db.save_index()
        
        # Update session state
        st.session_state.documents_processed += len(processed_docs)
        st.session_state.vector_db_loaded = True
    
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
                    st.warning(f"🔁 **Duplicate Document Detected!** This file is identical to: {result['duplicate_of']}")
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