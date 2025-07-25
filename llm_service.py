import requests
import json
import logging
from typing import List, Dict, Any, Optional
import ollama

class LLMService:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.client = ollama.Client()
        self._ensure_model_available()
    
    def _ensure_model_available(self):
        """Ensure the specified model is available in Ollama"""
        try:
            # Check if model exists
            models = self.client.list()
            model_names = [model['name'] for model in models['models']]
            
            if self.config.LOCAL_LLM_MODEL not in model_names:
                self.logger.info(f"Model {self.config.LOCAL_LLM_MODEL} not found. Attempting to pull...")
                self.client.pull(self.config.LOCAL_LLM_MODEL)
                self.logger.info(f"Successfully pulled {self.config.LOCAL_LLM_MODEL}")
            else:
                self.logger.info(f"Model {self.config.LOCAL_LLM_MODEL} is available")
                
        except Exception as e:
            self.logger.error(f"Error checking/pulling model: {str(e)}")
            self.logger.info("Make sure Ollama is running and accessible")
    
    def generate_response(self, query: str, context_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate response using local LLM with retrieved context
        
        Args:
            query: User's search query
            context_docs: Retrieved relevant documents
            
        Returns:
            Generated response with metadata
        """
        try:
            # Prepare context from retrieved documents
            context = self._prepare_context(context_docs)
            
            # Create prompt with context and query
            prompt = self._create_rag_prompt(query, context)
            
            # Generate response using Ollama
            response = self.client.generate(
                model=self.config.LOCAL_LLM_MODEL,
                prompt=prompt,
                options={
                    'temperature': self.config.LLM_TEMPERATURE,
                    'num_predict': self.config.MAX_TOKENS,
                    'top_k': 40,
                    'top_p': 0.9,
                }
            )
            
            return {
                'answer': response['response'],
                'model': self.config.LOCAL_LLM_MODEL,
                'context_used': len(context_docs),
                'sources': self._extract_sources(context_docs),
                'query': query
            }
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return {
                'answer': f"Sorry, I encountered an error while generating the response: {str(e)}",
                'model': self.config.LOCAL_LLM_MODEL,
                'context_used': 0,
                'sources': [],
                'query': query
            }
    
    def _prepare_context(self, context_docs: List[Dict[str, Any]]) -> str:
        """Prepare context string from retrieved documents"""
        if not context_docs:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            source_info = f"Source {i}: {doc.get('file_name', 'Unknown')}"
            content = doc.get('content', '').strip()
            
            # Limit context length per document
            if len(content) > 1000:
                content = content[:1000] + "..."
            
            context_parts.append(f"{source_info}\n{content}")
        
        return "\n\n".join(context_parts)
    
    def _create_rag_prompt(self, query: str, context: str) -> str:
        """Create RAG prompt template"""
        prompt = f"""You are a helpful AI assistant that answers questions based on the provided context. 
Use the context below to answer the user's question. If the answer cannot be found in the context, say so clearly.
Always cite the sources you use in your answer.

Context:
{context}

Question: {query}

Answer: """
        
        return prompt
    
    def _extract_sources(self, context_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract source information from context documents"""
        sources = []
        for doc in context_docs:
            source = {
                'file_name': doc.get('file_name', 'Unknown'),
                'file_path': doc.get('file_path', ''),
                'similarity_score': doc.get('similarity_score', 0.0),
                'chunk_id': doc.get('chunk_id', ''),
                'file_type': doc.get('file_type', '')
            }
            sources.append(source)
        
        return sources
    
    def summarize_document(self, document_content: str, file_name: str) -> str:
        """Generate a summary of a document"""
        try:
            # Truncate content if too long
            max_content_length = 4000
            if len(document_content) > max_content_length:
                document_content = document_content[:max_content_length] + "..."
            
            prompt = f"""Please provide a concise summary of the following document:

Document: {file_name}

Content:
{document_content}

Summary:"""
            
            response = self.client.generate(
                model=self.config.LOCAL_LLM_MODEL,
                prompt=prompt,
                options={
                    'temperature': 0.3,
                    'num_predict': 500,
                }
            )
            
            return response['response']
            
        except Exception as e:
            self.logger.error(f"Error summarizing document: {str(e)}")
            return f"Error generating summary: {str(e)}"
    
    def generate_keywords(self, text: str) -> List[str]:
        """Extract keywords from text using LLM"""
        try:
            prompt = f"""Extract the most important keywords and phrases from the following text. Return only the keywords, separated by commas.

Text:
{text[:2000]}  # Limit text length

Keywords:"""
            
            response = self.client.generate(
                model=self.config.LOCAL_LLM_MODEL,
                prompt=prompt,
                options={
                    'temperature': 0.2,
                    'num_predict': 200,
                }
            )
            
            # Parse keywords from response
            keywords_text = response['response'].strip()
            keywords = [kw.strip() for kw in keywords_text.split(',') if kw.strip()]
            
            return keywords[:10]  # Return top 10 keywords
            
        except Exception as e:
            self.logger.error(f"Error generating keywords: {str(e)}")
            return []
    
    def check_model_availability(self) -> bool:
        """Check if the LLM model is available and working"""
        try:
            test_response = self.client.generate(
                model=self.config.LOCAL_LLM_MODEL,
                prompt="Hello, are you working?",
                options={'num_predict': 50}
            )
            return True
        except Exception as e:
            self.logger.error(f"Model availability check failed: {str(e)}")
            return False