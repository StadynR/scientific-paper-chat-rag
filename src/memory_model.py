import json
import requests
from pathlib import Path
from typing import List, Dict, Optional
from .config import Config
from .utils import setup_logger

logger = setup_logger(__name__)


class MemoryModel:
    """
    Memory Model for MemoRAG implementation.
    
    This model processes documents to create a compressed "global memory"
    and generates clues to guide retrieval based on queries.
    """
    
    def __init__(self, 
                 model_name: Optional[str] = None,
                 temperature: Optional[float] = None,
                 ollama_base_url: Optional[str] = None):
        """
        Initialize the Memory Model.
        
        Args:
            model_name: Name of the Ollama model to use (default: llama3.2)
            temperature: Temperature for generation (default from config)
            ollama_base_url: Base URL for Ollama API
        """
        self.model_name = model_name or Config.MEMORY_MODEL
        self.temperature = temperature or Config.MEMORY_TEMPERATURE
        self.ollama_base_url = ollama_base_url or Config.OLLAMA_BASE_URL
        self.memory_store_path = Config.MEMORY_STORE_PATH
        
        logger.info(f"Initialized MemoryModel with model: {self.model_name}")
    
    def memorize_document(self, chunks: List[Dict[str, str]]) -> Dict[str, any]:
        """
        Process document chunks and create a compressed memory representation.
        
        This method analyzes the document structure and creates a semantic
        summary that captures key concepts, topics, and relationships.
        
        Args:
            chunks: List of document chunks with text and metadata
            
        Returns:
            Dictionary containing memory representation with:
                - summary: Compressed text summary
                - key_topics: Main topics/concepts
                - sections: Section-level summaries
                - metadata: Document metadata
        """
        logger.info(f"Memorizing document with {len(chunks)} chunks")
        
        try:
            # Group chunks by page for better structure
            pages_content = {}
            for chunk in chunks:
                page = chunk.get('page', 'unknown')
                if page not in pages_content:
                    pages_content[page] = []
                pages_content[page].append(chunk['text'])
            
            # Create page-level summaries
            page_summaries = []
            for page, texts in sorted(pages_content.items()):
                combined_text = "\n".join(texts)
                summary = self._compress_text(combined_text, page)
                page_summaries.append({
                    'page': page,
                    'summary': summary
                })
            
            # Create global summary from page summaries
            all_summaries = "\n\n".join([
                f"Page {ps['page']}: {ps['summary']}" 
                for ps in page_summaries
            ])
            
            global_summary = self._generate_global_summary(all_summaries)
            
            # Extract key topics
            key_topics = self._extract_key_topics(global_summary)
            
            memory = {
                'global_summary': global_summary,
                'key_topics': key_topics,
                'page_summaries': page_summaries,
                'metadata': {
                    'total_chunks': len(chunks),
                    'total_pages': len(pages_content),
                    'source': chunks[0].get('source', 'unknown') if chunks else 'unknown'
                }
            }
            
            # Persist memory
            self._save_memory(memory)
            
            logger.info("Document memory created successfully")
            return memory
            
        except Exception as e:
            logger.error(f"Error memorizing document: {str(e)}")
            raise
    
    def generate_clues(self, query: str, memory: Optional[Dict] = None) -> List[str]:
        """
        Generate search clues based on the query and global memory.
        
        Clues are intermediate representations that help guide the retrieval
        process by expanding the query with relevant context from memory.
        
        Args:
            query: User's original query
            memory: Optional memory dict (loads from disk if not provided)
            
        Returns:
            List of clue strings to guide retrieval
        """
        logger.info(f"Generating clues for query: {query}")
        
        try:
            # Load memory if not provided
            if memory is None:
                memory = self._load_memory()
                if not memory:
                    logger.warning("No memory available, using query directly")
                    return [query]
            
            # Build context from memory
            memory_context = self._build_memory_context(memory)
            
            # Generate clues using the memory model
            clues = self._generate_clues_with_model(query, memory_context)
            
            logger.info(f"Generated {len(clues)} clues")
            return clues
            
        except Exception as e:
            logger.error(f"Error generating clues: {str(e)}")
            # Fallback to original query
            return [query]
    
    def _compress_text(self, text: str, page: int) -> str:
        """
        Compress a text chunk into a concise summary.
        
        Args:
            text: Text to compress
            page: Page number for context
            
        Returns:
            Compressed summary
        """
        prompt = f"""Summarize the following text from page {page} of a scientific paper. 
Focus on key concepts, methods, findings, and important technical details.
Keep the summary concise but informative (2-3 sentences).

Text:
{text[:2000]}  

Summary:"""
        
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get('response', '').strip()
            else:
                logger.warning(f"Compression failed, using truncated text")
                return text[:500]
                
        except Exception as e:
            logger.error(f"Error compressing text: {str(e)}")
            return text[:500]
    
    def _generate_global_summary(self, page_summaries: str) -> str:
        """
        Generate a global summary from page-level summaries.
        
        Args:
            page_summaries: Combined page summaries
            
        Returns:
            Global document summary
        """
        prompt = f"""Based on the following page-by-page summaries of a scientific paper, 
create a comprehensive overview that captures:
1. Main research topic and objectives
2. Key methodologies used
3. Important findings or results
4. Significant concepts or terms

Page Summaries:
{page_summaries[:3000]}

Comprehensive Overview:"""
        
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json().get('response', '').strip()
            else:
                return page_summaries[:1000]
                
        except Exception as e:
            logger.error(f"Error generating global summary: {str(e)}")
            return page_summaries[:1000]
    
    def _extract_key_topics(self, summary: str) -> List[str]:
        """
        Extract key topics from the global summary.
        
        Args:
            summary: Global document summary
            
        Returns:
            List of key topics/concepts
        """
        prompt = f"""Extract 5-8 key topics, concepts, or technical terms from this summary.
List only the topics, one per line, without numbering or explanations.

Summary:
{summary}

Key Topics:"""
        
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": 0.3,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                topics_text = response.json().get('response', '').strip()
                topics = [t.strip() for t in topics_text.split('\n') if t.strip()]
                return topics[:8]
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error extracting topics: {str(e)}")
            return []
    
    def _build_memory_context(self, memory: Dict) -> str:
        """
        Build a context string from memory for clue generation.
        
        Args:
            memory: Memory dictionary
            
        Returns:
            Formatted memory context string
        """
        context_parts = []
        
        # Add global summary
        if 'global_summary' in memory:
            context_parts.append(f"Document Overview:\n{memory['global_summary']}")
        
        # Add key topics
        if 'key_topics' in memory and memory['key_topics']:
            topics_str = ", ".join(memory['key_topics'])
            context_parts.append(f"\nKey Topics: {topics_str}")
        
        # Add abbreviated page summaries
        if 'page_summaries' in memory:
            top_pages = memory['page_summaries'][:5]  # Limit to first 5 pages
            pages_str = "\n".join([
                f"- Page {ps['page']}: {ps['summary'][:200]}"
                for ps in top_pages
            ])
            context_parts.append(f"\nKey Sections:\n{pages_str}")
        
        return "\n".join(context_parts)
    
    def _generate_clues_with_model(self, query: str, memory_context: str) -> List[str]:
        """
        Generate clues using the memory model.
        
        Args:
            query: User query
            memory_context: Memory context string
            
        Returns:
            List of generated clues
        """
        num_clues = Config.NUM_CLUES
        
        prompt = f"""You are helping to search a scientific paper. Given the user's question and 
an overview of the document, generate {num_clues} search clues or phrases that would help 
find relevant information.

Each clue should:
- Be a specific phrase or question related to the user's query
- Reference concepts or sections that likely contain the answer
- Be concrete and searchable

Document Context:
{memory_context}

User Question: {query}

Generate {num_clues} search clues (one per line):"""
        
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                clues_text = response.json().get('response', '').strip()
                clues = [c.strip() for c in clues_text.split('\n') if c.strip()]
                
                # Clean up clues (remove numbering, bullets, etc.)
                clean_clues = []
                for clue in clues:
                    # Remove common prefixes
                    clue = clue.lstrip('0123456789.-*> ')
                    if clue and len(clue) > 10:  # Filter too short clues
                        clean_clues.append(clue)
                
                # Always include original query as first clue
                return [query] + clean_clues[:num_clues]
            else:
                logger.warning("Clue generation failed, using query only")
                return [query]
                
        except Exception as e:
            logger.error(f"Error generating clues with model: {str(e)}")
            return [query]
    
    def _save_memory(self, memory: Dict) -> None:
        """
        Save memory to disk.
        
        Args:
            memory: Memory dictionary to save
        """
        try:
            memory_file = self.memory_store_path / "current_memory.json"
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(memory, f, indent=2, ensure_ascii=False)
            logger.info(f"Memory saved to {memory_file}")
        except Exception as e:
            logger.error(f"Error saving memory: {str(e)}")
    
    def _load_memory(self) -> Optional[Dict]:
        """
        Load memory from disk.
        
        Returns:
            Memory dictionary or None if not found
        """
        try:
            memory_file = self.memory_store_path / "current_memory.json"
            if memory_file.exists():
                with open(memory_file, 'r', encoding='utf-8') as f:
                    memory = json.load(f)
                logger.info("Memory loaded from disk")
                return memory
            else:
                logger.warning("No memory file found")
                return None
        except Exception as e:
            logger.error(f"Error loading memory: {str(e)}")
            return None
    
    def clear_memory(self) -> None:
        """
        Clear the stored memory.
        """
        try:
            memory_file = self.memory_store_path / "current_memory.json"
            if memory_file.exists():
                memory_file.unlink()
                logger.info("Memory cleared")
        except Exception as e:
            logger.error(f"Error clearing memory: {str(e)}")
