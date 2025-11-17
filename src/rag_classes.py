import fitz  # PyMuPDF
from pathlib import Path
from typing import TypedDict, List, Dict, Optional, Annotated
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
import requests
import operator
from .ollama_client import OllamaClient
from .utils import setup_logger

from .config import Config
from .utils import setup_logger, validate_pdf_file

logger = setup_logger(__name__)

class PDFProcessor:
    """
    Processes PDF documents by extracting text and splitting into chunks.
    """
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize the PDF processor.
        
        Args:
            chunk_size: Size of text chunks (default from config)
            chunk_overlap: Overlap between chunks (default from config)
        """
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        logger.info(f"Initialized PDFProcessor with chunk_size={self.chunk_size}, "
                   f"chunk_overlap={self.chunk_overlap}")
    
    def extract_text_from_pdf(self, pdf_path: Path) -> List[Dict[str, any]]:
        """
        Extract text from PDF file with page numbers.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of dictionaries containing text and metadata for each page
            
        Raises:
            ValueError: If PDF file is invalid
            Exception: If PDF processing fails
        """
        if not validate_pdf_file(pdf_path):
            raise ValueError(f"Invalid PDF file: {pdf_path}")
        
        logger.info(f"Extracting text from: {pdf_path}")
        
        try:
            doc = fitz.open(pdf_path)
            pages = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                if text.strip():  # Only add pages with content
                    pages.append({
                        'text': text,
                        'page': page_num + 1,
                        'source': pdf_path.name
                    })
            
            doc.close()
            logger.info(f"Extracted text from {len(pages)} pages")
            return pages
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
    
    def create_chunks(self, pages: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        Split pages into chunks with metadata preservation.
        
        Args:
            pages: List of page dictionaries from extract_text_from_pdf
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        logger.info(f"Creating chunks from {len(pages)} pages")
        
        chunks = []
        for page_data in pages:
            text = page_data['text']
            page_num = page_data['page']
            source = page_data['source']
            
            # Split text into chunks
            text_chunks = self.text_splitter.split_text(text)
            
            # Add metadata to each chunk
            for i, chunk_text in enumerate(text_chunks):
                chunks.append({
                    'text': chunk_text,
                    'page': page_num,
                    'source': source,
                    'chunk_id': f"{source}_page{page_num}_chunk{i}"
                })
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def process_pdf(self, pdf_path: Path) -> List[Dict[str, any]]:
        """
        Complete pipeline: extract text and create chunks.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of chunk dictionaries ready for embedding
        """
        pages = self.extract_text_from_pdf(pdf_path)
        chunks = self.create_chunks(pages)
        return chunks

class VectorStore:
    """
    Manages document embeddings and similarity search using ChromaDB.
    """
    
    def __init__(self, 
                 collection_name: str = "documents",
                 embedding_model: str = None,
                 persist_directory: Path = None,
                 ollama_base_url: str = None):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: Name of the Ollama embedding model
            persist_directory: Directory to persist the database
            ollama_base_url: Base URL for Ollama API
        """
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model or Config.EMBEDDING_MODEL
        self.persist_directory = persist_directory or Config.VECTOR_STORE_PATH
        self.ollama_base_url = ollama_base_url or Config.OLLAMA_BASE_URL
        
        logger.info(f"Initializing VectorStore with Ollama model: {self.embedding_model_name}")
        logger.info(f"Ollama URL: {self.ollama_base_url}")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Vector store initialized with {self.collection.count()} documents")
    
    def add_documents(self, chunks: List[Dict[str, str]]) -> None:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of chunk dictionaries with 'text' and metadata
        """
        if not chunks:
            logger.warning("No chunks to add")
            return
        
        logger.info(f"Adding {len(chunks)} documents to vector store")
        
        # Extract texts and metadata
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [
            {
                'page': str(chunk.get('page', 'unknown')),
                'source': chunk.get('source', 'unknown'),
                'chunk_id': chunk.get('chunk_id', 'unknown')
            }
            for chunk in chunks
        ]
        ids = [chunk.get('chunk_id', f"doc_{i}") for i, chunk in enumerate(chunks)]
        
        # Generate embeddings using Ollama
        logger.info("Generating embeddings with Ollama...")
        embeddings = []
        for i, text in enumerate(texts):
            if (i + 1) % 10 == 0:
                logger.info(f"Generating embedding {i + 1}/{len(texts)}")
            embedding = self._generate_embedding(text)
            embeddings.append(embedding)
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Successfully added {len(chunks)} documents")
    
    def similarity_search(self, 
                         query: str, 
                         k: int = 4,
                         score_threshold: float = None) -> List[Dict[str, str]]:
        """
        Search for similar documents using the query with optional re-ranking.
        
        Args:
            query: Query text
            k: Number of results to return
            score_threshold: Optional minimum similarity score (0-1, higher is more similar)
            
        Returns:
            List of dictionaries with document text and metadata
        """
        logger.info(f"Searching for: '{query}' (k={k})")
        
        # Generate query embedding using Ollama
        query_embedding = self._generate_embedding(query)
        
        # Search with more results for filtering
        search_k = k * 2  # Get more results for better filtering
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=search_k
        )
        
        # Format results with similarity scores
        documents = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                # Convert distance to similarity score (cosine distance: 0=identical, 2=opposite)
                distance = results['distances'][0][i] if 'distances' in results else 1.0
                similarity = 1.0 - (distance / 2.0)  # Convert to 0-1 scale
                
                doc = {
                    'text': results['documents'][0][i],
                    'page': results['metadatas'][0][i].get('page', 'unknown'),
                    'source': results['metadatas'][0][i].get('source', 'unknown'),
                    'distance': distance,
                    'similarity': similarity
                }
                
                # Apply score threshold if provided
                if score_threshold is None or similarity >= score_threshold:
                    documents.append(doc)
        
        # Sort by similarity and limit to k results
        documents = sorted(documents, key=lambda x: x['similarity'], reverse=True)[:k]
        
        # Simple re-ranking: boost documents with query terms
        query_terms = set(query.lower().split())
        for doc in documents:
            doc_terms = set(doc['text'].lower().split())
            term_overlap = len(query_terms.intersection(doc_terms))
            # Slight boost for term overlap
            doc['relevance_score'] = doc['similarity'] + (term_overlap * 0.01)
        
        # Re-sort by relevance score
        documents = sorted(documents, key=lambda x: x['relevance_score'], reverse=True)
        
        scores_str = [f"{d['similarity']:.3f}" for d in documents]
        logger.info(f"Found {len(documents)} relevant documents (scores: {scores_str})")
        return documents
    
    def clear_collection(self) -> None:
        """
        Clear all documents from the collection.
        """
        logger.info("Clearing collection")
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Collection cleared")
    
    def get_collection_stats(self) -> Dict[str, int]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        return {
            'total_documents': count,
            'collection_name': self.collection_name
        }
    
    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using Ollama.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/embeddings",
                json={
                    "model": self.embedding_model_name,
                    "prompt": text
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('embedding', [])
            else:
                logger.error(f"Error generating embedding: {response.status_code}")
                raise Exception(f"Failed to generate embedding: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error calling Ollama embeddings API: {str(e)}")
            raise

class RAGState(TypedDict):
    """
    State definition for the RAG graph.
    Contains all data passed between nodes.
    """
    question: str
    context: str
    sources: List[Dict[str, str]]
    answer: str
    error: str


class RAGGraph:
    """
    LangGraph-based RAG pipeline.
    Orchestrates retrieval and generation steps.
    """
    
    def __init__(self, vector_store: VectorStore, ollama_client: OllamaClient):
        """
        Initialize the RAG graph.
        
        Args:
            vector_store: VectorStore instance for retrieval
            ollama_client: OllamaClient instance for generation
        """
        self.vector_store = vector_store
        self.ollama_client = ollama_client
        
        # Build the graph
        self.graph = self._build_graph()
        
        logger.info("RAG Graph initialized")
    
    def _build_graph(self) -> StateGraph:
        """
        Build the RAG state graph.
        
        Returns:
            Compiled StateGraph
        """
        # Define the graph
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate", self._generate_node)
        
        # Define edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        # Compile
        return workflow.compile()
    
    def _retrieve_node(self, state: RAGState) -> RAGState:
        """
        Retrieval node: search for relevant documents.
        
        Args:
            state: Current RAG state
            
        Returns:
            Updated state with context and sources
        """
        try:
            logger.info(f"Retrieving documents for: {state['question']}")
            
            # Search for relevant documents
            results = self.vector_store.similarity_search(
                query=state['question'],
                k=Config.TOP_K_DOCUMENTS,
                score_threshold=0.3  # Filter low-quality matches
            )
            
            # Build context from results
            context_parts = []
            for i, doc in enumerate(results, 1):
                context_parts.append(
                    f"[Document {i} - Page {doc['page']}]\n{doc['text']}"
                )
            
            context = "\n\n".join(context_parts)
            
            return {
                **state,
                "context": context,
                "sources": results,
                "error": ""
            }
            
        except Exception as e:
            logger.error(f"Error in retrieve node: {str(e)}")
            return {
                **state,
                "context": "",
                "sources": [],
                "error": f"Retrieval error: {str(e)}"
            }
    
    def _generate_node(self, state: RAGState) -> RAGState:
        """
        Generation node: generate answer using LLM.
        
        Args:
            state: Current RAG state with context
            
        Returns:
            Updated state with answer
        """
        try:
            logger.info("Generating answer")
            
            # Check if there was a retrieval error
            if state.get('error'):
                return {
                    **state,
                    "answer": "Unable to retrieve relevant documents. Please try again."
                }
            
            # Generate answer
            answer = self.ollama_client.generate(
                prompt=state['question'],
                context=state['context'],
                temperature=Config.TEMPERATURE
            )
            
            return {
                **state,
                "answer": answer,
                "error": ""
            }
            
        except Exception as e:
            logger.error(f"Error in generate node: {str(e)}")
            return {
                **state,
                "answer": f"Error generating answer: {str(e)}",
                "error": f"Generation error: {str(e)}"
            }
    
    def query(self, question: str) -> Dict[str, any]:
        """
        Execute the RAG pipeline for a question.
        
        Args:
            question: User question
            
        Returns:
            Dictionary with answer and sources
        """
        logger.info(f"Processing query: {question}")
        
        # Initial state
        initial_state = RAGState(
            question=question,
            context="",
            sources=[],
            answer="",
            error=""
        )
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        return {
            "answer": final_state.get("answer", "No answer generated."),
            "sources": final_state.get("sources", []),
            "error": final_state.get("error", "")
        }
    
    def stream_query(self, question: str):
        """
        Execute RAG pipeline with streaming response.
        
        Args:
            question: User question
            
        Yields:
            Tuples of (chunk_type, content)
        """
        logger.info(f"Processing streaming query: {question}")
        
        try:
            # Retrieve documents
            yield ("status", "Buscando documentos relevantes...")
            
            results = self.vector_store.similarity_search(
                query=question,
                k=Config.TOP_K_DOCUMENTS,
                score_threshold=0.3  # Filter low-quality matches
            )
            
            # Build context
            context_parts = []
            for i, doc in enumerate(results, 1):
                context_parts.append(
                    f"[Document {i} - Page {doc['page']}]\n{doc['text']}"
                )
            
            context = "\n\n".join(context_parts)
            
            yield ("status", "Generating response...")
            yield ("sources", results)
            
            # Stream generation
            for chunk in self.ollama_client.generate_stream(
                prompt=question,
                context=context,
                temperature=Config.TEMPERATURE
            ):
                yield ("answer", chunk)
                
        except Exception as e:
            logger.error(f"Error in stream query: {str(e)}")
            yield ("error", str(e))
