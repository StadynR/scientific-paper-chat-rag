import streamlit as st
from pathlib import Path
import tempfile
import os

from src.config import Config
from src.rag_classes import PDFProcessor, VectorStore, RAGGraph
from src.ollama_client import OllamaClient
from src.memory_model import MemoryModel
from src.utils import setup_logger


logger = setup_logger(__name__)


# Page configuration
st.set_page_config(
    page_title="Chat PDF Académico",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_session_state():
    """
    Initialize Streamlit session state variables.
    """
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    
    if 'memory_model' not in st.session_state:
        st.session_state.memory_model = None
    
    if 'rag_graph' not in st.session_state:
        st.session_state.rag_graph = None
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    
    if 'current_pdf' not in st.session_state:
        st.session_state.current_pdf = None
    
    if 'use_memorag' not in st.session_state:
        st.session_state.use_memorag = True


def load_components():
    """
    Load and initialize core components.
    
    Returns:
        Tuple of (vector_store, ollama_client, rag_graph)
    """
    try:
        # Initialize vector store
        vector_store = VectorStore()
        
        # Initialize Ollama client
        ollama_client = OllamaClient()
        
        # Check Ollama connection
        if not ollama_client.check_connection():
            st.error("Cannot connect to Ollama. Please ensure Ollama is running.")
            st.info(f"Expected Ollama URL: {Config.OLLAMA_BASE_URL}")
            st.stop()
        
        # Initialize Memory Model
        memory_model = MemoryModel()
        
        # Initialize RAG graph with MemoRAG support
        rag_graph = RAGGraph(vector_store, ollama_client, memory_model)
        
        return vector_store, ollama_client, memory_model, rag_graph
        
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        logger.error(f"Component initialization error: {str(e)}")
        st.stop()


def process_uploaded_pdf(uploaded_file, vector_store, memory_model):
    """
    Process uploaded PDF and add to vector store.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        vector_store: VectorStore instance
        memory_model: MemoryModel instance
    """
    try:
        with st.spinner("Processing PDF..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = Path(tmp_file.name)
            
            # Process PDF
            processor = PDFProcessor()
            chunks = processor.process_pdf(tmp_path)
            
            # Clear existing collection and add new chunks
            vector_store.clear_collection()
            vector_store.add_documents(chunks)
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            st.success(f"Successfully processed {uploaded_file.name}")
            st.info(f"Total chunks: {len(chunks)}")
            
            # Build global memory with MemoRAG (if enabled)
            if st.session_state.use_memorag:
                with st.spinner("Building global memory (MemoRAG)..."):
                    memory = memory_model.memorize_document(chunks)
                    st.success("✓ Global memory created")
                    
                    # Show memory stats
                    if memory.get('key_topics'):
                        with st.expander("📝 Memory Summary"):
                            st.write("**Key Topics:**")
                            for topic in memory['key_topics'][:5]:
                                st.write(f"- {topic}")
                            st.write(f"\n**Pages processed:** {memory['metadata']['total_pages']}")
            else:
                st.info("ℹ️ MemoRAG disabled - memory not built")
            
            return True
            
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        logger.error(f"PDF processing error: {str(e)}")
        return False


def display_chat_interface(rag_graph):
    """
    Display the chat interface for querying documents.
    
    Args:
        rag_graph: RAGGraph instance
    """
    st.subheader("Chat with your document")
    
    # Add CSS for animated loading dots
    st.markdown("""
    <style>
    .loading-text {
        color: rgba(128, 128, 128, 0.6);
        font-style: italic;
        font-size: 14px;
        animation: fadeInOut 1.5s ease-in-out infinite;
    }
    
    .loading-dots {
        display: inline-block;
    }
    
    .loading-dots span {
        animation: blink 1.4s infinite;
        animation-fill-mode: both;
    }
    
    .loading-dots span:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .loading-dots span:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    @keyframes blink {
        0%, 80%, 100% {
            opacity: 0;
        }
        40% {
            opacity: 1;
        }
    }
    
    @keyframes fadeInOut {
        0%, 100% {
            opacity: 0.4;
        }
        50% {
            opacity: 0.8;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources if available
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}** (Page {source['page']}):")
                        st.text(source['text'][:300] + "...")
                        st.markdown("---")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your document"):
        # Add user message to chat
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt
        })
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            sources = []
            is_generating = False
            
            try:
                # Show animated loading message
                message_placeholder.markdown(
                    '<div class="loading-text">Generating response<span class="loading-dots"><span>.</span><span>.</span><span>.</span></span></div>',
                    unsafe_allow_html=True
                )
                
                # Container for clues
                clues_display = None
                clues_container = st.container()
                
                # Stream response
                for chunk_type, content in rag_graph.stream_query(prompt, use_memorag=st.session_state.use_memorag):
                    if chunk_type == "clues" and st.session_state.use_memorag:
                        # Display generated clues only if MemoRAG is enabled
                        with clues_container:
                            with st.expander("🧠 MemoRAG Search Clues", expanded=False):
                                st.markdown("**Generated search strategies:**")
                                for i, clue in enumerate(content, 1):
                                    if i == 1:
                                        st.markdown(f"- 🎯 {clue} _(original)_")
                                    else:
                                        st.markdown(f"- 🔍 {clue}")
                    elif chunk_type == "answer":
                        if not is_generating:
                            is_generating = True
                            # Clear loading message when first token arrives
                        full_response += content
                        # Show cursor while streaming
                        message_placeholder.markdown(full_response + "▌")
                    elif chunk_type == "sources":
                        sources = content
                    elif chunk_type == "status":
                        # Only show loading if we haven't started generating yet
                        if not is_generating:
                            if content == "generating_clues" and st.session_state.use_memorag:
                                message_placeholder.markdown(
                                    '<div class="loading-text">🧠 Analyzing document memory<span class="loading-dots"><span>.</span><span>.</span><span>.</span></span></div>',
                                    unsafe_allow_html=True
                                )
                            elif content == "retrieving":
                                retrieval_mode = "MemoRAG" if st.session_state.use_memorag else "Standard"
                                message_placeholder.markdown(
                                    f'<div class="loading-text">🔍 Retrieving ({retrieval_mode})<span class="loading-dots"><span>.</span><span>.</span><span>.</span></span></div>',
                                    unsafe_allow_html=True
                                )
                            else:
                                message_placeholder.markdown(
                                    '<div class="loading-text">Generating response<span class="loading-dots"><span>.</span><span>.</span><span>.</span></span></div>',
                                    unsafe_allow_html=True
                                )
                    elif chunk_type == "error":
                        message_placeholder.error(f"Error: {content}")
                        return
                
                # Final update - remove cursor
                if full_response:
                    message_placeholder.markdown(full_response)
                else:
                    message_placeholder.warning("No response generated.")
                    return
                
                # Add assistant message to chat
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources
                })
                
                # Display sources
                if sources:
                    with st.expander("View Sources"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**Source {i}** (Page {source['page']}):")
                            st.text(source['text'][:300] + "...")
                            st.markdown("---")
                
            except Exception as e:
                message_placeholder.error(f"Error generating response: {str(e)}")
                logger.error(f"Query error: {str(e)}")


def main():
    """
    Main application entry point.
    """
    initialize_session_state()
    
    # Header
    st.title("📚 Academic PDF Chat")
    st.markdown("Upload a PDF document and ask questions about its content using AI.")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Model information
        st.subheader("Model Settings")
        st.info(f"**Ollama Model:** {Config.OLLAMA_MODEL}")
        st.info(f"**Memory Model:** {Config.MEMORY_MODEL}")
        st.info(f"**Embedding Model:** {Config.EMBEDDING_MODEL}")
        
        # MemoRAG toggle
        st.subheader("Retrieval Mode")
        use_memorag = st.toggle(
            "Enable MemoRAG",
            value=st.session_state.use_memorag,
            help="Use MemoRAG for intelligent clue-based retrieval. When disabled, uses standard similarity search."
        )
        
        if use_memorag != st.session_state.use_memorag:
            st.session_state.use_memorag = use_memorag
            st.rerun()
        
        # Show status
        if st.session_state.use_memorag:
            st.success("🧠 MemoRAG Active")
        else:
            st.info("📋 Standard Retrieval")
        
        # Load components
        if st.session_state.vector_store is None:
            vector_store, ollama_client, memory_model, rag_graph = load_components()
            st.session_state.vector_store = vector_store
            st.session_state.ollama_client = ollama_client
            st.session_state.memory_model = memory_model
            st.session_state.rag_graph = rag_graph
        
        # Document upload
        st.subheader("Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help=f"Maximum file size: {Config.MAX_UPLOAD_SIZE_MB}MB"
        )
        
        if uploaded_file is not None:
            if st.button("Process PDF"):
                success = process_uploaded_pdf(
                    uploaded_file,
                    st.session_state.vector_store,
                    st.session_state.memory_model
                )
                if success:
                    st.session_state.documents_loaded = True
                    st.session_state.current_pdf = uploaded_file.name
                    st.rerun()
        
        # Current document status
        if st.session_state.documents_loaded:
            st.success(f"Document loaded: {st.session_state.current_pdf}")
            
            # Collection stats
            stats = st.session_state.vector_store.get_collection_stats()
            st.metric("Total Chunks", stats['total_documents'])
        
        # Clear conversation
        if st.button("Clear Conversation"):
            st.session_state.chat_history = []
            st.rerun()
        
        # Clear database
        if st.button("Clear Database"):
            if st.session_state.vector_store:
                st.session_state.vector_store.clear_collection()
                if st.session_state.memory_model:
                    st.session_state.memory_model.clear_memory()
                st.session_state.documents_loaded = False
                st.session_state.current_pdf = None
                st.session_state.chat_history = []
                st.success("Database and memory cleared")
                st.rerun()
    
    # Main content
    if st.session_state.documents_loaded:
        display_chat_interface(st.session_state.rag_graph)
    else:
        st.info("Please upload a PDF document to begin.")
        
        # Instructions
        st.markdown("""
        ### How to use:
        1. Upload a PDF document using the sidebar
        2. Click "Process PDF" to analyze the document
        3. Ask questions about the document content
        4. The AI will provide answers based on the document
        
        ### Features:
        - **MemoRAG**: Intelligent clue-based retrieval using document memory
        - **Standard RAG**: Classic similarity search (toggle to compare)
        - Source citations with page numbers
        - Streaming responses for better user experience
        - Local processing with Ollama
        """)


if __name__ == "__main__":
    main()