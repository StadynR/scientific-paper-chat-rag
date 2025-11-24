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


def apply_custom_theme(theme: str):
    """
    Apply custom theme styling.
    
    Args:
        theme: 'dark' or 'light'
    """
    if theme == 'dark':
        st.markdown("""
        <style>
            /* Dark theme colors - improved */
            [data-testid="stAppViewContainer"] {
                background-color: #0f1318;
            }
            
            [data-testid="stHeader"] {
                background-color: #0f1318;
            }
            
            [data-testid="stSidebar"] {
                background-color: #1a1d24;
            }
            
            [data-testid="stSidebarNav"] {
                background-color: #1a1d24;
            }
            
            /* Main content background */
            .main .block-container {
                background-color: #0f1318;
            }
            
            /* Text colors */
            .stApp {
                color: #e8eaed;
            }
            
            /* Titles and headers */
            h1, h2, h3, h4, h5, h6 {
                color: #b8bcc2 !important;
            }
            
            .stMarkdown h1 {
                color: #b8bcc2 !important;
            }
            
            /* Chat messages */
            [data-testid="stChatMessage"] {
                background-color: #1e2329 !important;
                border: 1px solid #2d3139;
            }
            
            [data-testid="stChatMessageContent"] {
                color: #e8eaed;
            }
            
            /* Buttons */
            .stButton>button {
                background-color: #2563eb;
                color: white;
                border-radius: 8px;
                border: none;
                transition: all 0.3s;
            }
            
            .stButton>button:hover {
                background-color: #1d4ed8;
                box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
            }
            
            /* Selectbox and inputs */
            [data-testid="stSelectbox"] > div > div {
                background-color: #252a31 !important;
                color: #e8eaed;
                border: 1px solid #3d4451 !important;
            }
            
            .stSelectbox label, .stTextInput label {
                color: #b8bcc2 !important;
            }
            
            /* Selectbox inner elements */
            .stSelectbox > div > div > div {
                background-color: #252a31 !important;
            }
            
            .stSelectbox [data-baseweb="select"] {
                background-color: #252a31 !important;
            }
            
            .stSelectbox [data-baseweb="select"] > div {
                background-color: #252a31 !important;
                color: #e8eaed !important;
            }
            
            /* Selectbox arrow (dropdown icon) */
            .stSelectbox svg {
                fill: #f0f2f5 !important;
                color: #f0f2f5 !important;
            }
            
            /* Selectbox on hover */
            .stSelectbox [data-baseweb="select"]:hover {
                border-color: #2563eb !important;
            }
            
            /* Dropdown menu */
            [data-baseweb="popover"] {
                background-color: #252a31 !important;
            }
            
            [role="listbox"] {
                background-color: #252a31 !important;
                border: 1px solid #3d4451 !important;
            }
            
            [role="option"] {
                background-color: #252a31 !important;
                color: #e8eaed !important;
            }
            
            [role="option"]:hover {
                background-color: #2d3340 !important;
                color: #fff !important;
            }
            
            /* File uploader */
            [data-testid="stFileUploader"] {
                background-color: #1e2329;
                border: 1px solid #2d3139;
                border-radius: 8px;
            }
            
            /* Expander */
            [data-testid="stExpander"] {
                background-color: #1e2329;
                border: 1px solid #2d3139;
                border-radius: 8px;
            }
            
            .streamlit-expanderHeader {
                background-color: #1e2329;
                color: #e8eaed;
            }
            
            /* Info boxes */
            .stAlert {
                background-color: #1e2329;
                border: 1px solid #2d3139;
                color: #f0f2f5 !important;
            }
            
            .stAlert > div {
                color: #f0f2f5 !important;
            }
            
            /* Success box */
            [data-testid="stSuccess"] {
                background-color: #1e3a2a;
                border: 1px solid #2d5a3d;
            }
            
            [data-testid="stSuccess"], [data-testid="stSuccess"] * {
                color: #d1fae5 !important;
            }
            
            /* Warning box */
            [data-testid="stWarning"] {
                background-color: #3a2e1e;
                border: 1px solid #5a4a2d;
            }
            
            [data-testid="stWarning"], [data-testid="stWarning"] * {
                color: #fef3c7 !important;
            }
            
            /* Info box text */
            [data-testid="stInfo"] {
                background-color: #1e2d3a;
                border: 1px solid #2d4a5a;
            }
            
            [data-testid="stInfo"], [data-testid="stInfo"] * {
                color: #bfdbfe !important;
            }
            
            /* Divider */
            hr {
                border-color: #2d3139;
            }
            
            /* Toggle switch */
            .stCheckbox, .stToggle {
                color: #f0f2f5 !important;
            }
            
            .stCheckbox label, .stToggle label {
                color: #f0f2f5 !important;
            }
            
            .stCheckbox span, .stToggle span {
                color: #f0f2f5 !important;
            }
            
            .stCheckbox p, .stToggle p {
                color: #f0f2f5 !important;
            }
            
            .stCheckbox div, .stToggle div {
                color: #f0f2f5 !important;
            }
            
            [data-testid="stCheckbox"] label span {
                color: #f0f2f5 !important;
            }
            
            label[data-testid="stWidgetLabel"] {
                color: #f0f2f5 !important;
            }
            
            /* Markdown text */
            .stMarkdown {
                color: #e8eaed;
            }
            
            /* Code blocks */
            code {
                background-color: #1e2329;
                color: #60a5fa;
                padding: 2px 6px;
                border-radius: 4px;
            }
            
            pre {
                background-color: #1e2329;
                border: 1px solid #2d3139;
            }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
            /* Light theme colors - improved */
            [data-testid="stAppViewContainer"] {
                background-color: #ffffff;
            }
            
            [data-testid="stHeader"] {
                background-color: #ffffff;
            }
            
            [data-testid="stSidebar"] {
                background-color: #f8f9fa;
            }
            
            [data-testid="stSidebarNav"] {
                background-color: #f8f9fa;
            }
            
            /* Main content background */
            .main .block-container {
                background-color: #ffffff;
            }
            
            /* Text colors */
            .stApp {
                color: #1f2937;
            }
            
            /* Chat messages */
            [data-testid="stChatMessage"] {
                background-color: #f3f4f6 !important;
                border: 1px solid #e5e7eb;
            }
            
            [data-testid="stChatMessageContent"] {
                color: #1f2937;
            }
            
            /* Buttons */
            .stButton>button {
                background-color: #2563eb;
                color: white;
                border-radius: 8px;
                border: none;
                transition: all 0.3s;
            }
            
            .stButton>button:hover {
                background-color: #1d4ed8;
                box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
            }
            
            /* Selectbox and inputs */
            [data-testid="stSelectbox"] > div > div {
                background-color: #ffffff;
                color: #1f2937;
                border: 1px solid #d1d5db;
            }
            
            .stSelectbox label, .stTextInput label {
                color: #6b7280 !important;
            }
            
            /* File uploader */
            [data-testid="stFileUploader"] {
                background-color: #f9fafb;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
            }
            
            /* Expander */
            [data-testid="stExpander"] {
                background-color: #f9fafb;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
            }
            
            .streamlit-expanderHeader {
                background-color: #f9fafb;
                color: #1f2937;
            }
            
            /* Info boxes */
            .stAlert {
                background-color: #f9fafb;
                border: 1px solid #e5e7eb;
                color: #1f2937;
            }
            
            /* Success box */
            [data-testid="stSuccess"] {
                background-color: #ecfdf5;
                border: 1px solid #a7f3d0;
            }
            
            /* Warning box */
            [data-testid="stWarning"] {
                background-color: #fffbeb;
                border: 1px solid #fde68a;
            }
            
            /* Divider */
            hr {
                border-color: #e5e7eb;
            }
            
            /* Toggle switch */
            .stCheckbox, .stToggle {
                color: #1f2937;
            }
            
            /* Markdown text */
            .stMarkdown {
                color: #1f2937;
            }
            
            /* Code blocks */
            code {
                background-color: #f3f4f6;
                color: #2563eb;
                padding: 2px 6px;
                border-radius: 4px;
            }
            
            pre {
                background-color: #f9fafb;
                border: 1px solid #e5e7eb;
            }
        </style>
        """, unsafe_allow_html=True)


def initialize_session_state():
    """
    Initialize Streamlit session state variables.
    """
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    
    if 'ollama_client' not in st.session_state:
        st.session_state.ollama_client = None
    
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
    
    if 'selected_generation_model' not in st.session_state:
        st.session_state.selected_generation_model = Config.OLLAMA_MODEL
    
    if 'selected_memory_model' not in st.session_state:
        st.session_state.selected_memory_model = Config.MEMORY_MODEL
    
    if 'selected_embedding_model' not in st.session_state:
        st.session_state.selected_embedding_model = Config.EMBEDDING_MODEL
    
    if 'available_models' not in st.session_state:
        st.session_state.available_models = None
    
    if 'theme' not in st.session_state:
        st.session_state.theme = 'dark'


def load_components():
    """
    Load and initialize core components.
    
    Returns:
        Tuple of (vector_store, ollama_client, rag_graph)
    """
    try:
        # Initialize vector store with selected embedding model
        vector_store = VectorStore(
            embedding_model=st.session_state.selected_embedding_model
        )
        
        # Initialize Ollama client with selected generation model
        ollama_client = OllamaClient(
            model=st.session_state.selected_generation_model
        )
        
        # Check Ollama connection
        if not ollama_client.check_connection():
            st.error("Cannot connect to Ollama. Please ensure Ollama is running.")
            st.info(f"Expected Ollama URL: {Config.OLLAMA_BASE_URL}")
            st.stop()
        
        # Initialize Memory Model with selected memory model
        memory_model = MemoryModel(
            model_name=st.session_state.selected_memory_model
        )
        
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
    
    # Apply theme CSS first
    apply_custom_theme(st.session_state.theme)
    
    # Header with theme toggle
    col1, col2 = st.columns([11, 1])
    with col1:
        st.title("📚 Academic PDF Chat")
    with col2:
        st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
        theme_icon = "🌙" if st.session_state.theme == 'dark' else "☀️"
        if st.button(theme_icon, key="theme_toggle_btn", help="Toggle dark/light mode"):
            st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'
            st.rerun()
    
    st.markdown("Upload a PDF document and ask questions about its content using AI.")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Load components first if not loaded
        if st.session_state.ollama_client is None:
            vector_store, ollama_client, memory_model, rag_graph = load_components()
            st.session_state.vector_store = vector_store
            st.session_state.ollama_client = ollama_client
            st.session_state.memory_model = memory_model
            st.session_state.rag_graph = rag_graph
        
        # Model Selection with refresh button
        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown("### ⚙️ Model Settings")
        with col2:
            if st.button("🔄", help="Refresh model list", key="refresh_models", use_container_width=False):
                st.session_state.available_models = None
                st.rerun()
        
        # Get available models from Ollama
        if st.session_state.available_models is None:
            with st.spinner("Loading available models..."):
                st.session_state.available_models = st.session_state.ollama_client.get_models_detailed()
        
        models = st.session_state.available_models
        
        # Text Generation Model selector
        if models['text_generation']:
            current_gen_idx = 0
            if st.session_state.selected_generation_model in models['text_generation']:
                current_gen_idx = models['text_generation'].index(st.session_state.selected_generation_model)
            
            selected_gen = st.selectbox(
                "🤖 Generation Model",
                options=models['text_generation'],
                index=current_gen_idx,
                help="Model used for generating responses"
            )
            
            if selected_gen != st.session_state.selected_generation_model:
                st.session_state.selected_generation_model = selected_gen
                if st.session_state.ollama_client:
                    st.session_state.ollama_client.set_model(selected_gen)
                if st.session_state.rag_graph:
                    st.session_state.rag_graph.ollama_client.set_model(selected_gen)
                st.success(f"✓ Generation model changed to {selected_gen}")
        
        # Memory Model selector (for MemoRAG)
        if models['memory'] or models['text_generation']:
            memory_options = models['memory'] if models['memory'] else models['text_generation']
            current_mem_idx = 0
            if st.session_state.selected_memory_model in memory_options:
                current_mem_idx = memory_options.index(st.session_state.selected_memory_model)
            
            selected_mem = st.selectbox(
                "🧠 Memory Model",
                options=memory_options,
                index=current_mem_idx,
                help="Lightweight model for MemoRAG clue generation"
            )
            
            if selected_mem != st.session_state.selected_memory_model:
                st.session_state.selected_memory_model = selected_mem
                if st.session_state.memory_model and hasattr(st.session_state.memory_model, 'model_name'):
                    st.session_state.memory_model.model_name = selected_mem
                st.success(f"✓ Memory model changed to {selected_mem}")
        
        # Embedding Model selector
        if models['embedding']:
            current_emb_idx = 0
            if st.session_state.selected_embedding_model in models['embedding']:
                current_emb_idx = models['embedding'].index(st.session_state.selected_embedding_model)
            
            selected_emb = st.selectbox(
                "📊 Embedding Model",
                options=models['embedding'],
                index=current_emb_idx,
                help="Model used for document embeddings and search"
            )
            
            if selected_emb != st.session_state.selected_embedding_model:
                st.warning("⚠️ Changing embedding model requires reprocessing documents")
                st.session_state.selected_embedding_model = selected_emb
        
        st.divider()
        
        # MemoRAG toggle
        st.subheader("🔍 Retrieval Mode")
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
        
        st.divider()
        
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