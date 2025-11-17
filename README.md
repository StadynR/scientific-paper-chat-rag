# Academic PDF Chat - RAG Application

A Retrieval-Augmented Generation (RAG) application for academic PDF analysis using Streamlit, LangGraph, and Ollama.

## Features

- PDF document upload and processing
- Intelligent text chunking with overlap
- Vector embeddings using Ollama (mxbai-embed-large)
- ChromaDB for efficient similarity search
- LangGraph-based RAG pipeline
- Ollama integration with DeepSeek-R1 model
- Real-time streaming responses
- Source citation with page numbers

## Prerequisites

- Python 3.8+
- Ollama installed locally with the following models:
  - `deepseek-r1` (LLM for generation)
  - `mxbai-embed-large` (for embeddings)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Final
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install and setup Ollama:
```bash
# Visit https://ollama.ai to download Ollama
# Pull required models
ollama pull deepseek-r1
ollama pull mxbai-embed-large
```

4. Create environment file (optional for customization):
```bash
cp .env.example .env
# Edit .env to adjust parameters like chunk size, temperature, etc.
```

## Local Usage

1. Start Ollama:
```bash
ollama serve
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser at `http://localhost:8501`

4. Upload a PDF and start chatting!

## Project Structure

```
Final/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
├── .gitignore             # Git ignore rules
├── README.md              # This file
├── IMPLEMENTATION.md      # Technical implementation details
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── utils.py           # Utility functions
│   ├── rag_classes.py     # RAG pipeline classes (PDFProcessor, VectorStore, RAGGraph)
│   └── ollama_client.py   # Ollama API client
├── data/
│   ├── pdfs/              # Uploaded PDF files (auto-created)
│   └── vectorstore/       # ChromaDB persistence (auto-created)
└── .streamlit/
    └── config.toml        # Streamlit configuration
```

## How It Works

### RAG Pipeline

1. **Document Processing**
   - PDF uploaded through Streamlit interface
   - Text extracted using PyMuPDF
   - Content split into chunks with overlap

2. **Embedding Generation**
   - Chunks converted to vector embeddings
   - Using Ollama's mxbai-embed-large model
   - Stored in ChromaDB with metadata

3. **Query Processing (LangGraph)**
   - **Retrieve Node**: 
     - User question embedded
     - Similarity search in vector store
     - Top-k relevant chunks retrieved
   - **Generate Node**:
     - Context built from retrieved chunks
     - Prompt constructed with context
     - Ollama generates answer
     - Response streamed to user

4. **Source Citation**
   - Retrieved chunks tracked with page numbers
   - Sources displayed with excerpts
   - Enables verification and transparency

## Configuration

### Environment Variables

All settings have sensible defaults and can be customized in `.env`:

- `OLLAMA_BASE_URL`: Ollama API endpoint (default: http://localhost:11434)
- `OLLAMA_MODEL`: LLM model name (fixed: deepseek-r1)
- `EMBEDDING_MODEL`: Embedding model (fixed: mxbai-embed-large)
- `CHUNK_SIZE`: Text chunk size in characters (default: 800)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 150)
- `TOP_K_DOCUMENTS`: Number of documents to retrieve (default: 6)
- `TEMPERATURE`: Generation temperature 0-1 (default: 0.3)
- `VECTOR_STORE_PATH`: Path to ChromaDB storage (default: ./data/vectorstore)
- `MAX_UPLOAD_SIZE_MB`: Maximum PDF file size (default: 50)

### Models Used

The application uses fixed, optimized models:

- **LLM**: `deepseek-r1` - Excellent reasoning capabilities and logical thinking (~4.7GB)
- **Embeddings**: `mxbai-embed-large` - High-quality semantic embeddings for retrieval

These models have been selected for optimal performance in academic document analysis.

## Usage Tips

1. **Document Quality**: Best results with text-based PDFs (not scanned images)

2. **Question Formulation**: Be specific and reference document terminology

3. **First Time Setup**:
   - Upload your PDF
   - Click "Process PDF"
   - Wait for processing (shows chunk count)
   - Start asking questions

4. **For Best Results**:
   - Ask specific questions with context
   - Check "View Sources" to verify information
   - Adjust temperature in `.env` if needed:
     - Lower (0.2-0.3) for technical/factual content
     - Higher (0.5-0.7) for creative responses

## 🚀 Optimizations Implemented

This application includes several optimizations for maximum precision:

- ✅ **Smaller chunks** (800 chars) for granular retrieval
- ✅ **Document re-ranking** with relevance scoring
- ✅ **Low temperature** (0.3) to reduce hallucinations
- ✅ **Advanced prompting** with explicit instructions
- ✅ **Source filtering** (similarity threshold: 0.3)
- ✅ **6 documents retrieved** with best-match selection
- ✅ **Optimized models** (DeepSeek-R1 for generation, mxbai-embed-large for embeddings)

Parameters can be tuned in `.env` for specific use cases.

## Performance Considerations

- **Embedding Generation**: First PDF upload takes longer (model download)
- **Vector Search**: Sub-second for typical document sizes
- **LLM Generation**: Depends on model size and hardware
  - Small models (7B): 20-50 tokens/sec
  - Large models (13B+): 10-20 tokens/sec

## Limitations

- PDF must be text-based (OCR not included)
- Single document at a time
- Local Ollama requires sufficient RAM (~8GB recommended)
- Works best with academic/technical documents

## Credits

Built with:
- [Streamlit](https://streamlit.io/)
- [LangChain](https://python.langchain.com/)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Ollama](https://ollama.ai/)
- [ChromaDB](https://www.trychroma.com/)

## Support

For issues and questions, please open an issue in the repository.
