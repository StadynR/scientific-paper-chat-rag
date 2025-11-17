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

## Remote Access with Ngrok

For sharing the application online, use ngrok to create a secure public tunnel:

### Prerequisites for Remote Access

- Ngrok account (free tier works fine)
- Ngrok authtoken configured
- Ollama running locally with required models

### Quick Start - One Command Setup

1. **Install ngrok**

https://ngrok.com/download

1. **Configure ngrok authtoken** (one time only):
```bash
ngrok config add-authtoken YOUR_AUTHTOKEN_HERE
```

1. **Run everything with one command**:
```bash
python run_with_ngrok.py
```

The script will automatically:
- Verify Ollama is running
- Start Streamlit app locally
- Create ngrok tunnel for public access
- Display the public URL to share
- Keep everything running until you stop it

**Note**: Your computer needs to be on while the script is running

### Manual Setup (Alternative)

If you prefer manual control:

1. **Start Ollama tunnel**:
```bash
ngrok http 11434
```

2. **Update Ollama URL** in your `.env` or Streamlit app:
```bash
# Use the ngrok URL shown in terminal
OLLAMA_BASE_URL=https://your-ngrok-url.ngrok.io
```

3. **Run Streamlit**:
```bash
streamlit run app.py
```

4. **Access remotely**: Use the Streamlit URL (typically `http://localhost:8501`) or create another ngrok tunnel for Streamlit if needed

### Important Notes for Cloud Deployment

**Security Considerations**:
- Ngrok free tier URLs are public - anyone with the URL can access
- Consider using ngrok's authentication features for sensitive documents
- Don't share URLs publicly if processing confidential PDFs
- Free ngrok tunnels expire after inactivity

**Performance**:
- Remote access adds latency to Ollama responses
- LLM generation will be slower over the internet
- Embedding generation is less affected by network latency
- Best suited for demonstration/testing, not production use

**Resource Usage**:
- Ollama still runs on your local machine
- Your computer must stay on and connected
- Sufficient RAM required (~8GB minimum for models)

### Troubleshooting Cloud Deployment

**Ollama connection errors**:
- Verify ngrok tunnel is active: check ngrok dashboard
- Ensure Ollama is running: `curl http://localhost:11434/api/tags`
- Check firewall settings aren't blocking connections

**Slow responses**:
- Expected due to network overhead
- Consider using smaller models for faster inference
- Check your internet upload speed (affects response streaming)

**Ngrok tunnel disconnects**:
- Free tier has session limits
- Restart with `python setup_tunnel.py`
- Consider ngrok paid plans for stable connections

## Project Structure

```
Final/
├── app.py                  # Main Streamlit application
├── setup_tunnel.py         # Automated ngrok tunnel setup script
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
