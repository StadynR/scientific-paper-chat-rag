# Academic PDF Chat - RAG Application

A Retrieval-Augmented Generation (RAG) application for academic PDF analysis using Streamlit, LangGraph, and Ollama.

## Features

- 📄 PDF document upload and processing
- ✂️ Intelligent text chunking with overlap
- 🔢 Vector embeddings using Ollama (mxbai-embed-large)
- 💾 ChromaDB for efficient similarity search
- 🔄 LangGraph-based RAG pipeline
- 🤖 Ollama integration with DeepSeek-R1 model
- ⚡ Real-time streaming responses with animated loading states
- 📖 References appear directly in the text (e.g., "The Transformer uses attention mechanisms [pg. 1]")
- 📚 Source documents with excerpts and page numbers

## Prerequisites

- Python 3.10+
- Ollama

## Installation

1. Clone the repository:
```bash
git clone https://github.com/StadynR/scientific-paper-chat-rag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install and setup [https://ollama.ai](Ollama) (Example with recommended models):
```bash
ollama pull deepseek-r1
ollama pull mxbai-embed-large
```

4. Create environment file (optional for customization):
```bash
cp .env.example .env
# Edit .env to adjust parameters like chunk size, temperature, etc.
```

## Usage

1. **Start Ollama** (if not already running):
```bash
ollama serve
```

2. **Run the Streamlit app**:
```bash
streamlit run app.py
```

3. **Open your browser** at `http://localhost:8501`

4. **Upload a PDF** and start chatting!

### Using the Application

https://github.com/user-attachments/assets/e8fe6a1a-3778-441f-881d-7e5c19e6c16e

## How It Works

### RAG Pipeline

1. **Document Processing**
   - PDF uploaded through Streamlit interface
   - Text extracted using PyMuPDF (fitz)
   - Content split into chunks with configurable size and overlap

2. **Embedding Generation**
   - Chunks converted to vector embeddings
   - Using Ollama's mxbai-embed-large model
   - Stored in ChromaDB with metadata (page numbers, source, chunk IDs)

3. **Query Processing (LangGraph)**
   - **Retrieve Node**: 
     - User question embedded using same model
     - Cosine similarity search in vector store
     - Top-k relevant chunks retrieved with re-ranking
     - Score threshold filtering for quality
   - **Generate Node**:
     - Context built from retrieved chunks with page metadata
     - Prompt constructed with inline citation instructions
     - DeepSeek-R1 generates answer using sources
     - Response streamed token-by-token to user

4. **Inline Citations & Sources**
   - Model instructed to cite pages inline as it generates
   - Citations appear in format: `[pg. 1]`, `[pg. 3, 7]`
   - Full source documents accessible via expandable sections
   - Page numbers enable easy verification

## Configuration

### Environment Variables

All settings have sensible defaults and can be customized in `.env`:

- `OLLAMA_BASE_URL`: Ollama API endpoint (default: http://localhost:11434)
- `OLLAMA_MODEL`: LLM model name (e.g.: deepseek-r1)
- `EMBEDDING_MODEL`: Embedding model (e.g.: mxbai-embed-large)
- `CHUNK_SIZE`: Text chunk size in characters (default: 800)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 150)
- `TOP_K_DOCUMENTS`: Number of documents to retrieve (default: 6)
- `TEMPERATURE`: Generation temperature 0-1 (default: 0.3)
- `VECTOR_STORE_PATH`: Path to ChromaDB storage (default: ./data/vectorstore)
- `MAX_UPLOAD_SIZE_MB`: Maximum PDF file size in MB (default: 50)

## Limitations

- ⚠️ PDF must be text-based (OCR not included for scanned documents)
- 📄 Single document at a time (clears previous when uploading new)
- 💻 Local Ollama requires sufficient RAM (~8GB minimum, 16GB recommended)
- 🎯 Optimized for academic/technical documents (may struggle with highly visual content)
- ⏱️ First-token latency depends on context size and model processing time
- 🌐 No built-in ngrok/cloud deployment (local deployment only)

## Troubleshooting

### Slow Response Times
- Reduce `TOP_K_DOCUMENTS` in `.env` (try 4 instead of 6)
- Use smaller chunk sizes for faster retrieval
- Ensure no other heavy processes are running
- Check available RAM (close unnecessary applications)

### Empty or No Response
- Check Ollama logs: `ollama logs`
- Verify PDF has extractable text (not scanned image)
- Try a smaller/simpler question first
- Check console/terminal for error messages

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the GPL-3 License - see the [LICENSE](LICENSE) file for details.

## Support

For issues and questions, please open an issue in the repository.
