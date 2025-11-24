# Academic PDF Chat - RAG Application

A Retrieval-Augmented Generation (RAG) application for academic PDF analysis using Streamlit, LangGraph, and Ollama.

## Features

- 📄 PDF document upload and processing
- ✂️ Intelligent text chunking with overlap
- 🔢 Vector embeddings using Ollama
- 💾 ChromaDB for efficient similarity search
- 🔄 LangGraph-based RAG pipeline
- 🧠 **MemoRAG**: Memory-augmented retrieval with clue generation
- 🎚️ **Toggle MemoRAG**: Compare standard RAG vs memory-enhanced retrieval
- 🎨 **Dark/Light Theme**: Switch between color schemes
- 🔧 **Dynamic Model Selection**: Choose generation, memory, and embedding models from UI
- 🤖 Ollama integration (deepseek-r1, llama3.2, mxbai-embed-large)
- ⚡ Real-time streaming responses with animated loading states
- 📖 References appear directly in the text (e.g., "The Transformer uses attention mechanisms [pg. 1]")
- 📚 Source documents with excerpts and page numbers
- 🔍 Clue visualization for MemoRAG queries

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

3. Install and setup [Ollama](https://ollama.ai) (Example with recommended models):
```bash
ollama pull deepseek-r1
ollama pull llama3.2
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
   - Using Ollama embedding model (configurable in UI)
   - Stored in ChromaDB with metadata (page numbers, source, chunk IDs)

3. **MemoRAG Memory Building** (when enabled)
   - **Document Compression**: Lightweight memory model (llama3.2) summarizes document
   - **Page Summaries**: Each page condensed to key points
   - **Global Summary**: Overall document overview
   - **Key Topics**: Extracted themes and concepts
   - **Persistence**: Memory stored in JSON format for reuse

4. **Query Processing (LangGraph)**
   
   **Standard RAG Mode:**
   - **Retrieve Node**: 
     - User question embedded using same model
     - Cosine similarity search in vector store
     - Top-k relevant chunks retrieved
   - **Generate Node**:
     - Context built from retrieved chunks
     - Generation model creates answer with inline citations
   
   **MemoRAG Mode:**
   - **Generate Clues Node**:
     - Memory model generates 3 search clues from query + document memory
     - Clues expand query coverage and context
   - **Retrieve Node**:
     - Multi-query search using all clues
     - Documents matched by multiple clues get boosted scores
     - Top-k chunks selected from enhanced results
   - **Generate Node**:
     - Context built with enriched retrieval results
     - Generation model creates answer with inline citations

5. **Inline Citations & Sources**
   - Model instructed to cite pages inline as it generates
   - Citations appear in format: `[pg. 1]`, `[pg. 3, 7]`
   - Full source documents accessible via expandable sections
   - Page numbers enable easy verification
   - MemoRAG mode shows generated clues in expandable section

## Configuration

### Environment Variables

All settings have sensible defaults and can be customized in `.env`:

**Ollama Configuration:**
- `OLLAMA_BASE_URL`: Ollama API endpoint (default: http://localhost:11434)
- `OLLAMA_MODEL`: Default generation model (e.g.: deepseek-r1)
- `EMBEDDING_MODEL`: Default embedding model (e.g.: mxbai-embed-large)

**MemoRAG Configuration:**
- `MEMORY_MODEL`: Memory model for clue generation (default: llama3.2)
- `MEMORY_TEMPERATURE`: Temperature for memory model (default: 0.5)
- `NUM_CLUES`: Number of search clues to generate (default: 3)
- `MEMORY_STORE_PATH`: Path to memory storage (default: ./data/memory)

**Document Processing:**
- `CHUNK_SIZE`: Text chunk size in characters (default: 800)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 150)
- `TOP_K_DOCUMENTS`: Number of documents to retrieve (default: 6)
- `MAX_UPLOAD_SIZE_MB`: Maximum PDF file size in MB (default: 50)

**Generation:**
- `TEMPERATURE`: Generation temperature 0-1 (default: 0.3)

**Storage:**
- `VECTOR_STORE_PATH`: Path to ChromaDB storage (default: ./data/vectorstore)

**Note:** Models can be changed dynamically from the UI without modifying `.env`

## Limitations

- ⚠️ PDF must be text-based (OCR not included for scanned documents)
- 📄 Single document at a time (clears previous when uploading new)
- 💻 Local Ollama requires sufficient RAM (~8GB minimum, 16GB+ recommended for MemoRAG)
- 🎯 Optimized for academic/technical documents (may struggle with highly visual content)
- ⏱️ First-token latency depends on context size and model processing time
- 🧠 MemoRAG memory building adds initial processing time (runs once per document)
- 🌐 No built-in ngrok/cloud deployment (local deployment only)

## Troubleshooting

### Slow Response Times
- Disable MemoRAG toggle for faster queries (skips clue generation)
- Reduce `TOP_K_DOCUMENTS` in `.env` (try 4 instead of 6)
- Use smaller chunk sizes for faster retrieval
- Switch to lighter models in UI (e.g., qwen2.5 instead of deepseek-r1)
- Ensure no other heavy processes are running
- Check available RAM (close unnecessary applications)

### Empty or No Response
- Check Ollama logs: `ollama logs`
- Verify PDF has extractable text (not scanned image)
- Try a smaller/simpler question first
- Check console/terminal for error messages

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Credits & Acknowledgments

This project builds upon and integrates the following open-source technologies:

- **[MemoRAG](https://github.com/qhjqhj00/MemoRAG)**: Memory-augmented retrieval-augmented generation system with clue-guided search. Implementation inspired by the research paper and architecture.
- **[Ollama](https://ollama.ai)**: Local LLM inference server enabling self-hosted AI capabilities.
- **[LangChain](https://github.com/langchain-ai/langchain)**: Framework for LLM application development.
- **[LangGraph](https://github.com/langchain-ai/langgraph)**: Graph-based orchestration for agentic workflows.
- **[ChromaDB](https://github.com/chroma-core/chroma)**: Open-source embedding database for vector storage and retrieval.
- **[Streamlit](https://streamlit.io)**: Web application framework for data science and ML projects.

Special thanks to the MemoRAG team for their innovative approach to memory-augmented retrieval systems.

## License

This project is licensed under the GPL-3 License - see the [LICENSE](LICENSE) file for details.

## Support

For issues and questions, please open an issue in the repository.
