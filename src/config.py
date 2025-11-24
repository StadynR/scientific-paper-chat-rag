"""
Configuration management module.
Handles loading and accessing environment variables and application settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Central configuration class that manages all application settings.
    Loads values from environment variables with sensible defaults.
    """
    
    # Ollama Configuration
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1")
    
    # Memory Model Configuration (MemoRAG)
    MEMORY_MODEL = os.getenv("MEMORY_MODEL", "llama3.2")
    MEMORY_TEMPERATURE = float(os.getenv("MEMORY_TEMPERATURE", "0.5"))
    NUM_CLUES = int(os.getenv("NUM_CLUES", "3"))  # Number of clues to generate per query
    
    # Embedding Model Configuration (Ollama model)
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")
    
    # Chunk Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
    
    # Retrieval Configuration
    TOP_K_DOCUMENTS = int(os.getenv("TOP_K_DOCUMENTS", "6"))
    
    # Generation Configuration
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
    
    # Vector Store Configuration
    VECTOR_STORE_PATH = Path(os.getenv("VECTOR_STORE_PATH", "./data/vectorstore"))
    
    # Memory Store Configuration (MemoRAG)
    MEMORY_STORE_PATH = Path(os.getenv("MEMORY_STORE_PATH", "./data/memory"))
    
    # Application Configuration
    MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    PDFS_DIR = DATA_DIR / "pdfs"
    
    @classmethod
    def ensure_directories(cls):
        """
        Create necessary directories if they don't exist.
        """
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.PDFS_DIR.mkdir(exist_ok=True)
        cls.VECTOR_STORE_PATH.mkdir(exist_ok=True, parents=True)
        cls.MEMORY_STORE_PATH.mkdir(exist_ok=True, parents=True)


# Ensure directories exist on module import
Config.ensure_directories()
