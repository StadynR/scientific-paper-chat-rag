"""
Utility functions for the RAG application.
Provides helper functions for logging, error handling, and common operations.
"""

import logging
from typing import Optional
from pathlib import Path


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a logger instance.
    
    Args:
        name: Name of the logger
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def validate_pdf_file(file_path: Path) -> bool:
    """
    Validate that a file exists and is a PDF.
    
    Args:
        file_path: Path to the file to validate
        
    Returns:
        True if file is valid, False otherwise
    """
    if not file_path.exists():
        return False
    
    if not file_path.is_file():
        return False
    
    if file_path.suffix.lower() != '.pdf':
        return False
    
    return True


def format_sources(sources: list[dict]) -> str:
    """
    Format source citations for display.
    
    Args:
        sources: List of source dictionaries with page and content info
        
    Returns:
        Formatted string with source citations
    """
    if not sources:
        return "No sources found."
    
    formatted = []
    for i, source in enumerate(sources, 1):
        page = source.get('page', 'Unknown')
        content = source.get('content', '')[:200]  # Truncate to 200 chars
        formatted.append(f"**Source {i}** (Page {page}):\n{content}...")
    
    return "\n\n".join(formatted)


def truncate_text(text: str, max_length: int = 500) -> str:
    """
    Truncate text to a maximum length with ellipsis.
    
    Args:
        text: Text to truncate
        max_length: Maximum length of output
        
    Returns:
        Truncated text with ellipsis if needed
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."
