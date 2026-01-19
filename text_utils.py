"""
Text processing utilities for safe, memory-efficient handling of large content.
"""
import re
import logging
from typing import List, Tuple

logger = logging.getLogger("[ML]")


def clean_text(text: str, max_length: int = 25000) -> str:
    """
    Clean and normalize text for safe processing.
    
    Args:
        text: Raw text (may contain HTML, excessive whitespace, etc.)
        max_length: Maximum character length after cleaning
    
    Returns:
        Cleaned text string
    """
    if not text:
        return ""
    
    # Remove excessive whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common noise patterns
    text = re.sub(r'http[s]?://\S+', '[LINK]', text)  # Replace URLs with placeholder
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII
    
    # Truncate huge code blocks (but keep some context)
    # Match code blocks that are very long
    text = re.sub(r'(```[\s\S]{500,}?```)', '[CODE_BLOCK]', text)
    
    # Final cleanup
    text = text.strip()
    
    # Hard cap on length
    if len(text) > max_length:
        logger.warning(f"text_truncated original_len={len(text)} max_len={max_length}")
        text = text[:max_length]
    
    return text


def chunk_text(text: str, chunk_chars: int = 1000, overlap: int = 150, max_chunks: int = 12) -> List[str]:
    """
    Split text into overlapping chunks for incremental processing.
    
    Args:
        text: Cleaned text to chunk
        chunk_chars: Target characters per chunk
        overlap: Character overlap between chunks
        max_chunks: Maximum number of chunks to return
    
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    if len(text) <= chunk_chars:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text) and len(chunks) < max_chunks:
        end = start + chunk_chars
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence ending (.!?) in the last 100 chars
            chunk_end = text[start:end]
            last_period = max(chunk_end.rfind('.'), chunk_end.rfind('!'), chunk_end.rfind('?'))
            
            if last_period > chunk_chars - 100:  # Found a good break point
                end = start + last_period + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move to next chunk with overlap
        start = end - overlap
        
        if start >= len(text):
            break
    
    logger.info(f"text_chunked total_len={len(text)} num_chunks={len(chunks)} chunk_size={chunk_chars}")
    
    return chunks


def extract_title_from_text(text: str, max_length: int = 150) -> str:
    """
    Extract a title from the beginning of text content.
    Useful when no explicit title is provided.
    
    Args:
        text: Content text
        max_length: Maximum title length
    
    Returns:
        Extracted title or empty string
    """
    if not text:
        return ""
    
    # Take first line/sentence
    first_line = text.split('\n')[0].strip()
    
    # If it's too long, take first sentence
    if len(first_line) > max_length:
        # Find first sentence ending
        match = re.search(r'[.!?]', first_line)
        if match:
            first_line = first_line[:match.end()].strip()
        else:
            first_line = first_line[:max_length].strip() + "..."
    
    return first_line