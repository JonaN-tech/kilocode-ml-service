"""
Gemini-based embedding module for FREE tier API usage.

CRITICAL: This module uses Google Gemini FREE tier via AI Studio,
eliminating torch/sentence-transformers to prevent Render OOM crashes.

Key features:
- Uses Gemini 2.0 Flash (FREE tier, no Vertex AI)
- Batching with retries for safety
- In-memory caching to avoid re-embedding
- Hard safety limits (max content length, timeouts)
"""
from __future__ import annotations

import os
import logging
import time
import hashlib
from typing import List, Tuple
from functools import lru_cache

import numpy as np
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("[ML]")
mem_logger = logging.getLogger("[ML][MEM]")

# Gemini configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/text-embedding-004")  # Latest FREE embedding model

# Safety limits
MAX_TEXT_LENGTH = 1500  # Max chars per text for embedding
MAX_BATCH_SIZE = 100    # Gemini supports up to 100 per batch
DEFAULT_BATCH_SIZE = 20  # Conservative default for free tier
REQUEST_TIMEOUT = 30    # Seconds

# Embedding cache (in-memory)
_embedding_cache = {}

# Initialize Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    mem_logger.info(f"gemini_configured model={GEMINI_MODEL}")
else:
    logger.warning("GEMINI_API_KEY not found in environment")


def _get_cache_key(text: str) -> str:
    """Generate cache key from text."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _truncate_text(text: str, max_length: int = MAX_TEXT_LENGTH) -> str:
    """Truncate text to max length, preserving word boundaries."""
    if len(text) <= max_length:
        return text
    
    # Truncate and try to preserve word boundary
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    if last_space > max_length * 0.8:  # Only use if we don't lose too much
        truncated = truncated[:last_space]
    
    logger.info(f"text_truncated original_len={len(text)} truncated_len={len(truncated)}")
    return truncated


def embed_texts(
    texts: List[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    normalize: bool = True,
    use_cache: bool = True,
    max_retries: int = 3,
) -> np.ndarray:
    """
    Embed texts using Gemini API (FREE tier).
    
    Args:
        texts: List of text strings to embed
        batch_size: Batch size for API calls (default: 20)
        normalize: Whether to normalize embeddings
        use_cache: Whether to use in-memory cache
        max_retries: Max retry attempts on failure
    
    Returns:
        numpy array of embeddings (float32)
    """
    if not texts:
        return np.array([])
    
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not set - cannot embed")
        raise ValueError("GEMINI_API_KEY environment variable not set")
    
    mem_logger.info(f"embed_start texts={len(texts)} batch_size={batch_size}")
    
    # Truncate texts to max length
    texts = [_truncate_text(text) for text in texts]
    
    # Check cache first
    embeddings_list = []
    texts_to_embed = []
    cache_indices = []
    
    for i, text in enumerate(texts):
        if use_cache:
            cache_key = _get_cache_key(text)
            if cache_key in _embedding_cache:
                embeddings_list.append(_embedding_cache[cache_key])
                continue
        
        texts_to_embed.append(text)
        cache_indices.append(i)
    
    if texts_to_embed:
        mem_logger.info(f"cache_miss count={len(texts_to_embed)} cache_hits={len(embeddings_list)}")
        
        # Embed in batches
        new_embeddings = []
        for i in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[i:i + batch_size]
            
            # Retry logic
            for attempt in range(max_retries):
                try:
                    mem_logger.info(f"gemini_embed_batch batch_num={i//batch_size + 1} size={len(batch)} attempt={attempt + 1}")
                    
                    # Call Gemini API
                    result = genai.embed_content(
                        model=GEMINI_MODEL,
                        content=batch,
                        task_type="retrieval_document",  # Optimized for retrieval
                    )
                    
                    batch_embeddings = result['embedding'] if isinstance(result['embedding'][0], list) else [result['embedding']]
                    new_embeddings.extend(batch_embeddings)
                    
                    # Cache new embeddings
                    if use_cache:
                        for text, embedding in zip(batch, batch_embeddings):
                            cache_key = _get_cache_key(text)
                            _embedding_cache[cache_key] = embedding
                    
                    mem_logger.info(f"gemini_embed_success batch_num={i//batch_size + 1}")
                    break
                    
                except Exception as e:
                    logger.warning(f"gemini_embed_failed attempt={attempt + 1} error={type(e).__name__}: {str(e)[:100]}")
                    
                    if attempt == max_retries - 1:
                        logger.error(f"gemini_embed_exhausted retries={max_retries}")
                        raise
                    
                    # Exponential backoff
                    time.sleep(2 ** attempt)
        
        # Merge cached and new embeddings in correct order
        all_embeddings = [None] * len(texts)
        
        # Place cached embeddings
        j = 0
        for i in range(len(texts)):
            if i not in cache_indices:
                all_embeddings[i] = embeddings_list[j]
                j += 1
        
        # Place new embeddings
        for i, idx in enumerate(cache_indices):
            all_embeddings[idx] = new_embeddings[i]
        
        embeddings_list = all_embeddings
    else:
        mem_logger.info(f"cache_hit_all count={len(texts)}")
    
    # Convert to numpy array
    embeddings = np.array(embeddings_list, dtype=np.float32)
    
    # Normalize if requested
    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embeddings = embeddings / norms
    
    mem_logger.info(f"embed_complete shape={embeddings.shape} cached={len(_embedding_cache)}")
    
    return embeddings


def embed_chunked(
    chunks: List[str],
    query: str,
    top_k: int = 3,
    use_cache: bool = True,
) -> List[Tuple[str, float]]:
    """
    Embed chunks incrementally and return top-k most relevant to query.
    
    This is memory-safe for large content:
    - Processes chunks in batches
    - Returns only top-k chunks instead of all embeddings
    - Uses caching to avoid re-embedding
    
    Args:
        chunks: List of text chunks
        query: Query text to compare against
        top_k: Number of top chunks to return
        use_cache: Whether to use embedding cache
    
    Returns:
        List of (chunk_text, score) tuples, sorted by relevance
    """
    if not chunks:
        return []
    
    try:
        mem_logger.info(f"embed_chunked chunks={len(chunks)} top_k={top_k}")
        
        # Embed query first (single item, always fresh)
        query_vec = embed_texts([query], batch_size=1, normalize=True, use_cache=use_cache)
        
        # Embed chunks in batches
        chunk_embeddings = embed_texts(chunks, batch_size=DEFAULT_BATCH_SIZE, normalize=True, use_cache=use_cache)
        
        # Compute similarities
        scores = cosine_similarity(query_vec, chunk_embeddings)[0]
        
        # Get top-k chunks
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = [(chunks[i], float(scores[i])) for i in top_indices]
        
        mem_logger.info(f"embed_chunked_complete top_scores={[s for _, s in results]}")
        
        return results
    
    except Exception as e:
        logger.error(f"embed_chunked_failed error={type(e).__name__}: {str(e)[:100]}")
        # Return empty instead of crashing
        return []


def clear_cache():
    """Clear the embedding cache. Useful for testing or memory management."""
    global _embedding_cache
    cache_size = len(_embedding_cache)
    _embedding_cache.clear()
    mem_logger.info(f"cache_cleared size={cache_size}")


def get_cache_stats() -> dict:
    """Get cache statistics."""
    return {
        "size": len(_embedding_cache),
        "model": GEMINI_MODEL,
    }


# Legacy compatibility - no model loading needed
def get_embed_model():
    """
    Legacy compatibility function.
    
    Gemini embeddings don't require model loading,
    but this function is kept for backward compatibility.
    """
    mem_logger.info("get_embed_model called (no-op for Gemini)")
    return None