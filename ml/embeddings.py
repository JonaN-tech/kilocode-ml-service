"""
Centralized embedding module with singleton pattern.

CRITICAL: This module ensures SentenceTransformer is loaded ONCE at startup,
preventing per-request RAM spikes that crash Render.
"""
from __future__ import annotations

import logging
import threading
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("[ML]")
mem_logger = logging.getLogger("[ML][MEM]")

# Global singleton
_embed_model = None
_lock = threading.Lock()


def get_embed_model():
    """
    Get or create the singleton embedding model.
    
    Thread-safe lazy initialization with double-checked locking.
    This should be called at startup to preload the model.
    """
    global _embed_model
    if _embed_model is None:
        with _lock:
            if _embed_model is None:
                mem_logger.info("model_loading model=all-MiniLM-L6-v2")
                
                _embed_model = SentenceTransformer(
                    "all-MiniLM-L6-v2",
                    device="cpu"
                )
                
                # Restrict sequence length for memory efficiency
                _embed_model.max_seq_length = 256
                
                # Reduce CPU thread usage to lower memory pressure
                try:
                    import torch
                    torch.set_num_threads(1)
                    mem_logger.info("torch_threads set_to=1")
                except Exception:
                    pass
                
                mem_logger.info("model_loaded max_seq_length=256")
    
    return _embed_model


def embed_texts(texts: List[str], batch_size: int = 2, normalize: bool = True) -> np.ndarray:
    """
    Embed texts using the singleton model.
    
    Args:
        texts: List of text strings to embed
        batch_size: Small batch size for memory safety (default: 2)
        normalize: Whether to normalize embeddings
    
    Returns:
        numpy array of embeddings (float32)
    """
    model = get_embed_model()
    
    try:
        mem_logger.info(f"embed_start texts={len(texts)} batch_size={batch_size}")
        
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=batch_size,
            normalize_embeddings=normalize
        )
        
        embeddings = embeddings.astype(np.float32)
        mem_logger.info(f"embed_complete shape={embeddings.shape}")
        
        return embeddings
        
    except Exception as e:
        mem_logger.error(f"embed_failed error={type(e).__name__}")
        raise


def embed_chunked(chunks: List[str], query: str, top_k: int = 3) -> List[Tuple[str, float]]:
    """
    Embed chunks incrementally and return top-k most relevant to query.
    
    This is memory-safe for large content:
    - Processes chunks in small batches
    - Returns only top-k chunks instead of all embeddings
    
    Args:
        chunks: List of text chunks
        query: Query text to compare against
        top_k: Number of top chunks to return
    
    Returns:
        List of (chunk_text, score) tuples, sorted by relevance
    """
    if not chunks:
        return []
    
    try:
        mem_logger.info(f"embed_chunked chunks={len(chunks)} top_k={top_k}")
        
        # Embed query first
        query_vec = embed_texts([query], batch_size=1, normalize=True)
        
        # Embed chunks in small batches
        batch_size = min(4, len(chunks))
        chunk_embeddings = embed_texts(chunks, batch_size=batch_size, normalize=True)
        
        # Compute similarities
        scores = cosine_similarity(query_vec, chunk_embeddings)[0]
        
        # Get top-k chunks
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = [(chunks[i], float(scores[i])) for i in top_indices]
        
        mem_logger.info(f"embed_chunked_complete top_scores={[s for _, s in results]}")
        
        return results
    
    except Exception as e:
        mem_logger.error(f"embed_chunked_failed error={type(e).__name__}")
        # Return empty instead of crashing
        return []