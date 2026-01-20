from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ml.embeddings import embed_texts, embed_chunked

logger = logging.getLogger("[ML]")
mem_logger = logging.getLogger("[ML][MEM]")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# Global cache for indexes
_indexes_cache = {}


class Embedder:
    """
    Embedder wrapper using centralized ml.embeddings module.
    
    This no longer loads models - it delegates to the singleton in ml/embeddings.py
    which is preloaded at startup to prevent per-request RAM spikes.
    """
    
    def embed(self, texts: list[str], batch_size: int = 2, normalize: bool = True):
        """
        Embed texts using centralized singleton model.
        
        Args:
            texts: List of text strings to embed
            batch_size: Small batch size for memory safety (default: 2)
            normalize: Whether to normalize embeddings
        
        Returns:
            numpy array of embeddings (float32)
        """
        return embed_texts(texts, batch_size=batch_size, normalize=normalize)
    
    def embed_chunked(self, chunks: List[str], query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Embed chunks and return top-k most relevant to query.
        
        Delegates to centralized ml.embeddings module.
        
        Args:
            chunks: List of text chunks
            query: Query text to compare against
            top_k: Number of top chunks to return
        
        Returns:
            List of (chunk_text, score) tuples, sorted by relevance
        """
        return embed_chunked(chunks, query, top_k)


def save_index(vectors: np.ndarray, meta: List[Dict], name: str):
    """Save index vectors and metadata to disk."""
    # Ensure float32 for memory efficiency
    vectors = vectors.astype(np.float32)
    np.save(DATA_DIR / f"{name}_vectors.npy", vectors)
    with open(DATA_DIR / f"{name}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_index(name: str):
    """Load index from disk with caching to prevent reloading."""
    global _indexes_cache
    
    # Check cache first
    if name in _indexes_cache:
        mem_logger.info(f"index_cache_hit name={name}")
        return _indexes_cache[name]
    
    try:
        mem_logger.info(f"index_loading name={name}")
        vectors = np.load(DATA_DIR / f"{name}_vectors.npy")
        # Ensure float32
        vectors = vectors.astype(np.float32)
        
        with open(DATA_DIR / f"{name}_meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        # Cache for future use
        _indexes_cache[name] = (vectors, meta)
        mem_logger.info(f"index_loaded name={name} size={vectors.shape}")
        
        return vectors, meta
    except Exception as e:
        mem_logger.error(f"index_load_failed name={name} error={type(e).__name__}")
        raise


def search_by_name(query: str, index_name: str, embedder: Embedder, top_k: int = 5):
    """Search index using cached vectors."""
    try:
        # Use cached index
        vectors, meta = load_index(index_name)
        
        q_vec = embedder.embed([query], batch_size=1, normalize=True)
        scores = cosine_similarity(q_vec, vectors)[0]
        
        ranked = sorted(
            zip(scores, meta),
            key=lambda x: x[0],
            reverse=True
        )[:top_k]
        
        return [{"score": float(s), **m} for s, m in ranked]
    except Exception as e:
        mem_logger.error(f"search_failed index={index_name} error={type(e).__name__}")
        # Return empty results instead of crashing
        return []
