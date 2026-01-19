from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("[ML]")
mem_logger = logging.getLogger("[ML][MEM]")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# Global singleton for lazy loading
_model = None
_indexes_cache = {}


class Embedder:
    """Lazy-loading embedder with memory-safe chunked processing."""
    
    def _load(self):
        global _model
        if _model is None:
            try:
                mem_logger.info("model_loading model=all-MiniLM-L6-v2")
                
                # Use smaller, faster model: ~40% less RAM than L12
                _model = SentenceTransformer(
                    "all-MiniLM-L6-v2",
                    device="cpu"
                )
                
                # Restrict sequence length for memory efficiency
                _model.max_seq_length = 256  # Reduce from default 384
                
                # Reduce CPU thread usage to lower memory pressure
                try:
                    import torch
                    torch.set_num_threads(1)
                    mem_logger.info("torch_threads set_to=1")
                except Exception:
                    pass
                
                mem_logger.info("model_loaded max_seq_length=256")
            except Exception as e:
                mem_logger.error(f"model_load_failed error={type(e).__name__}")
                raise

    def embed(self, texts: list[str], batch_size: int = 2, normalize: bool = True):
        """
        Lazy load model and embed texts with memory-safe batching.
        
        Args:
            texts: List of text strings to embed
            batch_size: Small batch size for memory safety (default: 2)
            normalize: Whether to normalize embeddings
        
        Returns:
            numpy array of embeddings (float32)
        """
        self._load()
        try:
            mem_logger.info(f"embed_start texts={len(texts)} batch_size={batch_size}")
            
            # Ensure float32 only (no float64)
            embeddings = _model.encode(
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
    
    def embed_chunked(self, chunks: List[str], query: str, top_k: int = 3) -> List[Tuple[str, float]]:
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
            query_vec = self.embed([query], batch_size=1, normalize=True)
            
            # Embed chunks in small batches
            batch_size = min(4, len(chunks))
            chunk_embeddings = self.embed(chunks, batch_size=batch_size, normalize=True)
            
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
