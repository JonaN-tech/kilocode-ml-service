from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Dict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("[ML][MEM]")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# Global singleton for lazy loading
_model = None
_indexes_cache = {}


class Embedder:
    """Lazy-loading embedder to prevent OOM on startup."""
    
    def _load(self):
        global _model
        if _model is None:
            try:
                logger.info("model_loading model=all-MiniLM-L6-v2")
                # Use smaller, faster model: ~40% less RAM than L12
                _model = SentenceTransformer(
                    "all-MiniLM-L6-v2",
                    device="cpu"
                )
                logger.info("model_loaded successfully")
            except Exception as e:
                logger.error(f"model_load_failed error={type(e).__name__}")
                raise

    def embed(self, texts: list[str]):
        """Lazy load model on first embed call."""
        self._load()
        try:
            # Ensure float32 only (no float64)
            embeddings = _model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"embed_failed error={type(e).__name__}")
            raise


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
        logger.info(f"index_cache_hit name={name}")
        return _indexes_cache[name]
    
    try:
        logger.info(f"index_loading name={name}")
        vectors = np.load(DATA_DIR / f"{name}_vectors.npy")
        # Ensure float32
        vectors = vectors.astype(np.float32)
        
        with open(DATA_DIR / f"{name}_meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        # Cache for future use
        _indexes_cache[name] = (vectors, meta)
        logger.info(f"index_loaded name={name} size={vectors.shape}")
        
        return vectors, meta
    except Exception as e:
        logger.error(f"index_load_failed name={name} error={type(e).__name__}")
        raise


def search_by_name(query: str, index_name: str, embedder: Embedder, top_k: int = 5):
    """Search index using cached vectors."""
    try:
        # Use cached index
        vectors, meta = load_index(index_name)
        
        q_vec = embedder.embed([query])
        scores = cosine_similarity(q_vec, vectors)[0]
        
        ranked = sorted(
            zip(scores, meta),
            key=lambda x: x[0],
            reverse=True
        )[:top_k]
        
        return [{"score": float(s), **m} for s, m in ranked]
    except Exception as e:
        logger.error(f"search_failed index={index_name} error={type(e).__name__}")
        # Return empty results instead of crashing
        return []
