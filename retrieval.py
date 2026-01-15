from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        return np.array(self.model.encode(texts, normalize_embeddings=True))


def save_index(vectors: np.ndarray, meta: List[Dict], name: str):
    np.save(DATA_DIR / f"{name}_vectors.npy", vectors)
    with open(DATA_DIR / f"{name}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_index(name: str):
    vectors = np.load(DATA_DIR / f"{name}_vectors.npy")
    with open(DATA_DIR / f"{name}_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return vectors, meta


def search(query: str, vectors: np.ndarray, meta: List[Dict], embedder: Embedder, top_k: int = 5):
    q_vec = embedder.embed([query])
    scores = cosine_similarity(q_vec, vectors)[0]

    ranked = sorted(
        zip(scores, meta),
        key=lambda x: x[0],
        reverse=True
    )[:top_k]

    return [
        {
            "score": float(score),
            **item
        }
        for score, item in ranked
    ]
