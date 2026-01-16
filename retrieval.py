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
    _model = None

    def _load(self):
        if self._model is None:
            self._model = SentenceTransformer(
                "all-MiniLM-L12-v2",
                device="cpu"
            )

    def embed(self, texts: list[str]):
        self._load()
        return self._model.encode(texts, convert_to_numpy=True)


def save_index(vectors: np.ndarray, meta: List[Dict], name: str):
    np.save(DATA_DIR / f"{name}_vectors.npy", vectors)
    with open(DATA_DIR / f"{name}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_index(name: str):
    vectors = np.load(DATA_DIR / f"{name}_vectors.npy")
    with open(DATA_DIR / f"{name}_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return vectors, meta


# retrieval.py
def search_by_name(query: str, index_name: str, embedder: Embedder, top_k: int = 5):
    vectors = np.load(DATA_DIR / f"{index_name}_vectors.npy")
    with open(DATA_DIR / f"{index_name}_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    q_vec = embedder.embed([query])
    scores = cosine_similarity(q_vec, vectors)[0]

    ranked = sorted(
        zip(scores, meta),
        key=lambda x: x[0],
        reverse=True
    )[:top_k]

    return [{"score": float(s), **m} for s, m in ranked]

