from pathlib import Path
from typing import List, Dict

from fastapi import FastAPI
from pydantic import BaseModel

from retrieval import Embedder, load_index, search, load_jsonl

BASE = Path(__file__).resolve().parent
INDEXES = BASE / "indexes"

app = FastAPI(title="KiloCode ML Context Service")


class ContextRequest(BaseModel):
    post_text: str
    top_k_style: int = 5
    top_k_docs: int = 5


class ContextResponse(BaseModel):
    style_examples: List[Dict]
    doc_facts: List[Dict]


@app.on_event("startup")
def startup():
    global embedder, style_index, docs_index, style_meta, docs_meta
    embedder = Embedder()
    style_index = load_index(INDEXES / "style.faiss")
    docs_index = load_index(INDEXES / "docs.faiss")
    style_meta = load_jsonl(INDEXES / "style_meta.jsonl")
    docs_meta = load_jsonl(INDEXES / "docs_meta.jsonl")


@app.post("/ml/context", response_model=ContextResponse)
def get_context(req: ContextRequest):
    q = embedder.embed([req.post_text])[0]

    style_examples = []
    if req.top_k_style > 0:
        scores, idxs = search(style_index, q, req.top_k_style)
        for s, i in zip(scores, idxs):
            style_examples.append({**style_meta[int(i)], "score": float(s)})

    scores, idxs = search(docs_index, q, req.top_k_docs)
    doc_facts = [{**docs_meta[int(i)], "score": float(s)} for s, i in zip(scores, idxs)]

    return ContextResponse(style_examples=style_examples, doc_facts=doc_facts)
