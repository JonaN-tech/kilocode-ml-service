from typing import List, Dict
from fastapi import FastAPI
from pydantic import BaseModel

from retrieval import Embedder, load_index, search

app = FastAPI(title="KiloCode ML Context Service")


# =========================
# Request / Response Models
# =========================

class ContextRequest(BaseModel):
    post_text: str
    top_k_style: int = 5
    top_k_docs: int = 5


class ContextResponse(BaseModel):
    style_examples: List[Dict]
    doc_facts: List[Dict]


# =========================
# Startup (load models + data)
# =========================

@app.on_event("startup")
def startup():
    global embedder
    global comment_vectors, comment_meta
    global doc_vectors, doc_meta

    embedder = Embedder()

    # Load prebuilt indexes (npy + json)
    comment_vectors, comment_meta = load_index("comments")
    doc_vectors, doc_meta = load_index("docs")


# =========================
# API Endpoint
# =========================

@app.post("/ml/context", response_model=ContextResponse)
def get_context(req: ContextRequest):
    style_examples = []
    doc_facts = []

    # -------- Style examples (past comments)
    if req.top_k_style > 0:
        style_examples = search(
            query=req.post_text,
            vectors=comment_vectors,
            meta=comment_meta,
            embedder=embedder,
            top_k=req.top_k_style,
        )

    # -------- Documentation facts
    if req.top_k_docs > 0:
        doc_facts = search(
            query=req.post_text,
            vectors=doc_vectors,
            meta=doc_meta,
            embedder=embedder,
            top_k=req.top_k_docs,
        )

    return ContextResponse(
        style_examples=style_examples,
        doc_facts=doc_facts,
    )
