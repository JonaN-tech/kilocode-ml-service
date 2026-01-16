from typing import List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from retrieval import Embedder, load_index, search
from fetchers import fetch_post_content
from summarizer import summarize_text
from comment_engine import generate_comment

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


class GenerateCommentRequest(BaseModel):
    url: str
    platform: str
    tone: str = "professional"
    length: str = "short"


class GenerateCommentResponse(BaseModel):
    post_summary: str
    suggested_comment: str
    confidence: float


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
# Existing Context Endpoint
# =========================

@app.post("/ml/context", response_model=ContextResponse)
def get_context(req: ContextRequest):
    style_examples = []
    doc_facts = []

    if req.top_k_style > 0:
        style_examples = search(
            query=req.post_text,
            vectors=comment_vectors,
            meta=comment_meta,
            embedder=embedder,
            top_k=req.top_k_style,
        )

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


# =========================
# NEW: Comment Generation Endpoint
# =========================

@app.post("/ml/generate-comment", response_model=GenerateCommentResponse)
def generate_comment_endpoint(req: GenerateCommentRequest):
    try:
        # 1️⃣ Fetch post content from URL
        raw_text = fetch_post_content(req.url)

        # 2️⃣ Summarize post
        post_summary = summarize_text(raw_text)

        # 3️⃣ Retrieve relevant past comments (style)
        style_examples = search(
            query=post_summary,
            vectors=comment_vectors,
            meta=comment_meta,
            embedder=embedder,
            top_k=3,
        )

        # 4️⃣ Retrieve relevant documentation facts
        doc_facts = search(
            query=post_summary,
            vectors=doc_vectors,
            meta=doc_meta,
            embedder=embedder,
            top_k=3,
        )

        # 5️⃣ Generate final comment suggestion
        suggested_comment = generate_comment(
            post_summary=post_summary,
            style_examples=style_examples,
            doc_facts=doc_facts,
            tone=req.tone,
            length=req.length,
        )

        return GenerateCommentResponse(
            post_summary=post_summary,
            suggested_comment=suggested_comment,
            confidence=0.9,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
