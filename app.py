from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Optional, Literal, Any, List

from retrieval import Embedder, load_index
from fetchers import fetch_post_content
from comment_engine import generate_comment

Platform = Literal[
    "reddit",
    "twitter",
    "github",
    "hackernews",
    "youtube",
    "substack",
]


class NormalizedPost(BaseModel):
    id: str
    platform: Platform
    title: str = ""
    content: str
    url: str
    author: Optional[str] = None
    sourceContext: Optional[str] = None
    createdAt: Optional[str] = None
    keywordsMatched: List[str] = []
    raw: Optional[Any] = None


class GenerateCommentRequest(BaseModel):
    post_url: HttpUrl
    top_k_style: int = 5
    top_k_docs: int = 5


class GenerateCommentResponse(BaseModel):
    comment: str


app = FastAPI(title="KiloCode ML Context & Comment Service")


# -----------------
# Startup
# -----------------
@app.on_event("startup")
def startup():
    global embedder

    embedder = Embedder()

    # Fail early if indexes are missing
    try:
        load_index("comments")
        load_index("docs")
    except Exception as e:
        raise RuntimeError(f"Failed to load indexes: {e}")


# -----------------
# REAL endpoint
# -----------------
@app.post("/ml/generate-comment", response_model=GenerateCommentResponse)
def generate(req: GenerateCommentRequest):
    extracted_text = fetch_post_content(str(req.post_url))

    if not extracted_text.strip():
        return {
            "comment": (
                "This topic raises some interesting points. "
                "Based on similar discussions, KiloCode usually approaches this "
                "by focusing on clear context and practical implementation details."
            )
        }

    post = NormalizedPost(
        id=str(req.post_url),
        platform="reddit",
        content=extracted_text,
        url=str(req.post_url),
    )

    comment = generate_comment(
        post=post,
        embedder=embedder,
        top_k_style=req.top_k_style,
        top_k_docs=req.top_k_docs,
    )

    return {"comment": comment}


# -----------------
# ML-only test endpoint
# -----------------
@app.post("/ml/test-direct", response_model=GenerateCommentResponse)
def test_direct():
    post = NormalizedPost(
        id="test",
        platform="reddit",
        title="New free model Corethink on KiloCode",
        content=(
            "Someone is discussing a new free AI model on KiloCode "
            "and comparing it with GPT and Claude."
        ),
        url="test",
    )

    comment = generate_comment(post, embedder)

    return {"comment": comment}
