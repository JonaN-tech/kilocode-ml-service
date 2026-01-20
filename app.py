import re
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Optional, Literal, Any, List, Tuple

from ml.embeddings import get_embed_model
from retrieval import Embedder, load_index
from fetchers import fetch_post_content
from comment_engine import generate_comment

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s %(levelname)s %(message)s'
)
logger = logging.getLogger("[ML]")
mem_logger = logging.getLogger("[ML][MEM]")

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
    top_k_style: int = 3
    top_k_docs: int = 3
    source: Optional[str] = "api"


class GenerateCommentResponse(BaseModel):
    comment: str


app = FastAPI(title="KiloCode ML Context & Comment Service")

# Global singleton embedder (lazy-loaded on first request)
embedder = None


def detect_platform(url: str) -> Platform:
    """Detect platform from URL."""
    url_lower = url.lower()
    if "twitter.com" in url_lower or "x.com" in url_lower:
        return "twitter"
    elif "reddit.com" in url_lower:
        return "reddit"
    elif "github.com" in url_lower:
        return "github"
    elif "news.ycombinator.com" in url_lower:
        return "hackernews"
    elif "youtube.com" in url_lower or "youtu.be" in url_lower:
        return "youtube"
    elif "substack.com" in url_lower:
        return "substack"
    else:
        # Default to reddit for unknown platforms
        return "reddit"


def extract_title_from_url(url: str) -> str:
    """Extract a fallback title from URL path for display purposes."""
    try:
        # For reddit: /r/subreddit/comments/xyz/title
        match = re.search(r'comments/[\w]+/(.+?)(?:\?|/|$)', url)
        if match:
            title = match.group(1)
            title = title.replace('-', ' ')
            return title.title()
    except Exception:
        pass
    return ""


# -----------------
# Startup
# -----------------
@app.on_event("startup")
def startup():
    """
    Load embedding model and indexes at startup.
    
    CRITICAL: This prevents per-request model loading that causes RAM spikes.
    """
    global embedder
    
    mem_logger.info("startup begin")
    
    # Preload embedding model (singleton pattern)
    try:
        get_embed_model()
        mem_logger.info("embedding_model_preloaded")
    except Exception as e:
        mem_logger.error(f"model_preload_failed error={type(e).__name__}")
        # Continue - embedder will try lazy load if needed
    
    # Create embedder instance
    embedder = Embedder()
    
    # Pre-load indexes into cache (they're small)
    try:
        load_index("comments")
        load_index("docs")
        mem_logger.info("indexes_preloaded")
    except Exception as e:
        logger.error(f"index_preload_failed error={type(e).__name__}")
        # Don't fail startup - we can still work without indexes
    
    mem_logger.info("startup complete")


# -----------------
# REAL endpoint
# -----------------
@app.post("/ml/generate-comment", response_model=GenerateCommentResponse)
def generate(req: GenerateCommentRequest):
    """
    Generate comment endpoint with ZERO-5XX GUARANTEE.
    
    This endpoint MUST NEVER throw exceptions or return 5xx.
    All errors are caught and handled gracefully.
    """
    url = str(req.post_url)
    
    # Detect /docs calls and return safe placeholder
    if req.source == "docs":
        logger.info("docs_call_detected returning_placeholder")
        return {
            "comment": "This is a placeholder response. The /docs UI is for testing only. "
            "Production calls use 'source=api' and run the full ML pipeline."
        }
    
    # TOP-LEVEL TRY/CATCH - NOTHING escapes this
    try:
        mem_logger.info("request_start")
        logger.info(f"generate_comment url={url[:80]}")
        
        # Detect platform
        platform = detect_platform(url)
        logger.info(f"platform={platform}")
        
        # Fetch content with retries - returns dict with title, text, status, content_len
        fetch_result = fetch_post_content(url)
        text = fetch_result["text"]
        title = fetch_result["title"]
        fetch_status = fetch_result["fetch_status"]
        content_len = fetch_result["content_len"]
        
        logger.info(f"fetch_status={fetch_status} content_len={content_len} title_len={len(title)}")
        
        # Build normalized post
        post = NormalizedPost(
            id=url,
            platform=platform,
            title=title,
            content=text,
            url=url,
        )
        
        # Handle fetch failures
        if fetch_status != "success":
            logger.warning(f"fetch_failed status={fetch_status}")
            if text.strip():
                logger.info(f"fallback=partial_content text_length={len(text)}")
            elif title.strip():
                logger.info(f"fallback=title_only")
            else:
                logger.warning(f"fallback=emergency_no_content")
        
        # Generate comment with multiple safety layers
        try:
            mem_logger.info("before_generate")
            
            comment = generate_comment(
                post=post,
                embedder=embedder,
                top_k_style=req.top_k_style,
                top_k_docs=req.top_k_docs,
                fetch_status=fetch_status,
            )
            
            mem_logger.info("after_generate")
            logger.info(f"comment_generated length={len(comment)}")
            
            return {"comment": comment}
            
        except MemoryError as e:
            mem_logger.error(f"OOM_detected error={type(e).__name__}")
            # Return lightweight fallback without crashing
            fallback = generate_safe_fallback(post)
            logger.info("OOM_fallback_used")
            return {"comment": fallback}
            
        except Exception as e:
            logger.error(f"comment_generation_failed error={type(e).__name__}")
            # Inner fallback
            fallback = generate_safe_fallback(post)
            logger.info("safe_fallback_used")
            return {"comment": fallback}
    
    except HTTPException:
        # Re-raise HTTP exceptions (proper 4xx responses)
        raise
    except Exception as e:
        # OUTER CATCH-ALL - Last line of defense
        # This catches ANY error including in fetch, post creation, etc.
        logger.error(f"CRITICAL_ERROR error={type(e).__name__} url={url[:50]}")
        mem_logger.error(f"critical_failure")
        
        # Return JSON error response (never HTML)
        raise HTTPException(
            status_code=500,
            detail="Internal ML pipeline error - returning safe fallback"
        )


def generate_safe_fallback(post: NormalizedPost) -> str:
    """Generate a safe fallback comment when everything fails."""
    if post.platform == "twitter":
        # Short conversational reply for Twitter
        return "Thanks for sharing this!"
    else:
        # Generic but useful for other platforms
        return "This is an interesting discussion. Thanks for starting this thread!"


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


# Health check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}
