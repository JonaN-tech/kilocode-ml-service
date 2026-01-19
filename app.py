import re
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Optional, Literal, Any, List, Tuple

from retrieval import Embedder, load_index
from fetchers import fetch_post_content
from comment_engine import generate_comment

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s %(levelname)s %(message)s'
)
logger = logging.getLogger("[ML]")

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


class GenerateCommentResponse(BaseModel):
    comment: str


app = FastAPI(title="KiloCode ML Context & Comment Service")


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
    global embedder

    logger.info("ML service starting up...")
    embedder = Embedder()

    # Fail early if indexes are missing
    try:
        load_index("comments")
        load_index("docs")
        logger.info("Indexes loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load indexes: {e}")
        raise RuntimeError(f"Failed to load indexes: {e}")


# -----------------
# REAL endpoint
# -----------------
@app.post("/ml/generate-comment", response_model=GenerateCommentResponse)
def generate(req: GenerateCommentRequest):
    url = str(req.post_url)
    logger.info(f"generate_comment request url={url[:80]}")
    
    # Detect platform from URL
    platform = detect_platform(url)
    logger.info(f"detected_platform={platform}")
    
    # Fetch content with retries
    text, fetch_status = fetch_post_content(url)
    
    # Build normalized post
    post = NormalizedPost(
        id=url,
        platform=platform,
        content=text,
        url=url,
    )
    
    # Handle fetch failures gracefully
    if fetch_status != "success":
        logger.warning(f"fetch_failed status={fetch_status} url={url[:50]}")
        
        # Try title-only generation as fallback
        if text.strip():
            # We got some text, just not full content
            logger.info(f"fallback=partial_content text_length={len(text)}")
        else:
            # No content at all - generate from URL-derived title
            logger.info(f"fallback=title_only")
            # The generate_comment function will handle title-only case
    
    # Generate comment with platform-aware logic
    try:
        comment = generate_comment(
            post=post,
            embedder=embedder,
            top_k_style=req.top_k_style,
            top_k_docs=req.top_k_docs,
            fetch_status=fetch_status,
        )
        
        logger.info(f"comment_generated length={len(comment)} url={url[:50]}")
        return {"comment": comment}
        
    except Exception as e:
        logger.error(f"comment_generation_failed: {type(e).__name__} url={url[:50]}")
        
        # Final fallback - NEVER return 5xx, always return valid comment
        # Use a generic but contextual fallback
        fallback_comment = generate_safe_fallback(post)
        logger.info(f"safe_fallback_used url={url[:50]}")
        return {"comment": fallback_comment}


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
