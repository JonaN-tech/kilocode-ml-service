import requests
from bs4 import BeautifulSoup
import time
import logging
import re

logger = logging.getLogger("[ML]")


def fetch_post_content(url: str, max_retries: int = 2, timeout: int = 8) -> dict:
    """
    Fetch post content with retries and structured logging.
    
    Returns:
        dict: {
            "text": str,           # Full body content
            "title": str,          # Extracted title
            "fetch_status": str,   # "success", "http_error", "timeout", "blocked", "error"
            "content_len": int     # Total character count
        }
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"fetch_attempt url={url[:50]}... attempt={attempt+1}")
            
            res = requests.get(url, headers=headers, timeout=timeout)
            
            if res.status_code == 403:
                logger.warning(f"fetch_blocked status=403 url={url[:50]}")
                if attempt < max_retries:
                    time.sleep(1 * (attempt + 1))
                    continue
                return {"text": "", "title": "", "fetch_status": "blocked", "content_len": 0}
            
            if res.status_code >= 400:
                logger.warning(f"fetch_failed status={res.status_code} url={url[:50]}")
                return {"text": "", "title": "", "fetch_status": "http_error", "content_len": 0}
            
            soup = BeautifulSoup(res.text, "html.parser")
            
            # Extract title from common tags
            title = extract_title(soup, url)
            
            # Remove unwanted tags
            for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "iframe", "form"]):
                tag.decompose()
            
            # Extract body text
            text = soup.get_text(separator=" ")
            text = " ".join(text.split())  # Collapse whitespace
            
            # Check if we got meaningful content
            if len(text.strip()) < 20:
                logger.warning(f"fetch_empty content_len={len(text)}")
                return {"text": "", "title": title, "fetch_status": "empty", "content_len": len(text)}
            
            content_len = len(text)
            logger.info(f"fetch_success content_len={content_len} title_len={len(title)}")
            
            return {
                "text": text[:50000],  # Hard cap to prevent extreme cases
                "title": title,
                "fetch_status": "success",
                "content_len": content_len
            }
            
        except requests.exceptions.Timeout:
            logger.warning(f"fetch_timeout timeout={timeout}s url={url[:50]}")
            if attempt < max_retries:
                time.sleep(1 * (attempt + 1))
                continue
            return {"text": "", "title": "", "fetch_status": "timeout", "content_len": 0}
            
        except requests.exceptions.HTTPError as e:
            logger.warning(f"fetch_http_error error={e} url={url[:50]}")
            last_error = e
            if attempt < max_retries:
                time.sleep(1 * (attempt + 1))
                continue
            return {"text": "", "title": "", "fetch_status": "http_error", "content_len": 0}
            
        except Exception as e:
            logger.error(f"fetch_error error={type(e).__name__} url={url[:50]}")
            last_error = e
            if attempt < max_retries:
                time.sleep(1 * (attempt + 1))
                continue
            return {"text": "", "title": "", "fetch_status": "error", "content_len": 0}
    
    logger.error(f"fetch_failed_after_retries error={last_error}")
    return {"text": "", "title": "", "fetch_status": "error", "content_len": 0}


def extract_title(soup: BeautifulSoup, url: str) -> str:
    """
    Extract title from HTML soup with platform-specific logic.
    
    Args:
        soup: BeautifulSoup object
        url: Original URL for fallback extraction
    
    Returns:
        Extracted title string
    """
    title = ""
    
    # Try common title tags in order of preference
    if soup.title:
        title = soup.title.string
    elif soup.find("h1"):
        title = soup.find("h1").get_text()
    elif soup.find("meta", property="og:title"):
        title = soup.find("meta", property="og:title")["content"]
    elif soup.find("meta", {"name": "title"}):
        title = soup.find("meta", {"name": "title"})["content"]
    
    # Fallback: extract from URL
    if not title or len(title.strip()) < 3:
        title = extract_title_from_url(url)
    
    if title:
        # Clean up title
        title = title.strip()
        # Remove site name suffixes (e.g., " - Reddit", " | GitHub")
        title = re.split(r' [-|•:] ', title)[0]
        title = title[:200]  # Cap length
    
    return title


def extract_title_from_url(url: str) -> str:
    """
    Extract a fallback title from URL path.
    
    Examples:
        /r/programming/comments/xyz/my-post-title -> My Post Title
        /user/repo/issues/123 -> Issue 123
    """
    try:
        # Reddit pattern
        match = re.search(r'/comments/[\w]+/(.+?)(?:\?|/|$)', url)
        if match:
            title = match.group(1).replace('-', ' ').replace('_', ' ')
            return title.title()
        
        # GitHub issues pattern
        match = re.search(r'/issues/(\d+)', url)
        if match:
            return f"Issue #{match.group(1)}"
        
        # GitHub PR pattern
        match = re.search(r'/pull/(\d+)', url)
        if match:
            return f"Pull Request #{match.group(1)}"
        
        # Generic: take last path segment
        parts = url.rstrip('/').split('/')
        if parts:
            last_part = parts[-1].replace('-', ' ').replace('_', ' ')
            if last_part and len(last_part) > 3:
                return last_part.title()
    
    except Exception:
        pass
    
    return ""
