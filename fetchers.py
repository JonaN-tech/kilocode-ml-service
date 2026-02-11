import requests
from bs4 import BeautifulSoup
import time
import logging
import re
import json

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
    
    ENHANCED: Better handling for Reddit's dynamic content.
    
    Args:
        soup: BeautifulSoup object
        url: Original URL for fallback extraction
    
    Returns:
        Extracted title string
    """
    title = ""
    is_reddit = "reddit.com" in url.lower()
    
    # Platform-specific extraction
    if is_reddit:
        title = extract_reddit_title(soup, url)
        if title and len(title) > 10:
            logger.info(f"reddit_title_extracted length={len(title)} method=platform_specific")
            return title
    
    # Try og:title first (most reliable for modern sites)
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        title = og_title["content"]
        if title and len(title.strip()) > 10:
            logger.info(f"title_from_og_title length={len(title)}")
            return clean_title(title, url)
    
    # Try Twitter card title
    twitter_title = soup.find("meta", {"name": "twitter:title"})
    if twitter_title and twitter_title.get("content"):
        title = twitter_title["content"]
        if title and len(title.strip()) > 10:
            logger.info(f"title_from_twitter_card length={len(title)}")
            return clean_title(title, url)
    
    # Try <title> tag
    if soup.title and soup.title.string:
        title = soup.title.string
        if title and len(title.strip()) > 10:
            logger.info(f"title_from_title_tag length={len(title)}")
            return clean_title(title, url)
    
    # Try <h1> tag
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(strip=True)
        if title and len(title.strip()) > 10:
            logger.info(f"title_from_h1 length={len(title)}")
            return clean_title(title, url)
    
    # Try meta name="title"
    meta_title = soup.find("meta", {"name": "title"})
    if meta_title and meta_title.get("content"):
        title = meta_title["content"]
        if title and len(title.strip()) > 10:
            logger.info(f"title_from_meta_title length={len(title)}")
            return clean_title(title, url)
    
    # Fallback: extract from URL
    title = extract_title_from_url(url)
    if title:
        logger.info(f"title_from_url_fallback length={len(title)}")
    else:
        logger.warning("title_extraction_failed all_methods_exhausted")
    
    return title


def extract_reddit_title(soup: BeautifulSoup, url: str) -> str:
    """
    Extract title specifically from Reddit pages.
    
    Reddit uses dynamic rendering, so we need multiple strategies:
    1. JSON-LD structured data
    2. og:title meta tag
    3. Specific Reddit HTML elements
    4. URL fallback
    """
    # Strategy 1: Try JSON-LD data (most reliable when present)
    json_ld = soup.find("script", {"type": "application/ld+json"})
    if json_ld and json_ld.string:
        try:
            data = json.loads(json_ld.string)
            if isinstance(data, dict):
                title = data.get("headline") or data.get("name")
                if title and len(title) > 10:
                    return title
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        title = item.get("headline") or item.get("name")
                        if title and len(title) > 10:
                            return title
        except (json.JSONDecodeError, TypeError):
            pass
    
    # Strategy 2: og:title (usually reliable for Reddit)
    og_title = soup.find("meta", property="og:title")
    if og_title:
        content = og_title.get("content", "")
        if content and len(content) > 10:
            # Reddit og:title sometimes has "r/subreddit - " prefix
            content = re.sub(r'^r/\w+\s*[-:]\s*', '', content)
            return content
    
    # Strategy 3: Reddit-specific elements (old and new Reddit)
    # New Reddit: data-testid attributes
    title_elem = soup.find(attrs={"data-testid": "post-title"})
    if not title_elem:
        title_elem = soup.find("h1", {"slot": "title"})
    if not title_elem:
        # Old Reddit: .title class
        title_elem = soup.find("a", {"class": "title"})
    if not title_elem:
        title_elem = soup.find("p", {"class": "title"})
    if not title_elem:
        # Try any h1 with substantial content
        for h1 in soup.find_all("h1"):
            text = h1.get_text(strip=True)
            if len(text) > 15 and not text.startswith("r/"):
                title_elem = h1
                break
    
    if title_elem:
        title = title_elem.get_text(strip=True)
        if len(title) > 10:
            return title
    
    # Strategy 4: Title tag with cleanup
    if soup.title and soup.title.string:
        title = soup.title.string
        # Reddit titles often have " : subreddit" or " - Reddit" suffix
        title = re.sub(r'\s*[:\-|]\s*(r/\w+|Reddit).*$', '', title, flags=re.IGNORECASE)
        if len(title) > 10:
            return title
    
    # Strategy 5: URL fallback
    url_title = extract_title_from_url(url)
    if url_title and len(url_title) > 5:
        return url_title
    
    return ""


def clean_title(title: str, url: str) -> str:
    """Clean up extracted title."""
    if not title:
        return ""
    
    title = title.strip()
    
    # Remove site name suffixes (e.g., " - Reddit", " | GitHub")
    title = re.split(r'\s+[-|•:]\s+(?:Reddit|GitHub|Hacker News|HN|YouTube|Substack)', title, flags=re.IGNORECASE)[0]
    
    # Remove "r/subreddit - " prefix if present
    title = re.sub(r'^r/\w+\s*[-:]\s*', '', title)
    
    # Cap length
    title = title[:250]
    
    return title.strip()


def extract_title_from_url(url: str) -> str:
    """
    Extract a fallback title from URL path.
    
    ENHANCED: Better handling for Reddit URL structure.
    
    Examples:
        /r/programming/comments/xyz/my_post_title_here -> My Post Title Here
        /user/repo/issues/123 -> Issue 123
    """
    try:
        # Reddit pattern - enhanced to handle various formats
        # Format: /r/subreddit/comments/id/slug_title_here
        match = re.search(r'/comments/[\w]+/([^/?]+)', url)
        if match:
            title = match.group(1)
            # Handle URL encoding
            title = requests.utils.unquote(title)
            # Replace separators with spaces
            title = title.replace('_', ' ').replace('-', ' ')
            # Remove multiple spaces
            title = ' '.join(title.split())
            # Title case but keep acronyms
            words = title.split()
            titled_words = []
            for word in words:
                if word.isupper() and len(word) <= 5:
                    titled_words.append(word)  # Keep acronyms like API, CSS
                else:
                    titled_words.append(word.capitalize())
            title = ' '.join(titled_words)
            
            if len(title) > 5:
                logger.info(f"title_from_reddit_url length={len(title)}")
                return title
        
        # GitHub issues pattern
        match = re.search(r'/issues/(\d+)', url)
        if match:
            return f"Issue #{match.group(1)}"
        
        # GitHub PR pattern
        match = re.search(r'/pull/(\d+)', url)
        if match:
            return f"Pull Request #{match.group(1)}"
        
        # Hacker News pattern
        match = re.search(r'news\.ycombinator\.com/item\?id=(\d+)', url)
        if match:
            return f"HN Discussion #{match.group(1)}"
        
        # Generic: take last meaningful path segment
        parts = url.rstrip('/').split('/')
        for part in reversed(parts):
            # Skip numeric-only parts and very short parts
            if part and len(part) > 5 and not part.isdigit():
                # Clean up
                clean_part = part.replace('-', ' ').replace('_', ' ')
                clean_part = requests.utils.unquote(clean_part)
                if len(clean_part) > 5:
                    return clean_part.title()
    
    except Exception as e:
        logger.warning(f"url_title_extraction_error error={type(e).__name__}")
    
    return ""
