import requests
from bs4 import BeautifulSoup
import time
import logging

logger = logging.getLogger("[ML]")


def fetch_post_content(url: str, max_retries: int = 2, timeout: int = 8) -> tuple[str, str]:
    """
    Fetch post content with retries and structured logging.
    
    Returns:
        tuple: (extracted_text, fetch_status)
            - extracted_text: The cleaned text content or empty string
            - fetch_status: One of: "success", "http_error", "timeout", "blocked", "error"
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"fetch_attempt url_hash={url[:50]}... attempt={attempt+1}")
            
            res = requests.get(url, headers=headers, timeout=timeout)
            
            if res.status_code == 403:
                logger.warning(f"reddit fetch blocked: 403 url={url[:50]}")
                if attempt < max_retries:
                    time.sleep(1 * (attempt + 1))  # Backoff
                    continue
                return "", "blocked"
            
            if res.status_code >= 400:
                logger.warning(f"reddit fetch failed: {res.status_code} url={url[:50]}")
                return "", "http_error"
            
            soup = BeautifulSoup(res.text, "html.parser")
            
            for tag in soup(["script", "style", "noscript", "nav", "header", "footer"]):
                tag.decompose()
            
            text = soup.get_text(separator=" ")
            text = " ".join(text.split())
            
            # Check if we got meaningful content
            if len(text.strip()) < 20:
                logger.warning(f"reddit fetch returned empty/short content: {len(text)} chars")
                return "", "empty"
            
            logger.info(f"reddit fetch success: {len(text)} chars")
            return text[:8000], "success"
            
        except requests.exceptions.Timeout:
            logger.warning(f"reddit fetch timeout: {timeout}s url={url[:50]}")
            if attempt < max_retries:
                time.sleep(1 * (attempt + 1))
                continue
            return "", "timeout"
            
        except requests.exceptions.HTTPError as e:
            logger.warning(f"reddit fetch http error: {e} url={url[:50]}")
            last_error = e
            if attempt < max_retries:
                time.sleep(1 * (attempt + 1))
                continue
            return "", "http_error"
            
        except Exception as e:
            logger.error(f"reddit fetch error: {type(e).__name__} url={url[:50]}")
            last_error = e
            if attempt < max_retries:
                time.sleep(1 * (attempt + 1))
                continue
            return "", "error"
    
    logger.error(f"reddit fetch failed after retries: {last_error}")
    return "", "error"
