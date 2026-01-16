import requests
from bs4 import BeautifulSoup

def fetch_post_content(url: str) -> str:
    """
    Fetch and extract readable text from a public post URL.
    Works as a baseline for Twitter, Reddit, HN, Substack, GitHub.
    """
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    res = requests.get(url, headers=headers, timeout=10)
    res.raise_for_status()

    soup = BeautifulSoup(res.text, "html.parser")

    # remove scripts/styles
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    text = " ".join(text.split())

    return text[:8000]  # hard safety limit
