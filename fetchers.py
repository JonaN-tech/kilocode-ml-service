import requests
from bs4 import BeautifulSoup

def fetch_post_content(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; KiloCodeBot/1.0)"
    }

    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print("FETCH BLOCKED:", e)
        return ""
    except Exception as e:
        print("FETCH ERROR:", e)
        return ""

    soup = BeautifulSoup(res.text, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    text = " ".join(text.split())

    return text[:8000]
