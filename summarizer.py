def summarize_text(text: str, max_sentences: int = 3) -> str:
    sentences = text.split(". ")
    return ". ".join(sentences[:max_sentences]).strip()
