def detect_intent(text: str) -> str:
    t = text.lower()

    if "has anyone tried" in t or "anyone tried" in t:
        return "ask_experience"

    if "i have been trying" in t or "i've been trying" in t:
        return "share_experience"

    if "compare" in t or "vs" in t:
        return "comparison"

    return "general"
