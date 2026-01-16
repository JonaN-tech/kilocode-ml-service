def detect_intent(text: str) -> str:
    t = text.lower()

    if "has anyone tried" in t or "anyone tried" in t:
        return "ask_experience"

    if "i have been trying" in t or "i've been trying" in t:
        return "share_experience"

    if "compare" in t or "vs" in t:
        return "comparison"

    return "general"


def build_comment(post, style_examples, doc_facts):
    intent = detect_intent(post.content)

    # --- Base human-written comment (optional)
    base_comment = ""
    if style_examples:
        base_comment = style_examples[0].get("comment_text", "").strip()

    # --- Extract ONE short documentation hint max
    doc_hint = ""
    for d in doc_facts:
        text = d.get("text", "")
        if text and len(text.split()) < 25:
            doc_hint = text.rstrip(".")
            break

    parts = []

    # --- Intent-aware phrasing
    if intent == "share_experience":
        parts.append(
            "Interesting to hear your experience with Corethink on KiloCode."
        )

        if "fast" in post.content.lower():
            parts.append(
                "Speed is something many people notice first, especially in real-world workflows."
            )

    elif intent == "comparison":
        parts.append(
            "The comparison angle is a good one, especially when looking beyond raw benchmarks."
        )

    elif intent == "ask_experience":
        parts.append(
            "A few people have reported similar results in their own tests."
        )

    else:
        parts.append(
            "This is an interesting point to bring up."
        )

    # --- Optional short clarification (NOT a doc dump)
    if doc_hint:
        parts.append(
            "In practice, this often comes down to how the model is integrated into the overall workflow."
        )

    # --- Optional reuse of a short human-written sentence
    if base_comment and len(base_comment.split()) < 30:
        parts.append(base_comment)

    final = " ".join(parts).strip()

    # --- Guard rule: never ask what the post already answered
    if "have you" in final.lower() and "i have" in post.content.lower():
        final = final.replace("Have you", "").replace("have you", "")

    return final
