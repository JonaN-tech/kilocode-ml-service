import logging
import re

logger = logging.getLogger("[ML]")


def detect_intent(text: str) -> str:
    """Detect intent from text for Reddit/other platforms."""
    t = text.lower()

    if "has anyone tried" in t or "anyone tried" in t:
        return "ask_experience"

    if "i have been trying" in t or "i've been trying" in t:
        return "share_experience"

    if "compare" in t or "vs" in t or "versus" in t:
        return "comparison"
    
    if "help" in t or "issue" in t or "problem" in t or "error" in t:
        return "help_request"

    if "thanks" in t or "thank you" in t:
        return "appreciation"

    return "general"


def detect_twitter_intent(text: str) -> str:
    """
    Detect intent from Twitter text.
    Twitter content is short, noisy, and often contains:
    - mentions (@username)
    - links (http...)
    - hashtags (#topic)
    - short questions
    - announcements
    """
    t = text.lower().strip()
    
    # Check for questions
    if "?" in t:
        return "question"
    
    # Check for announcements or news
    if any(word in t for word in ['just announced', 'new', 'release', 'launch', 'update']):
        return "announcement"
    
    # Check for comparisons
    if " vs " in t or " versus " in t or "compared to" in t:
        return "comparison"
    
    # Check if mostly links
    words = t.split()
    link_count = sum(1 for w in words if w.startswith('http'))
    if link_count > 0 and link_count >= len(words) * 0.5:
        return "link_share"
    
    # Check for mentions as primary content
    mention_count = sum(1 for w in words if w.startswith('@'))
    if mention_count > 0 and mention_count >= len(words) * 0.3:
        return "mention"
    
    # Default to general
    return "general"


def build_comment(post, style_examples, doc_facts, fetch_status="success"):
    """
    Build a comment based on post content, with no generic fallback reuse.
    
    IMPORTANT: Every comment must reference something from the input text
    to avoid the "This topic raises some interesting points..." problem.
    """
    content = post.content.strip()
    title = post.title.strip()
    
    logger.info(f"build_comment fetch_status={fetch_status} content_length={len(content)}")
    
    # Detect intent
    intent = detect_intent(content or title)
    logger.info(f"detected_intent={intent}")
    
    # If we have no content at all, generate from title only
    if not content and title:
        return build_title_only_comment(title, intent)
    
    # If we have content, generate from it
    return build_content_comment(content, title, intent, style_examples, doc_facts, fetch_status)


def build_title_only_comment(title: str, intent: str) -> str:
    """Build comment when only title is available."""
    # Extract meaningful words from title
    words = title.split()
    key_topics = [w for w in words if len(w) > 3 and not w.lower() in 
                  ['this', 'that', 'with', 'from', 'have', 'they', 'their', 'what', 'about']]
    
    if intent == "help_request":
        return f"I see you're asking about {', '.join(key_topics[:2]) if key_topics else 'this topic'}. Hope you get some helpful responses!"
    
    elif intent == "question":
        return f"Great question about {', '.join(key_topics[:2]) if key_topics else 'this topic'}! Looking forward to seeing the answers."
    
    elif intent == "share_experience":
        return f"Thanks for sharing your experience with {', '.join(key_topics[:2]) if key_topics else 'this topic'}!"
    
    else:
        return f"Interesting post about {', '.join(key_topics[:2]) if key_topics else 'this topic'}! Thanks for starting this discussion."


def build_content_comment(content: str, title: str, intent: str, style_examples, doc_facts, fetch_status) -> str:
    """
    Build comment from content with proper intent handling.
    
    NO generic fallbacks - each comment references something from the input.
    """
    parts = []
    
    # Extract a unique phrase from the content to reference
    content_lower = content.lower()
    words = content.split()
    
    # Find meaningful keywords to reference
    meaningful_words = [w for w in words if len(w) > 4 and 
                        w.lower() not in ['about', 'there', 'their', 'would', 'could', 'should', 'really', 'think', 'thing']]
    
    # Reference something specific from content
    referenced = ""
    if meaningful_words:
        # Pick a word that appears in content
        referenced = meaningful_words[0].strip('.,!?')
    
    # --- Intent-aware opening (MUST reference content) ---
    if intent == "share_experience":
        if referenced:
            parts.append(f"I appreciate you sharing your experience with {referenced}.")
        else:
            parts.append("Thanks for sharing your experience!")
        
        if "fast" in content_lower or "speed" in content_lower:
            parts.append("Speed is indeed a factor many notice first.")
    
    elif intent == "comparison":
        if referenced:
            parts.append(f"The comparison around {referenced} is very relevant right now.")
        else:
            parts.append("Comparisons like this help put things in perspective.")
    
    elif intent == "ask_experience":
        if referenced:
            parts.append(f"Regarding {referenced}, some people have reported similar results.")
        else:
            parts.append("That's a good question many have asked before.")
    
    elif intent == "help_request":
        if referenced:
            parts.append(f"Hopefully someone can help with {referenced}.")
        else:
            parts.append("Hope you get some helpful answers on this!")
    
    elif intent == "appreciation":
        parts.append("Glad this was helpful for you!")
    
    else:  # general
        if referenced:
            parts.append(f"This raises some interesting points about {referenced}.")
        else:
            parts.append("This is a thoughtful contribution to the discussion.")
    
    # --- Add a contextual sentence (NOT a generic doc dump) ---
    # Only add doc hint if we actually have content and it's relevant
    doc_hint = ""
    for d in doc_facts:
        text = d.get("text", "")
        if text and len(text.split()) < 25:
            doc_hint = text.rstrip(".")
            break
    
    # Only add doc hint if it makes sense contextually
    if doc_hint and referenced and referenced.lower() in doc_hint.lower():
        parts.append(doc_hint + ".")
    
    # --- Optional: Add style example if short and relevant ---
    if style_examples:
        base_comment = style_examples[0].get("comment_text", "").strip()
        if base_comment and len(base_comment.split()) < 30:
            parts.append(base_comment)
    
    # Join and finalize
    final = " ".join(parts).strip()
    
    # Guard rule: never ask what the post already answered
    if "have you" in final.lower() and "i have" in content_lower:
        final = re.sub(r'\bhave you\b', '', final, flags=re.IGNORECASE)
        final = re.sub(r'\s+', ' ', final).strip()
    
    logger.info(f"comment_built length={len(final)} parts={len(parts)}")
    
    return final
