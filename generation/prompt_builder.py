import logging
import re
from typing import List

logger = logging.getLogger("[ML]")


def build_lightweight_comment(title: str, content: str, platform: str) -> str:
    """
    Build comment WITHOUT embeddings or retrieval.
    
    This is the primary path for Reddit/GitHub on 512MB Render
    to avoid OOM from model loading.
    
    Uses only title + content text analysis with no ML models.
    
    Args:
        title: Post title
        content: Post content (may be partial)
        platform: Platform name
    
    Returns:
        str: Generated comment
    """
    logger.info(f"build_lightweight title_len={len(title)} content_len={len(content)} platform={platform}")
    
    # Handle empty content
    if not title and not content:
        return "Thanks for starting this discussion!"
    
    if not content and title:
        return build_title_only_comment(title, "general")
    
    # Detect intent from content
    intent = detect_intent(content)
    logger.info(f"lightweight_intent={intent}")
    
    # Extract keywords from content
    words = content.split()
    meaningful_words = [w for w in words if len(w) > 4 and
                        w.lower() not in ['about', 'there', 'their', 'would', 'could', 'should',
                                          'really', 'think', 'thing', 'these', 'those', 'where', 'which']]
    
    # Build intent-aware comment referencing actual content
    parts = []
    
    if intent == "help_request":
        if meaningful_words:
            parts.append(f"Hope you find a solution to the {meaningful_words[0]} issue.")
        else:
            parts.append("Hope you find a solution to this!")
    
    elif intent == "share_experience":
        if meaningful_words:
            parts.append(f"Thanks for sharing your experience with {meaningful_words[0]}.")
        else:
            parts.append("Thanks for sharing your experience!")
    
    elif intent == "comparison":
        if len(meaningful_words) >= 2:
            parts.append(f"The comparison between {meaningful_words[0]} and {meaningful_words[1]} is useful.")
        elif meaningful_words:
            parts.append(f"Good comparison regarding {meaningful_words[0]}.")
        else:
            parts.append("This comparison is helpful!")
    
    elif intent == "ask_experience":
        if meaningful_words:
            parts.append(f"Regarding {meaningful_words[0]}, others have had similar questions.")
        else:
            parts.append("That's a common question in the community.")
    
    else:  # general
        if title and meaningful_words:
            parts.append(f"Interesting post about {title.split()[0] if title.split() else meaningful_words[0]}.")
        elif meaningful_words:
            parts.append(f"Your points about {meaningful_words[0]} are worth considering.")
        elif title:
            parts.append(f"Thanks for posting about {title[:50]}!")
        else:
            parts.append("Thanks for sharing this!")
    
    # Add a closing
    if platform == "github":
        parts.append("Looking forward to updates on this.")
    else:
        parts.append("Thanks for starting this discussion.")
    
    final = " ".join(parts).strip()
    logger.info(f"lightweight_comment_built length={len(final)}")
    
    return final


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


def build_comment(post, title: str, top_chunks: List[str], style_examples, doc_facts, fetch_status="success"):
    """
    Build a comment from title and top relevant chunks.
    
    This uses the full body content indirectly (via chunks), ensuring
    comments reference specific details from the post.
    
    Args:
        post: NormalizedPost object
        title: Post title
        top_chunks: List of most relevant content chunks (1-3 chunks)
        style_examples: Retrieved style examples
        doc_facts: Retrieved documentation facts
        fetch_status: Fetch status
    
    Returns:
        str: Generated comment
    """
    logger.info(f"build_comment title_len={len(title)} num_chunks={len(top_chunks)} fetch_status={fetch_status}")
    
    # Combine title and top chunks for analysis
    combined_text = title
    if top_chunks:
        combined_text += " " + " ".join(top_chunks[:2])  # Use top 2 chunks
    
    # Detect intent from combined text
    intent = detect_intent(combined_text)
    logger.info(f"detected_intent={intent}")
    
    # If we have no content at all, use title-only
    if not top_chunks and title:
        return build_title_only_comment(title, intent)
    
    # Build comment from chunks
    return build_chunk_comment(title, top_chunks, intent, style_examples, doc_facts)


def build_title_only_comment(title: str, intent: str) -> str:
    """Build comment when only title is available."""
    # Extract meaningful words from title
    words = title.split()
    key_topics = [w for w in words if len(w) > 3 and not w.lower() in 
                  ['this', 'that', 'with', 'from', 'have', 'they', 'their', 'what', 'about']]
    
    if intent == "help_request":
        return f"I see you're asking about {', '.join(key_topics[:2]) if key_topics else 'this topic'}. Hope you get some helpful responses!"
    
    elif intent == "comparison":
        return f"Great question about {', '.join(key_topics[:2]) if key_topics else 'this topic'}! Looking forward to seeing the answers."
    
    elif intent == "share_experience":
        return f"Thanks for sharing your experience with {', '.join(key_topics[:2]) if key_topics else 'this topic'}!"
    
    else:
        return f"Interesting post about {', '.join(key_topics[:2]) if key_topics else 'this topic'}! Thanks for starting this discussion."


def build_chunk_comment(title: str, chunks: List[str], intent: str, style_examples, doc_facts) -> str:
    """
    Build comment from title and relevant chunks.
    
    IMPORTANT: Each comment MUST reference specific details from the chunks
    to avoid generic fallbacks.
    """
    parts = []
    
    # Extract meaningful keywords from the chunks
    all_chunk_text = " ".join(chunks[:2])  # Use top 2 chunks
    words = all_chunk_text.split()
    
    # Find meaningful keywords to reference
    meaningful_words = [w for w in words if len(w) > 4 and 
                        w.lower() not in ['about', 'there', 'their', 'would', 'could', 'should', 
                                          'really', 'think', 'thing', 'these', 'those', 'where']]
    
    # Reference something specific from chunks
    referenced = ""
    if meaningful_words:
        # Pick a distinctive word that appears in chunks
        referenced = meaningful_words[0].strip('.,!?')
    
    # --- Intent-aware opening (MUST reference chunk content) ---
    if intent == "share_experience":
        if referenced:
            parts.append(f"I appreciate you sharing your experience with {referenced}.")
        else:
            parts.append(f"Thanks for sharing your experience in this post.")
        
        if "fast" in all_chunk_text.lower() or "speed" in all_chunk_text.lower():
            parts.append("Performance is definitely a key consideration.")
    
    elif intent == "comparison":
        if referenced:
            parts.append(f"The comparison around {referenced} raises important points.")
        else:
            parts.append(f"This comparison between {title.split()[:2]} is very relevant.")
    
    elif intent == "ask_experience":
        if referenced:
            parts.append(f"Regarding {referenced}, others have had similar questions.")
        else:
            parts.append("That's a common question in the community.")
    
    elif intent == "help_request":
        if referenced:
            parts.append(f"The issue with {referenced} is worth investigating.")
        else:
            parts.append("Hope you find a solution to this!")
    
    elif intent == "appreciation":
        parts.append("Glad this has been helpful!")
    
    else:  # general
        if referenced:
            parts.append(f"This raises interesting points about {referenced}.")
        elif title:
            # Reference title
            title_words = [w for w in title.split() if len(w) > 4]
            if title_words:
                parts.append(f"The discussion about {title_words[0]} is very timely.")
            else:
                parts.append("This is a thoughtful contribution to the discussion.")
        else:
            parts.append("This is a thoughtful contribution.")
    
    # --- Add contextual insight from chunks if available ---
    # Find a specific detail or quote from the chunks
    if chunks and len(chunks[0]) > 50:
        # Take a relevant snippet from the first chunk
        first_chunk = chunks[0]
        sentences = re.split(r'[.!?]', first_chunk)
        if len(sentences) > 1 and len(sentences[1].strip()) > 20:
            # Reference a specific sentence
            snippet = sentences[1].strip()[:100]
            if snippet:
                parts.append(f"The point about {snippet.lower().split()[:5]} is particularly relevant.")
    
    # --- Optional: Add style example if relevant ---
    if style_examples:
        base_comment = style_examples[0].get("comment_text", "").strip()
        if base_comment and len(base_comment.split()) < 25:
            # Only add if it's short and relevant
            if any(keyword in base_comment.lower() for keyword in meaningful_words[:3]):
                parts.append(base_comment)
    
    # Join and finalize
    final = " ".join(parts).strip()
    
    # Guard rule: avoid redundant questions
    if "have you" in final.lower() and "i have" in all_chunk_text.lower():
        final = re.sub(r'\bhave you\b', '', final, flags=re.IGNORECASE)
        final = re.sub(r'\s+', ' ', final).strip()
    
    logger.info(f"comment_built length={len(final)} parts={len(parts)}")
    
    return final
