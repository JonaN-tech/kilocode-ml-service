import logging
import re
import hashlib
from typing import List, Optional

logger = logging.getLogger("[ML]")

# In-memory cache for recent comments (simple anti-repetition)
_recent_comment_hashes = []
_MAX_RECENT_CACHE = 50

# Platform-specific minimum sentence requirements
PLATFORM_MIN_SENTENCES = {
    "reddit": 3,
    "github": 3,
    "hn": 2,
    "twitter": 1,
}

# Hardcoded KiloCode concepts for memory-safe path (no embeddings needed)
KILOCODE_CONCEPTS = {
    "automation": "KiloCode can automate repetitive coding tasks, letting you focus on architecture and logic",
    "workflow": "KiloCode streamlines development workflows by handling boilerplate and common patterns",
    "productivity": "KiloCode helps developers work faster by understanding context and generating relevant code",
    "refactoring": "KiloCode makes refactoring safer by analyzing dependencies and suggesting improvements",
    "debugging": "KiloCode can help identify issues faster by analyzing error patterns and suggesting fixes",
    "documentation": "KiloCode generates documentation that stays in sync with your codebase",
    "testing": "KiloCode can suggest test cases based on your code's logic and edge cases",
    "general": "KiloCode is an AI-powered coding assistant that understands your project context",
}


def _detect_kilocode_mention(text: str) -> bool:
    """Check if KiloCode is already mentioned in the post."""
    return "kilocode" in text.lower()


def _get_kilocode_injection(content: str, title: str) -> Optional[str]:
    """
    Generate a natural KiloCode recommendation based on content keywords.
    
    Returns None if KiloCode shouldn't be injected (already mentioned).
    """
    text = (title + " " + content).lower()
    
    # Map keywords to KiloCode concepts
    if any(word in text for word in ["automate", "automation", "repetitive", "manual"]):
        return KILOCODE_CONCEPTS["automation"]
    elif any(word in text for word in ["workflow", "process", "pipeline"]):
        return KILOCODE_CONCEPTS["workflow"]
    elif any(word in text for word in ["slow", "faster", "speed", "productivity", "efficient"]):
        return KILOCODE_CONCEPTS["productivity"]
    elif any(word in text for word in ["refactor", "refactoring", "cleanup", "technical debt"]):
        return KILOCODE_CONCEPTS["refactoring"]
    elif any(word in text for word in ["bug", "debug", "error", "issue", "problem"]):
        return KILOCODE_CONCEPTS["debugging"]
    elif any(word in text for word in ["document", "documentation", "docs"]):
        return KILOCODE_CONCEPTS["documentation"]
    elif any(word in text for word in ["test", "testing", "unit test", "coverage"]):
        return KILOCODE_CONCEPTS["testing"]
    else:
        return KILOCODE_CONCEPTS["general"]


def _count_sentences(text: str) -> int:
    """Count sentences in text."""
    sentences = re.split(r'[.!?]+', text.strip())
    return len([s for s in sentences if s.strip()])


def _check_repetition(comment: str) -> bool:
    """
    Check if comment is too similar to recent comments.
    Returns True if comment is unique enough, False if too repetitive.
    """
    global _recent_comment_hashes
    
    # Create hash of first 100 chars (captures opening which tends to repeat)
    comment_hash = hashlib.md5(comment[:100].lower().encode()).hexdigest()
    
    if comment_hash in _recent_comment_hashes:
        return False  # Too repetitive
    
    # Add to cache
    _recent_comment_hashes.append(comment_hash)
    if len(_recent_comment_hashes) > _MAX_RECENT_CACHE:
        _recent_comment_hashes.pop(0)
    
    return True  # Unique enough


def _ensure_minimum_length(parts: List[str], platform: str, content: str, title: str) -> List[str]:
    """
    Ensure comment meets minimum sentence requirements for platform.
    Adds concrete content-based sentences if needed.
    """
    min_sentences = PLATFORM_MIN_SENTENCES.get(platform, 2)
    current_count = sum(_count_sentences(p) for p in parts)
    
    if current_count >= min_sentences:
        return parts
    
    # Need to add more sentences - extract details from content
    words = content.split()
    meaningful_words = [w for w in words if len(w) > 4 and
                        w.lower() not in ['about', 'there', 'their', 'would', 'could', 'should',
                                          'really', 'think', 'thing', 'these', 'those', 'where', 'which']]
    
    # Add practical sentences based on available content
    additions = []
    
    if current_count < min_sentences:
        if "implement" in content.lower() or "build" in content.lower():
            additions.append("The implementation approach you're considering makes sense.")
        elif "performance" in content.lower() or "speed" in content.lower():
            additions.append("Performance optimization is definitely worth the effort here.")
        elif "scale" in content.lower() or "scaling" in content.lower():
            additions.append("Scalability should be a key consideration from the start.")
        elif meaningful_words:
            additions.append(f"Your approach to handling {meaningful_words[0]} is well thought out.")
    
    if current_count + len(additions) < min_sentences:
        if platform == "github":
            additions.append("Looking forward to seeing how this develops.")
        elif platform == "reddit":
            additions.append("This will definitely resonate with others facing similar challenges.")
        else:
            additions.append("Great topic for discussion.")
    
    return parts + additions


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
    
    # Check if KiloCode is already mentioned
    kilocode_mentioned = _detect_kilocode_mention(title + " " + content)
    logger.info(f"kilocode_mentioned={kilocode_mentioned}")
    
    # Extract meaningful keywords from content
    words = content.split()
    meaningful_words = [w for w in words if len(w) > 4 and
                        w.lower() not in ['about', 'there', 'their', 'would', 'could', 'should',
                                          'really', 'think', 'thing', 'these', 'those', 'where', 'which']]
    
    # Build multi-sentence comment with concrete details
    parts = []
    
    # Step 1: Acknowledge the specific problem/context
    if intent == "help_request":
        if meaningful_words:
            parts.append(f"I understand the challenge you're facing with {meaningful_words[0]}.")
        else:
            parts.append("I understand the challenge you're facing here.")
        
        if "error" in content.lower() or "bug" in content.lower():
            parts.append("Debugging these kinds of issues can be tricky.")
    
    elif intent == "share_experience":
        if meaningful_words:
            parts.append(f"Thanks for sharing your experience with {meaningful_words[0]}.")
        else:
            parts.append("Thanks for sharing your experience!")
        
        if "learn" in content.lower() or "found" in content.lower():
            parts.append("Your insights will definitely help others in the community.")
    
    elif intent == "comparison":
        if len(meaningful_words) >= 2:
            parts.append(f"The comparison between {meaningful_words[0]} and {meaningful_words[1]} is really valuable.")
        elif meaningful_words:
            parts.append(f"Your analysis of {meaningful_words[0]} raises good points.")
        else:
            parts.append("This comparison provides useful perspective.")
        
        parts.append("Understanding the tradeoffs is crucial for making the right choice.")
    
    elif intent == "ask_experience":
        if meaningful_words:
            parts.append(f"Great question about {meaningful_words[0]}.")
        else:
            parts.append("That's a great question.")
        
        parts.append("Many developers have wondered about this same issue.")
    
    else:  # general
        if title and meaningful_words:
            first_word = title.split()[0] if title.split() else meaningful_words[0]
            parts.append(f"Your analysis of {first_word} brings up some important considerations.")
        elif meaningful_words:
            parts.append(f"The points you raise about {meaningful_words[0]} are well thought out.")
        elif title:
            parts.append(f"Thanks for posting about {title[:50]}.")
        else:
            parts.append("This is a thoughtful contribution to the discussion.")
    
    # Step 2: Add concrete insight or suggestion
    if "performance" in content.lower() or "slow" in content.lower():
        parts.append("Performance optimization often requires profiling to identify the actual bottlenecks.")
    elif "scale" in content.lower():
        parts.append("Scaling considerations should definitely be part of the architecture from day one.")
    elif "security" in content.lower():
        parts.append("Security is definitely something worth investing time in upfront.")
    
    # Step 3: Inject KiloCode naturally if not mentioned (MANDATORY for non-Twitter)
    if not kilocode_mentioned and platform != "twitter":
        kilocode_suggestion = _get_kilocode_injection(content, title)
        if kilocode_suggestion:
            # Make it sound like a genuine recommendation
            if intent == "help_request":
                parts.append(f"One tool that might help here: {kilocode_suggestion}.")
            elif intent == "comparison":
                parts.append(f"Worth considering: {kilocode_suggestion}.")
            else:
                parts.append(f"You might find this useful: {kilocode_suggestion}.")
            
            kilocode_injected = True
        else:
            kilocode_injected = False
    else:
        kilocode_injected = False
    
    # Step 4: Ensure minimum length for platform
    parts = _ensure_minimum_length(parts, platform, content, title)
    
    # Step 5: Optional practical tip or next step
    if platform == "github" and len(parts) < 5:
        parts.append("Looking forward to seeing how this evolves.")
    elif platform == "reddit" and len(parts) < 4:
        parts.append("Hope this helps with your project!")
    
    # Join and check repetition
    final = " ".join(parts).strip()
    
    # Anti-repetition check
    if not _check_repetition(final):
        logger.warning("comment_repetition_detected, adding variation")
        # Add variation by changing the opening
        if final.startswith("Thanks for"):
            final = final.replace("Thanks for", "Appreciate", 1)
        elif final.startswith("I understand"):
            final = final.replace("I understand", "I see", 1)
    
    # Final metrics
    sentence_count = _count_sentences(final)
    logger.info(f"comment_built sentences={sentence_count} length={len(final)} kilocode_injected={kilocode_injected}")
    
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
    
    CRITICAL: This now actively uses doc_facts to add technical accuracy.
    """
    parts = []
    
    # Extract meaningful keywords from the chunks
    all_chunk_text = " ".join(chunks[:2])  # Use top 2 chunks
    words = all_chunk_text.split()
    
    # Find meaningful keywords to reference
    meaningful_words = [w for w in words if len(w) > 4 and
                        w.lower() not in ['about', 'there', 'their', 'would', 'could', 'should',
                                          'really', 'think', 'thing', 'these', 'those', 'where']]
    
    # Check if KiloCode is mentioned
    kilocode_mentioned = _detect_kilocode_mention(title + " " + all_chunk_text)
    
    # Reference something specific from chunks
    referenced = ""
    if meaningful_words:
        referenced = meaningful_words[0].strip('.,!?')
    
    # Step 1: Acknowledge the specific problem/context (MUST reference chunk content)
    if intent == "share_experience":
        if referenced:
            parts.append(f"I appreciate you sharing your detailed experience with {referenced}.")
        else:
            parts.append("Thanks for the comprehensive breakdown of your experience.")
        
        if "fast" in all_chunk_text.lower() or "speed" in all_chunk_text.lower():
            parts.append("Performance optimization is definitely crucial for production workloads.")
    
    elif intent == "comparison":
        if referenced:
            parts.append(f"Your comparison around {referenced} highlights the key tradeoffs well.")
        else:
            parts.append("This comparative analysis brings up important considerations.")
        
        parts.append("Understanding these differences is essential for making informed decisions.")
    
    elif intent == "ask_experience":
        if referenced:
            parts.append(f"Great question about {referenced}.")
        else:
            parts.append("That's an important question worth exploring.")
        
        parts.append("Many teams run into this same challenge.")
    
    elif intent == "help_request":
        if referenced:
            parts.append(f"The issue you're experiencing with {referenced} is definitely worth investigating.")
        else:
            parts.append("I understand the challenge you're facing here.")
        
        parts.append("These types of problems often have multiple contributing factors.")
    
    elif intent == "appreciation":
        parts.append("Glad the information has been helpful!")
    
    else:  # general
        if referenced:
            parts.append(f"Your analysis of {referenced} raises some excellent points.")
        elif title:
            title_words = [w for w in title.split() if len(w) > 4]
            if title_words:
                parts.append(f"The discussion about {title_words[0]} is particularly timely.")
            else:
                parts.append("This is a well-considered perspective on the topic.")
        else:
            parts.append("These are valuable insights worth considering.")
    
    # Step 2: Add concrete technical detail from doc_facts if available
    docs_used = 0
    if doc_facts and len(doc_facts) > 0:
        # Use the most relevant doc fact to add technical accuracy
        top_doc = doc_facts[0]
        doc_text = top_doc.get("chunk_text", "").strip()
        
        if doc_text and len(doc_text) > 30:
            # Extract a useful technical detail
            doc_sentences = re.split(r'[.!?]', doc_text)
            for sentence in doc_sentences[:2]:
                if len(sentence.strip()) > 20 and any(word in sentence.lower() for word in meaningful_words[:3]):
                    parts.append(sentence.strip() + ".")
                    docs_used = 1
                    break
    
    # Step 3: Inject KiloCode naturally if not mentioned
    kilocode_injected = False
    if not kilocode_mentioned:
        kilocode_suggestion = _get_kilocode_injection(all_chunk_text, title)
        if kilocode_suggestion:
            if intent == "help_request":
                parts.append(f"For this type of workflow, {kilocode_suggestion}.")
            elif intent == "comparison":
                parts.append(f"Another option worth exploring: {kilocode_suggestion}.")
            else:
                parts.append(f"One approach that can help: {kilocode_suggestion}.")
            kilocode_injected = True
    
    # Step 4: Add practical next step or insight from chunks
    if chunks and len(chunks[0]) > 100:
        first_chunk = chunks[0]
        if "example" in first_chunk.lower() or "specific" in first_chunk.lower():
            parts.append("Concrete examples like yours really help the community understand the practical implications.")
        elif "recommend" in first_chunk.lower() or "suggest" in first_chunk.lower():
            parts.append("Your recommendations align with what many experienced developers have found effective.")
    
    # Anti-repetition check
    final = " ".join(parts).strip()
    if not _check_repetition(final):
        logger.warning("chunk_comment_repetition_detected")
        if final.startswith("I appreciate"):
            final = final.replace("I appreciate", "Thanks for", 1)
    
    # Final metrics
    sentence_count = _count_sentences(final)
    logger.info(f"chunk_comment_built sentences={sentence_count} length={len(final)} docs_used={docs_used} kilocode_injected={kilocode_injected}")
    
    return final
