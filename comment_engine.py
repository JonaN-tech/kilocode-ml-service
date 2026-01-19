import logging
from retrieval import search_by_name
from generation.prompt_builder import build_comment, detect_intent, detect_twitter_intent

logger = logging.getLogger("[ML]")


def generate_comment(post, embedder, top_k_style=5, top_k_docs=5, fetch_status="success"):
    """
    Generate a platform-aware comment with proper handling for each platform.
    
    Args:
        post: NormalizedPost object
        embedder: Embedder instance
        top_k_style: Number of style examples to retrieve
        top_k_docs: Number of documentation facts to retrieve
        fetch_status: Status from fetch_post_content ("success", "http_error", "timeout", "blocked", "error")
    
    Returns:
        str: Generated comment
    """
    platform = post.platform
    logger.info(f"generate_comment platform={platform} fetch_status={fetch_status}")
    
    # Handle Twitter specifically - never use docs, short conversational replies
    if platform == "twitter":
        return generate_twitter_comment(post, embedder, fetch_status)
    
    # Handle Reddit and other platforms
    return generate_reddit_comment(post, embedder, top_k_style, top_k_docs, fetch_status)


def generate_twitter_comment(post, embedder, fetch_status):
    """
    Generate Twitter-specific comment.
    - NEVER use documentation chunks for Twitter
    - Generate 1-2 sentence conversational replies
    - Handle mentions, links, short/non-English text
    """
    text = post.content.strip()
    text_length = len(text)
    logger.info(f"twitter text_length={text_length}")
    
    # Detect Twitter-specific intent
    twitter_intent = detect_twitter_intent(text)
    logger.info(f"twitter_intent={twitter_intent}")
    
    # Handle edge cases
    if not text:
        logger.info(f"twitter_empty_content")
        return "Interesting! Thanks for sharing."
    
    # Check if it's mostly mentions (@username)
    words = text.split()
    mention_ratio = sum(1 for w in words if w.startswith('@')) / max(len(words), 1)
    
    # Check if it's mostly links
    link_ratio = sum(1 for w in words if w.startswith('http')) / max(len(words), 1)
    
    if mention_ratio > 0.5:
        logger.info(f"twitter_intent=mention")
        return "Good point! Interesting perspective on this."
    
    if link_ratio > 0.5:
        logger.info(f"twitter_intent=link_share")
        return "Thanks for sharing this resource!"
    
    # Generate conversational reply based on detected intent
    return build_twitter_comment(text, twitter_intent, text_length)


def build_twitter_comment(text, intent, text_length):
    """Build a Twitter-specific comment based on intent."""
    
    # Extract a keyword or phrase from the tweet to reference
    words = text.lower().split()
    
    # Find interesting words (exclude common stopwords)
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
                 'it', 'we', 'they', 'what', 'which', 'who', 'whom', 'and', 'or', 'but'}
    
    interesting_words = [w for w in words if w not in stopwords and len(w) > 3]
    
    # Generate intent-aware response
    if intent == "question":
        return f"That's an interesting question! {' '.join(interesting_words[:2]).title()} is definitely worth exploring."
    
    elif intent == "announcement":
        return f"Great update on {' '.join(interesting_words[:2]).title()}! Thanks for sharing."
    
    elif intent == "comparison":
        return f"The comparison is really insightful, especially regarding {' '.join(interesting_words[:2]).title()}."
    
    elif intent == "link_share":
        return "Thanks for sharing this link! Looks interesting."
    
    elif intent == "mention":
        return "Good point! Thanks for raising this."
    
    else:  # general
        # Reference something specific from the tweet
        if interesting_words:
            referenced = ' '.join(interesting_words[:2]).title()
            return f"I appreciate your thoughts on {referenced}. It's a thoughtful perspective!"
        else:
            return "Thanks for sharing this! Interesting perspective."


def generate_reddit_comment(post, embedder, top_k_style, top_k_docs, fetch_status):
    """
    Generate Reddit/other platform comment.
    - Prefer post title + body
    - If body fetch fails, fall back to title-only
    - Never hard-fail, always return valid comment
    """
    # Check if we have content
    has_content = bool(post.content.strip())
    has_title = bool(post.title.strip())
    
    logger.info(f"reddit has_content={has_content} has_title={has_title} fetch_status={fetch_status}")
    
    if not has_content:
        logger.info(f"reddit_fallback=title_only")
        # Generate from title only
        query_text = post.title.strip()
    else:
        query_text = f"{post.title}\n\n{post.content}".strip()
    
    # Determine what to search for
    if not has_content or fetch_status != "success":
        logger.info(f"reddit_no_body_fallback")
        # Don't search docs if we don't have content
        style_examples = []
        doc_facts = []
    else:
        # Normal retrieval
        style_examples = search_by_name(
            query=query_text,
            index_name="comments",
            embedder=embedder,
            top_k=top_k_style,
        )
        
        doc_facts = search_by_name(
            query=query_text,
            index_name="docs",
            embedder=embedder,
            top_k=top_k_docs,
        )
        
        logger.info(f"retrieved style_examples={len(style_examples)} doc_facts={len(doc_facts)}")
    
    # Build the comment
    comment = build_comment(
        post=post,
        style_examples=style_examples,
        doc_facts=doc_facts,
        fetch_status=fetch_status,
    )
    
    logger.info(f"reddit_comment_generated length={len(comment)}")
    return comment
