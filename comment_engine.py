import logging
import re
from fastapi import HTTPException
from retrieval import search_by_name
from generation.gemini_generator import (
    generate_comment_with_gemini,
    get_relevant_context_snippets,
    KILOCODE_CONTEXT_PACK,
)
from generation.prompt_builder import detect_intent, detect_twitter_intent
from text_utils import clean_text, chunk_text

logger = logging.getLogger("[ML]")
mem_logger = logging.getLogger("[ML][MEM]")

# CRITICAL SAFETY LIMITS - Prevent RAM spikes on Render (512MB)
MAX_CONTENT_LEN = 5000  # Max chars for synchronous processing
MAX_CHUNKS = 2          # Max chunks to prevent explosion


def generate_comment(post, embedder, top_k_style=5, top_k_docs=5, fetch_status="success"):
    """
    Generate a platform-aware comment with strict memory safety.
    
    CRITICAL SAFETY RULES:
    - Reddit: NEVER uses embeddings (always lightweight)
    - Twitter: NEVER uses embeddings (always lightweight)
    - Content > MAX_CONTENT_LEN: Rejected with 413 error
    
    Args:
        post: NormalizedPost object
        embedder: Embedder instance (may not be used)
        top_k_style: Number of style examples (ignored for lightweight path)
        top_k_docs: Number of docs (ignored for lightweight path)
        fetch_status: Status from fetch_post_content
    
    Returns:
        str: Generated comment
    """
    platform = post.platform
    content = post.content.strip()
    title = post.title.strip()
    content_len = len(content)
    
    logger.info(f"generate_comment platform={platform} content_len={content_len} fetch_status={fetch_status}")
    
    # HARD SAFETY LIMIT: Reject content that's too long
    if content_len > MAX_CONTENT_LEN:
        logger.warning(f"content_too_long content_len={content_len} max={MAX_CONTENT_LEN}")
        raise HTTPException(
            status_code=413,
            detail=f"Post too long for synchronous processing (max {MAX_CONTENT_LEN} chars)"
        )
    
    # CRITICAL: Twitter never uses embeddings
    if platform == "twitter":
        logger.info("embeddings_skipped=true reason=twitter_platform")
        return generate_twitter_comment(post, embedder, fetch_status)
    
    # Reddit: Uses Gemini generation with static context (no embeddings, but not lightweight)
    if platform == "reddit":
        logger.info(f"reddit_gemini_path platform={platform} content_len={content_len}")
        return generate_reddit_comment(post, title, content)
    
    # GitHub with short content uses lightweight
    if platform == "github" and content_len < 1500:
        logger.info(f"embeddings_skipped=true reason=short_content platform={platform} content_len={content_len}")
        return generate_lightweight_comment(post, title, content)
    
    # Only proceed with embeddings for non-Reddit platforms with longer content
    # This path should rarely be hit on Render (512MB)
    if content_len >= 1500:
        logger.warning(f"embeddings_required=true platform={platform} content_len={content_len}")
        return generate_long_form_comment(post, embedder, top_k_style, top_k_docs, fetch_status)
    
    # Default: lightweight without embeddings
    logger.info("embeddings_skipped=true reason=default_lightweight")
    return generate_lightweight_comment(post, title, content)


def extract_subreddit(url: str) -> str:
    """Extract subreddit name from Reddit URL."""
    match = re.search(r'reddit\.com/r/([^/]+)', url)
    return match.group(1) if match else ""


def generate_reddit_comment(post, title: str, content: str):
    """
    Generate comment for Reddit using Gemini with static KiloCode context.
    
    This path:
    - Does NOT use embeddings (RAM safe)
    - DOES use static KiloCode context pack (specific, not generic)
    - DOES pass subreddit for context
    - Uses enhanced Gemini generator with model fallback
    
    This is the FIX for the regression where docs_used=0 produced generic comments.
    """
    logger.info(f"reddit_comment_path title_len={len(title)} content_len={len(content)}")
    
    if not title and not content:
        return "Thanks for starting this discussion!"
    
    # Extract subreddit for context
    subreddit = extract_subreddit(post.url) if hasattr(post, 'url') else ""
    
    # Get relevant KiloCode context snippets (static, no embeddings)
    context_snippets = get_relevant_context_snippets(content, title, max_snippets=3)
    context_snippet_ids = [s['id'] for s in context_snippets]
    
    logger.info(f"context_selected subreddit={subreddit} snippets={context_snippet_ids}")
    
    # Convert static snippets to doc_facts format for the generator
    doc_facts = [
        {"id": s['id'], "title": s['title'], "text": s['content'], "chunk_text": s['content']}
        for s in context_snippets
    ]
    
    try:
        comment = generate_comment_with_gemini(
            post_title=title,
            post_content=content,
            doc_facts=doc_facts,  # NOW includes context!
            style_examples=[],
            subreddit=subreddit,
            max_retries=2
        )
        
        # Enhanced diagnostic logging
        sentence_count = len([s for s in re.split(r'[.!?]+', comment.strip()) if s.strip()])
        kilocode_in_comment = "kilocode" in comment.lower()
        
        logger.info(
            f"comment_generated "
            f"platform={post.platform} "
            f"subreddit={subreddit} "
            f"comment_length={sentence_count}_sentences "
            f"char_length={len(comment)} "
            f"kilocode_injected={kilocode_in_comment} "
            f"docs_used={len(doc_facts)} "
            f"context_snippets={context_snippet_ids} "
            f"embeddings_used=false "
            f"generation=gemini"
        )
        
        return comment
        
    except Exception as e:
        logger.error(f"reddit_comment_failed error={type(e).__name__}: {str(e)[:100]}")
        # Use enhanced fallback from gemini_generator (NOT generic)
        from generation.gemini_generator import _generate_enhanced_fallback, _extract_key_points
        key_points = _extract_key_points(title, content)
        return _generate_enhanced_fallback(title, content, key_points, context_snippet_ids)


def generate_lightweight_comment(post, title: str, content: str):
    """
    Generate comment using Gemini AI WITHOUT embeddings or vector retrieval.
    
    Only used for GitHub short content now. Reddit uses generate_reddit_comment.
    """
    logger.info(f"lightweight_comment_path title_len={len(title)} content_len={len(content)}")
    
    if not title and not content:
        return "Thanks for starting this discussion!"
    
    # Get relevant KiloCode context snippets (static, no embeddings)
    context_snippets = get_relevant_context_snippets(content, title, max_snippets=2)
    doc_facts = [
        {"id": s['id'], "title": s['title'], "text": s['content'], "chunk_text": s['content']}
        for s in context_snippets
    ]
    
    try:
        comment = generate_comment_with_gemini(
            post_title=title,
            post_content=content,
            doc_facts=doc_facts,
            style_examples=[],
            max_retries=1
        )
        
        sentence_count = len([s for s in re.split(r'[.!?]+', comment.strip()) if s.strip()])
        kilocode_in_comment = "kilocode" in comment.lower()
        
        logger.info(
            f"comment_generated "
            f"platform={post.platform} "
            f"comment_length={sentence_count}_sentences "
            f"char_length={len(comment)} "
            f"kilocode_injected={kilocode_in_comment} "
            f"docs_used={len(doc_facts)} "
            f"embeddings_used=false "
            f"generation=gemini"
        )
        
        return comment
    except Exception as e:
        logger.error(f"lightweight_comment_failed error={type(e).__name__}")
        from generation.gemini_generator import _generate_enhanced_fallback, _extract_key_points
        key_points = _extract_key_points(title, content)
        context_ids = [s['id'] for s in context_snippets]
        return _generate_enhanced_fallback(title, content, key_points, context_ids)


def generate_twitter_comment(post, embedder, fetch_status):
    """
    Generate Twitter-specific comment.
    - NEVER uses embeddings
    - NEVER uses documentation chunks
    - 1-2 sentence conversational replies only
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
    comment = build_twitter_comment(text, twitter_intent, text_length)
    
    # Enhanced diagnostic logging
    import re
    sentence_count = len([s for s in re.split(r'[.!?]+', comment.strip()) if s.strip()])
    kilocode_in_comment = "kilocode" in comment.lower()
    
    logger.info(
        f"[ML] comment_generated "
        f"platform=twitter "
        f"comment_length={sentence_count} sentences "
        f"char_length={len(comment)} "
        f"kilocode_injected={kilocode_in_comment} "
        f"docs_used=0 "
        f"embeddings_used=false"
    )
    
    return comment


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


def generate_long_form_comment(post, embedder, top_k_style, top_k_docs, fetch_status):
    """
    Generate comment for long-form content using Gemini + embeddings.
    
    This uses embeddings to retrieve relevant context, then uses Gemini
    to generate a high-quality, contextual comment.
    
    WARNING: This triggers embeddings (uses Gemini API for embeddings).
    Should only be called for non-Reddit platforms with content >= 1500 chars.
    """
    title = post.title.strip()
    content = post.content.strip()
    
    logger.warning(f"longform_path platform={post.platform} content_len={len(content)}")
    mem_logger.info("before_embeddings")
    
    try:
        # Clean and chunk with HARD LIMITS
        cleaned_content = clean_text(content, max_length=MAX_CONTENT_LEN)
        chunks = chunk_text(cleaned_content, chunk_chars=1000, overlap=150, max_chunks=8)
        
        # ENFORCE MAX_CHUNKS safety limit
        if len(chunks) > MAX_CHUNKS:
            logger.warning(f"chunk_limit_exceeded num_chunks={len(chunks)} max={MAX_CHUNKS}")
            chunks = chunks[:MAX_CHUNKS]
        
        logger.info(f"chunked num_chunks={len(chunks)}")
        
        if not chunks:
            logger.warning("chunk_empty_falling_back")
            mem_logger.info("embeddings_skipped reason=no_chunks")
            return generate_lightweight_comment(post, title, content[:500])
        
        # Embed chunks (uses Gemini Embeddings API)
        query_text = f"{title} {chunks[0][:500]}"
        top_chunks = embedder.embed_chunked(chunks=chunks, query=query_text, top_k=3)
        
        mem_logger.info("after_embeddings")
        logger.info(f"top_chunks selected={len(top_chunks)}")
        
        # Retrieve style/docs using embeddings
        style_examples = []
        doc_facts = []
        
        if fetch_status == "success" and top_chunks:
            retrieval_query = f"{title} {top_chunks[0][0][:300]}"
            
            style_examples = search_by_name(
                query=retrieval_query,
                index_name="comments",
                embedder=embedder,
                top_k=top_k_style,
            )
            
            doc_facts = search_by_name(
                query=retrieval_query,
                index_name="docs",
                embedder=embedder,
                top_k=top_k_docs,
            )
            
            logger.info(f"retrieved style={len(style_examples)} docs={len(doc_facts)}")
        
        # Use top chunks as content
        chunk_content = " ".join([chunk for chunk, score in top_chunks])
        
        # Generate comment using Gemini with retrieved context
        comment = generate_comment_with_gemini(
            post_title=title,
            post_content=chunk_content[:1500],  # Use chunked content
            doc_facts=doc_facts,
            style_examples=style_examples,
            max_retries=1
        )
        
        # Enhanced diagnostic logging
        import re
        sentence_count = len([s for s in re.split(r'[.!?]+', comment.strip()) if s.strip()])
        kilocode_in_comment = "kilocode" in comment.lower()
        
        logger.info(
            f"[ML] comment_generated "
            f"platform={post.platform} "
            f"comment_length={sentence_count} sentences "
            f"char_length={len(comment)} "
            f"kilocode_injected={kilocode_in_comment} "
            f"docs_used={len(doc_facts)} "
            f"examples_used={len(style_examples)} "
            f"embeddings_used=true "
            f"generation=gemini"
        )
        
        return comment
    
    except MemoryError as e:
        mem_logger.error(f"OOM_in_longform error={type(e).__name__}")
        # Emergency fallback without embeddings
        return generate_lightweight_comment(post, title, content[:500])
    
    except Exception as e:
        logger.error(f"longform_failed error={type(e).__name__}")
        # Fallback to lightweight
        return generate_lightweight_comment(post, title, content[:500])
