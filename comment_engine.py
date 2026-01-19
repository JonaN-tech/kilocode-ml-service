import logging
from retrieval import search_by_name
from generation.prompt_builder import build_comment, detect_intent, detect_twitter_intent
from text_utils import clean_text, chunk_text

logger = logging.getLogger("[ML]")
mem_logger = logging.getLogger("[ML][MEM]")


def generate_comment(post, embedder, top_k_style=5, top_k_docs=5, fetch_status="success"):
    """
    Generate a platform-aware comment with proper handling for each platform.
    
    Args:
        post: NormalizedPost object
        embedder: Embedder instance
        top_k_style: Number of style examples to retrieve
        top_k_docs: Number of documentation facts to retrieve
        fetch_status: Status from fetch_post_content
    
    Returns:
        str: Generated comment
    """
    platform = post.platform
    logger.info(f"generate_comment platform={platform} fetch_status={fetch_status}")
    
    # Handle Twitter specifically - never use docs, short conversational replies
    if platform == "twitter":
        return generate_twitter_comment(post, embedder, fetch_status)
    
    # Handle Reddit, GitHub, and other platforms with chunked processing
    return generate_long_form_comment(post, embedder, top_k_style, top_k_docs, fetch_status)


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


def generate_long_form_comment(post, embedder, top_k_style, top_k_docs, fetch_status):
    """
    Generate comment for long-form content (Reddit, GitHub, etc.).
    Uses chunked processing for memory safety on large posts.
    
    Process:
    1. Clean the full body text
    2. Chunk it into manageable pieces
    3. Embed chunks incrementally and find most relevant ones
    4. Generate comment from title + top chunks
    """
    title = post.title.strip()
    content = post.content.strip()
    
    has_content = bool(content)
    has_title = bool(title)
    
    logger.info(f"longform has_content={has_content} has_title={has_title} fetch_status={fetch_status}")
    
    # Handle empty content case
    if not has_content and not has_title:
        logger.warning("longform_no_content")
        return "Thanks for starting this discussion!"
    
    # Title-only fallback
    if not has_content:
        logger.info("longform_title_only")
        return generate_title_only_comment(title)
    
    try:
        # Step 1: Clean the full body (still use full content, just remove noise)
        mem_logger.info(f"clean_start content_len={len(content)}")
        cleaned_content = clean_text(content, max_length=25000)
        logger.info(f"cleaned content_len={len(content)} -> {len(cleaned_content)}")
        
        # Step 2: Chunk the cleaned content
        mem_logger.info("chunk_start")
        chunks = chunk_text(cleaned_content, chunk_chars=1000, overlap=150, max_chunks=12)
        logger.info(f"chunked num_chunks={len(chunks)}")
        
        if not chunks:
            logger.warning("chunk_empty_falling_back_to_title")
            return generate_title_only_comment(title)
        
        # Step 3: Find most relevant chunks using embedder
        mem_logger.info("embed_chunks_start")
        query_text = f"{title}\n\n{chunks[0][:500]}"  # Use title + first bit for query
        
        top_chunks = embedder.embed_chunked(
            chunks=chunks,
            query=query_text,
            top_k=3
        )
        
        logger.info(f"top_chunks selected={len(top_chunks)}")
        
        # Step 4: Retrieve style examples and docs (still useful for context)
        style_examples = []
        doc_facts = []
        
        if fetch_status == "success" and top_chunks:
            try:
                mem_logger.info("retrieval_start")
                
                # Use top chunk for retrieval
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
                
                mem_logger.info("retrieval_complete")
                logger.info(f"retrieved style={len(style_examples)} docs={len(doc_facts)}")
            
            except Exception as e:
                logger.warning(f"retrieval_failed error={type(e).__name__}")
                # Continue without retrieval
        
        # Step 5: Build final comment
        comment = build_comment(
            post=post,
            title=title,
            top_chunks=[chunk for chunk, score in top_chunks],
            style_examples=style_examples,
            doc_facts=doc_facts,
            fetch_status=fetch_status,
        )
        
        logger.info(f"comment_generated length={len(comment)}")
        return comment
    
    except MemoryError as e:
        mem_logger.error(f"OOM_in_chunked_processing error={type(e).__name__}")
        # Emergency fallback: use only title + first 300 chars
        return generate_emergency_fallback(title, content[:300])
    
    except Exception as e:
        logger.error(f"longform_generation_failed error={type(e).__name__}")
        # Fallback to title-based comment
        if title:
            return generate_title_only_comment(title)
        else:
            return "This is an interesting discussion. Thanks for sharing!"


def generate_title_only_comment(title: str) -> str:
    """Generate a comment when only title is available."""
    if not title:
        return "Thanks for starting this discussion!"
    
    # Extract key topics from title
    words = title.split()
    key_topics = [w for w in words if len(w) > 3 and w.lower() not in 
                  {'this', 'that', 'with', 'from', 'have', 'they', 'their', 'what', 'about'}]
    
    if key_topics:
        topic_str = ' '.join(key_topics[:2])
        return f"Interesting post about {topic_str}! Thanks for starting this discussion."
    else:
        return "Thanks for starting this discussion!"


def generate_emergency_fallback(title: str, content_snippet: str) -> str:
    """
    Emergency fallback for OOM or extreme errors.
    Uses minimal processing on title + snippet.
    """
    logger.info("emergency_fallback_triggered")
    
    if title and content_snippet:
        # Extract a keyword from content
        words = content_snippet.split()
        keywords = [w for w in words if len(w) > 4]
        if keywords:
            return f"Regarding {title}: {keywords[0]} is definitely worth discussing further."
    
    if title:
        return f"Thanks for posting about {title}!"
    
    return "Thanks for sharing this!"
