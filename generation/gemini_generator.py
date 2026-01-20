"""
Gemini-based comment generation with strict quality constraints.

This module uses Gemini's generative AI API (FREE tier) to create
contextual, specific comments that reference the Reddit post and
introduce KiloCode naturally.

CRITICAL RULES:
- Comments MUST reference specific details from the post
- Comments MUST mention KiloCode meaningfully
- Comments MUST be 2-5 sentences (200-800 chars)
- Generic/promotional phrases are FORBIDDEN
"""
import os
import re
import logging
from typing import List, Dict, Optional

import google.generativeai as genai

logger = logging.getLogger("[ML]")

# Gemini configuration for text generation
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_GEN_MODEL = os.getenv("GEMINI_GEN_MODEL", "gemini-2.0-flash-exp")  # FREE tier generative model

# Quality constraints
MIN_COMMENT_LENGTH = 200
MAX_COMMENT_LENGTH = 800
MIN_SENTENCES = 2
MAX_SENTENCES = 5

# Forbidden generic phrases
FORBIDDEN_PHRASES = [
    "interesting discussion",
    "thanks for sharing",
    "great post",
    "nice thread",
    "good topic",
    "appreciate this",
    "thanks for starting",
]

# Initialize Gemini for generation
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info(f"gemini_generator_configured model={GEMINI_GEN_MODEL}")


def _count_sentences(text: str) -> int:
    """Count sentences in text."""
    sentences = re.split(r'[.!?]+', text.strip())
    return len([s for s in sentences if s.strip()])


def _check_forbidden_phrases(comment: str) -> bool:
    """
    Check if comment contains forbidden generic phrases.
    Returns True if comment is acceptable, False if it contains forbidden phrases.
    """
    comment_lower = comment.lower()
    for phrase in FORBIDDEN_PHRASES:
        if phrase in comment_lower:
            logger.warning(f"forbidden_phrase_detected phrase='{phrase}'")
            return False
    return True


def _validate_comment_quality(comment: str, post_title: str, post_content: str) -> tuple[bool, str]:
    """
    Validate comment meets quality requirements.
    
    Returns:
        (is_valid, reason) - True if valid, False with reason if invalid
    """
    # Check minimum length
    if len(comment) < MIN_COMMENT_LENGTH:
        return False, f"too_short length={len(comment)} min={MIN_COMMENT_LENGTH}"
    
    # Check maximum length
    if len(comment) > MAX_COMMENT_LENGTH:
        return False, f"too_long length={len(comment)} max={MAX_COMMENT_LENGTH}"
    
    # Check sentence count
    sentence_count = _count_sentences(comment)
    if sentence_count < MIN_SENTENCES:
        return False, f"too_few_sentences count={sentence_count} min={MIN_SENTENCES}"
    if sentence_count > MAX_SENTENCES:
        return False, f"too_many_sentences count={sentence_count} max={MAX_SENTENCES}"
    
    # Check KiloCode mention
    if "kilocode" not in comment.lower():
        return False, "no_kilocode_mention"
    
    # Check for forbidden generic phrases
    if not _check_forbidden_phrases(comment):
        return False, "contains_forbidden_phrase"
    
    # Check if comment references post content
    # Extract meaningful words from post
    post_words = set(re.findall(r'\b\w{5,}\b', (post_title + " " + post_content).lower()))
    comment_words = set(re.findall(r'\b\w{5,}\b', comment.lower()))
    
    # Remove common words
    common = {'about', 'there', 'their', 'would', 'could', 'should', 'which', 'these', 'those'}
    post_words -= common
    comment_words -= common
    
    # Check overlap
    overlap = post_words & comment_words
    if len(overlap) < 2:  # Need at least 2 meaningful words in common
        return False, f"insufficient_context_reference overlap={len(overlap)}"
    
    return True, "valid"


def _build_system_prompt() -> str:
    """Build the strict system prompt for Gemini."""
    return """You are a helpful developer commenting on Reddit programming discussions.

STRICT RULES (MANDATORY):
1. Your comment MUST reference at least one concrete detail from the Reddit post (problem, tool, question, or context)
2. Your comment MUST mention "KiloCode" as a relevant tool or solution
3. Your comment MUST be helpful (provide solution, suggestion, or insight)
4. Your comment MUST be 2-5 sentences (200-800 characters)
5. Your comment MUST sound like a developer helping another developer

FORBIDDEN (your response will be rejected if it contains these):
- "Interesting discussion"
- "Thanks for sharing"
- "Great post"
- "Nice thread"
- Generic acknowledgments
- Marketing language
- Emojis
- Call-to-action phrases like "Check out our website"

TONE REQUIREMENTS:
- Sound human and natural
- Be specific and technical
- Reference actual content from the post
- Introduce KiloCode naturally as a useful tool
- No promotional language
- No emojis or excessive enthusiasm

EXAMPLE OF GOOD OUTPUT:
For a post about debugging legacy code:
"Debugging legacy systems without proper documentation is definitely challenging. I've found that tools like KiloCode can help by analyzing the codebase structure and suggesting where to add logging or breakpoints based on the control flow. It's particularly useful when you're dealing with unfamiliar architectures where manual tracing would take hours."

Notice: References specific problem (legacy debugging), mentions KiloCode naturally, provides concrete value, sounds like a developer."""


def _build_user_prompt(
    post_title: str,
    post_content: str,
    doc_context: str,
    style_examples: str,
    is_retry: bool = False
) -> str:
    """
    Build the user prompt with structured context.
    
    Args:
        post_title: Reddit post title
        post_content: Reddit post body/content
        doc_context: KiloCode documentation context
        style_examples: Example comments for style reference
        is_retry: Whether this is a retry with stronger instruction
    """
    prompt_parts = []
    
    # Section 1: Reddit Post
    prompt_parts.append("=== REDDIT POST ===")
    prompt_parts.append(f"Title: {post_title}")
    prompt_parts.append(f"\nContent:\n{post_content[:1500]}")  # Truncate if needed
    
    # Identify the problem/question
    prompt_parts.append("\nKey Challenge/Question:")
    if "?" in post_content:
        # Extract question
        sentences = re.split(r'[.!?]+', post_content)
        questions = [s.strip() for s in sentences if '?' in s]
        if questions:
            prompt_parts.append(questions[0][:200])
    else:
        # Summarize main topic
        first_sentence = post_content.split('.')[0] if '.' in post_content else post_content[:200]
        prompt_parts.append(first_sentence.strip())
    
    # Section 2: KiloCode Context
    prompt_parts.append("\n\n=== KILOCODE CONTEXT ===")
    if doc_context:
        prompt_parts.append(f"Relevant KiloCode Documentation:\n{doc_context[:800]}")
    else:
        prompt_parts.append("KiloCode is an AI-powered coding assistant that understands project context and helps with development tasks.")
    
    if style_examples:
        prompt_parts.append(f"\n\nExample Comment Style (for reference):\n{style_examples[:400]}")
    
    # Section 3: Instruction
    prompt_parts.append("\n\n=== YOUR TASK ===")
    
    if is_retry:
        prompt_parts.append("""Write a helpful Reddit comment that:
1. DIRECTLY addresses the specific problem/question mentioned above
2. References at least 2 concrete details from the post content
3. Naturally introduces KiloCode as a solution to THIS SPECIFIC problem
4. Provides actionable advice or insight
5. Is 2-5 sentences long

BE MORE SPECIFIC. Avoid generic statements. Reference the actual technical problem.""")
    else:
        prompt_parts.append("""Write a helpful Reddit comment that directly addresses the problem described above and naturally introduces KiloCode as a solution.

Your comment must:
- Reference specific details from the post
- Explain how KiloCode helps with THIS problem
- Sound like a developer sharing their experience
- Be 2-5 sentences (200-800 characters)

Do NOT use generic phrases like "interesting discussion" or "thanks for sharing".""")
    
    return "\n".join(prompt_parts)


def generate_comment_with_gemini(
    post_title: str,
    post_content: str,
    doc_facts: List[Dict],
    style_examples: List[Dict],
    max_retries: int = 1
) -> str:
    """
    Generate a high-quality comment using Gemini's generative API.
    
    Args:
        post_title: Reddit post title
        post_content: Reddit post content
        doc_facts: Retrieved KiloCode documentation facts
        style_examples: Retrieved example comments for style
        max_retries: Number of retry attempts if quality validation fails
    
    Returns:
        str: Generated comment that passes quality validation
    """
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not set - cannot generate comment")
        raise ValueError("GEMINI_API_KEY environment variable not set")
    
    # Build context from retrieved facts
    doc_context = ""
    if doc_facts:
        doc_texts = [fact.get("text", fact.get("chunk_text", "")) for fact in doc_facts[:3]]
        doc_context = "\n".join([text for text in doc_texts if text])[:800]
    
    style_context = ""
    if style_examples:
        style_texts = [ex.get("comment_text", "") for ex in style_examples[:2]]
        style_context = "\n\n".join([text for text in style_texts if text])[:400]
    
    # Build prompts
    system_prompt = _build_system_prompt()
    
    # Try generation with retries
    for attempt in range(max_retries + 1):
        is_retry = attempt > 0
        user_prompt = _build_user_prompt(
            post_title=post_title,
            post_content=post_content,
            doc_context=doc_context,
            style_examples=style_context,
            is_retry=is_retry
        )
        
        try:
            logger.info(f"gemini_generate_attempt attempt={attempt + 1} is_retry={is_retry}")
            
            # Initialize model with system instruction
            model = genai.GenerativeModel(
                model_name=GEMINI_GEN_MODEL,
                system_instruction=system_prompt,
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40,
                    "max_output_tokens": 300,
                }
            )
            
            # Generate comment
            response = model.generate_content(user_prompt)
            comment = response.text.strip()
            
            # Remove any markdown formatting if present
            comment = re.sub(r'\*\*', '', comment)
            comment = re.sub(r'\n+', ' ', comment)
            comment = comment.strip()
            
            logger.info(f"gemini_generated length={len(comment)} sentences={_count_sentences(comment)}")
            
            # Validate quality
            is_valid, reason = _validate_comment_quality(comment, post_title, post_content)
            
            if is_valid:
                logger.info(f"comment_quality_validated attempt={attempt + 1}")
                return comment
            else:
                logger.warning(f"comment_quality_failed attempt={attempt + 1} reason={reason}")
                
                if attempt < max_retries:
                    logger.info("retrying_with_stronger_prompt")
                    continue
                else:
                    # Last attempt failed - use emergency fallback
                    logger.error("all_attempts_failed using_emergency_fallback")
                    return _generate_emergency_fallback(post_title, post_content)
        
        except Exception as e:
            logger.error(f"gemini_generation_failed attempt={attempt + 1} error={type(e).__name__}: {str(e)[:100]}")
            
            if attempt < max_retries:
                continue
            else:
                # All retries exhausted
                return _generate_emergency_fallback(post_title, post_content)
    
    # Should never reach here, but safety fallback
    return _generate_emergency_fallback(post_title, post_content)


def _generate_emergency_fallback(post_title: str, post_content: str) -> str:
    """
    Generate a safe fallback comment when Gemini generation fails.
    
    This still follows the rules:
    - References post content
    - Mentions KiloCode
    - Provides value
    """
    # Extract key topic from title or content
    words = (post_title + " " + post_content).split()
    meaningful = [w for w in words if len(w) > 5 and w.lower() not in 
                  {'about', 'there', 'would', 'could', 'should', 'which', 'really'}]
    
    topic = meaningful[0] if meaningful else "this topic"
    
    # Build a structured fallback
    parts = [
        f"The challenge you're describing with {topic} is something many developers encounter.",
        "KiloCode can help analyze the problem systematically and suggest potential solutions based on similar patterns in your codebase.",
        "It's particularly useful when dealing with complex debugging scenarios where manual inspection would be time-consuming."
    ]
    
    return " ".join(parts)