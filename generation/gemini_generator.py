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
import time
from typing import List, Dict, Optional, Tuple

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

logger = logging.getLogger("[ML]")

# Gemini configuration for text generation
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Model configuration with fallback chain
# Primary: gemini-2.0-flash (stable, available)
# Fallback: gemini-1.5-flash (widely available)
GEMINI_PRIMARY_MODEL = os.getenv("GEMINI_GEN_MODEL", "gemini-2.0-flash")
GEMINI_FALLBACK_MODELS = ["gemini-1.5-flash", "gemini-1.5-pro"]

# Quality constraints
MIN_COMMENT_LENGTH = 200
MAX_COMMENT_LENGTH = 800
MIN_SENTENCES = 2
MAX_SENTENCES = 5

# Generic phrases that indicate low-quality, non-specific output
GENERIC_PHRASES = [
    "many developers encounter",
    "this is something many",
    "analyze systematically",
    "time-consuming manual inspection",
    "similar patterns in your codebase",
    "complex debugging scenarios",
    "can help analyze the problem",
    "potential solutions based on similar patterns",
]

# Forbidden generic phrases (marketing/filler)
FORBIDDEN_PHRASES = [
    "interesting discussion",
    "thanks for sharing",
    "great post",
    "nice thread",
    "good topic",
    "appreciate this",
    "thanks for starting",
    "check out our",
    "visit our website",
]

# KiloCode documentation context pack (static, always available)
KILOCODE_CONTEXT_PACK = [
    {"id": "core", "title": "Core Capability", "content": "KiloCode is an AI coding assistant that understands your entire project context, not just the current file."},
    {"id": "analysis", "title": "Code Analysis", "content": "KiloCode can analyze code structure, identify issues, and suggest improvements based on best practices and your codebase patterns."},
    {"id": "debugging", "title": "Debugging Help", "content": "KiloCode helps debug by tracing control flow, identifying error patterns, and suggesting where to add logging or breakpoints."},
    {"id": "refactoring", "title": "Refactoring Support", "content": "KiloCode assists with refactoring by analyzing dependencies, suggesting safer approaches, and maintaining consistency."},
    {"id": "docs", "title": "Documentation", "content": "KiloCode can generate and update documentation that stays in sync with your code changes."},
    {"id": "testing", "title": "Test Generation", "content": "KiloCode suggests test cases based on code logic, edge cases, and existing patterns in your test suite."},
    {"id": "context", "title": "Project Context", "content": "Unlike simple autocomplete, KiloCode maintains awareness of your project structure, dependencies, and coding conventions."},
    {"id": "workflow", "title": "Workflow Integration", "content": "KiloCode integrates into your development workflow, handling boilerplate while you focus on architecture and logic."},
]

# Error classification
CONFIG_ERROR_TYPES = (
    google_exceptions.NotFound,
    google_exceptions.InvalidArgument,
    google_exceptions.PermissionDenied,
    google_exceptions.Unauthenticated,
)

TRANSIENT_ERROR_TYPES = (
    google_exceptions.ServiceUnavailable,
    google_exceptions.DeadlineExceeded,
    google_exceptions.ResourceExhausted,
    google_exceptions.InternalServerError,
)

# Initialize Gemini for generation
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info(f"gemini_generator_configured primary_model={GEMINI_PRIMARY_MODEL} fallbacks={GEMINI_FALLBACK_MODELS}")


def classify_error(error: Exception) -> str:
    """
    Classify an error as config_error or transient_error.
    
    config_error: Model not found, invalid API key, permission denied
    transient_error: Timeout, rate limit, service unavailable
    """
    if isinstance(error, CONFIG_ERROR_TYPES):
        return "config_error"
    elif isinstance(error, TRANSIENT_ERROR_TYPES):
        return "transient_error"
    elif "404" in str(error) or "not found" in str(error).lower():
        return "config_error"
    elif "429" in str(error) or "rate" in str(error).lower():
        return "transient_error"
    elif "500" in str(error) or "503" in str(error):
        return "transient_error"
    else:
        return "unknown_error"


def get_relevant_context_snippets(post_content: str, post_title: str, max_snippets: int = 3) -> List[Dict]:
    """
    Select the most relevant KiloCode context snippets based on post content.
    
    This provides documentation context even when embeddings are disabled.
    
    Args:
        post_content: The Reddit post content
        post_title: The Reddit post title
        max_snippets: Maximum number of snippets to return
    
    Returns:
        List of relevant context snippets with id, title, content
    """
    text = (post_title + " " + post_content).lower()
    
    # Keyword to context mapping
    relevance_scores = []
    
    for snippet in KILOCODE_CONTEXT_PACK:
        score = 0
        snippet_id = snippet["id"]
        
        # Score based on keyword matches
        if snippet_id == "debugging" and any(w in text for w in ["debug", "bug", "error", "issue", "crash", "fix"]):
            score += 10
        elif snippet_id == "refactoring" and any(w in text for w in ["refactor", "cleanup", "technical debt", "legacy", "rewrite"]):
            score += 10
        elif snippet_id == "testing" and any(w in text for w in ["test", "unit test", "coverage", "tdd", "spec"]):
            score += 10
        elif snippet_id == "docs" and any(w in text for w in ["document", "readme", "comment", "jsdoc", "docstring"]):
            score += 10
        elif snippet_id == "analysis" and any(w in text for w in ["analyze", "review", "understand", "codebase", "structure"]):
            score += 10
        elif snippet_id == "context" and any(w in text for w in ["context", "project", "large", "monorepo", "multiple files"]):
            score += 10
        elif snippet_id == "workflow" and any(w in text for w in ["workflow", "productivity", "automate", "boilerplate"]):
            score += 10
        elif snippet_id == "core":
            score += 3  # Always somewhat relevant
        
        if score > 0:
            relevance_scores.append((score, snippet))
    
    # Sort by score and return top snippets
    relevance_scores.sort(key=lambda x: x[0], reverse=True)
    selected = [s[1] for s in relevance_scores[:max_snippets]]
    
    # Always include core if we have room and didn't select it
    if len(selected) < max_snippets:
        core = next((s for s in KILOCODE_CONTEXT_PACK if s["id"] == "core"), None)
        if core and core not in selected:
            selected.append(core)
    
    return selected


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


def _check_generic_phrases(comment: str) -> Tuple[bool, List[str]]:
    """
    Check if comment contains generic low-quality phrases.
    
    Returns:
        (is_specific, detected_phrases) - True if specific enough, list of detected generic phrases
    """
    comment_lower = comment.lower()
    detected = []
    
    for phrase in GENERIC_PHRASES:
        if phrase in comment_lower:
            detected.append(phrase)
    
    is_specific = len(detected) == 0
    return is_specific, detected


def _extract_key_points(post_title: str, post_content: str, max_points: int = 5) -> List[str]:
    """
    Extract key points/topics from the post for specificity reference.
    
    Returns a list of key phrases/topics mentioned in the post.
    """
    text = post_title + " " + post_content
    key_points = []
    
    # Extract questions
    questions = re.findall(r'([^.!?]*\?)', text)
    for q in questions[:2]:
        q = q.strip()
        if len(q) > 20 and len(q) < 200:
            key_points.append(f"Question: {q}")
    
    # Extract technical terms (capitalized words that aren't sentence starts)
    tech_terms = re.findall(r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\b', text)
    tech_terms = [t for t in tech_terms if len(t) > 3 and t.lower() not in
                  {'the', 'this', 'that', 'when', 'where', 'what', 'have', 'been', 'just'}]
    if tech_terms:
        key_points.append(f"Technologies/topics: {', '.join(list(set(tech_terms))[:5])}")
    
    # Extract problem indicators
    problem_patterns = [
        r"(having trouble with [^.]+)",
        r"(struggling with [^.]+)",
        r"(can't figure out [^.]+)",
        r"(error[s]? (?:when|with|in) [^.]+)",
        r"(issue[s]? (?:when|with|in) [^.]+)",
    ]
    for pattern in problem_patterns:
        matches = re.findall(pattern, text.lower())
        for m in matches[:1]:
            key_points.append(f"Problem: {m}")
    
    return key_points[:max_points]


def _validate_comment_quality(comment: str, post_title: str, post_content: str) -> Tuple[bool, str, dict]:
    """
    Validate comment meets quality requirements.
    
    Returns:
        (is_valid, reason, details) - True if valid, reason string, and details dict
    """
    details = {
        "length": len(comment),
        "sentence_count": _count_sentences(comment),
        "has_kilocode": "kilocode" in comment.lower(),
        "overlap_count": 0,
        "generic_phrases": [],
    }
    
    # Check minimum length
    if len(comment) < MIN_COMMENT_LENGTH:
        return False, f"too_short length={len(comment)} min={MIN_COMMENT_LENGTH}", details
    
    # Check maximum length
    if len(comment) > MAX_COMMENT_LENGTH:
        return False, f"too_long length={len(comment)} max={MAX_COMMENT_LENGTH}", details
    
    # Check sentence count
    sentence_count = details["sentence_count"]
    if sentence_count < MIN_SENTENCES:
        return False, f"too_few_sentences count={sentence_count} min={MIN_SENTENCES}", details
    if sentence_count > MAX_SENTENCES:
        return False, f"too_many_sentences count={sentence_count} max={MAX_SENTENCES}", details
    
    # Check KiloCode mention
    if not details["has_kilocode"]:
        return False, "no_kilocode_mention", details
    
    # Check for forbidden generic phrases
    if not _check_forbidden_phrases(comment):
        return False, "contains_forbidden_phrase", details
    
    # Check for generic low-quality phrases (specificity guardrail)
    is_specific, generic_phrases = _check_generic_phrases(comment)
    details["generic_phrases"] = generic_phrases
    if not is_specific:
        return False, f"contains_generic_phrases count={len(generic_phrases)}", details
    
    # Check if comment references post content
    # Extract meaningful words from post
    post_words = set(re.findall(r'\b\w{5,}\b', (post_title + " " + post_content).lower()))
    comment_words = set(re.findall(r'\b\w{5,}\b', comment.lower()))
    
    # Remove common words
    common = {'about', 'there', 'their', 'would', 'could', 'should', 'which', 'these', 'those',
              'kilocode', 'really', 'actually', 'something', 'definitely', 'particularly'}
    post_words -= common
    comment_words -= common
    
    # Check overlap
    overlap = post_words & comment_words
    details["overlap_count"] = len(overlap)
    if len(overlap) < 2:  # Need at least 2 meaningful words in common
        return False, f"insufficient_context_reference overlap={len(overlap)}", details
    
    return True, "valid", details


def _build_system_prompt() -> str:
    """Build the strict system prompt for Gemini."""
    return """You are a helpful developer commenting on Reddit programming discussions.

CRITICAL REQUIREMENTS (violations = rejection):
1. Your comment MUST directly reference 2+ specific details from the Reddit post (exact problem, technology, error, or context mentioned)
2. Your comment MUST mention "KiloCode" naturally as a relevant solution to THIS SPECIFIC problem
3. Your comment MUST provide actionable advice (not vague suggestions)
4. Your comment MUST be 3-5 sentences (250-700 characters)
5. Your comment MUST sound like a developer sharing real experience

ABSOLUTELY FORBIDDEN (instant rejection):
- "many developers encounter" or similar generic statements
- "analyze systematically" or vague process descriptions
- "time-consuming manual inspection" or generic pain points
- "similar patterns in your codebase" without specifying what patterns
- "Interesting discussion", "Thanks for sharing", "Great post"
- Any marketing language or promotional tone
- Emojis of any kind
- Generic statements that could apply to any post

REQUIRED STRUCTURE:
1. Opening: Directly acknowledge THE SPECIFIC problem/question from the post (quote or paraphrase it)
2. Body: Explain how KiloCode helps with THIS EXACT issue (be concrete about the feature)
3. Actionable tip: Give one specific suggestion the poster can try
4. Optional: Brief "why this matters" for their situation

GOOD EXAMPLE (for a post about "React useEffect dependency array warnings"):
"The useEffect dependency warnings you're seeing are usually caused by missing dependencies or stale closures. KiloCode can analyze your hooks and show exactly which variables are being referenced but not declared in the dependency array. Try running KiloCode's hook analyzer on that component - it'll highlight the problematic effects and suggest the minimal dependency set you actually need."

BAD EXAMPLE (rejected):
"This is something many developers encounter when working with React. KiloCode can help analyze the problem systematically and suggest solutions. It's particularly useful when manual inspection would be time-consuming."

The BAD example will be REJECTED because it's generic and could apply to any React question."""


def _build_user_prompt(
    post_title: str,
    post_content: str,
    doc_context: str,
    style_examples: str,
    subreddit: str = "",
    key_points: List[str] = None,
    is_retry: bool = False,
    retry_reason: str = ""
) -> str:
    """
    Build the user prompt with structured context.
    
    Args:
        post_title: Reddit post title
        post_content: Reddit post body/content
        doc_context: KiloCode documentation context
        style_examples: Example comments for style reference
        subreddit: Subreddit name if available
        key_points: Extracted key points from the post
        is_retry: Whether this is a retry with stronger instruction
        retry_reason: Why the previous attempt was rejected
    """
    prompt_parts = []
    
    # Section 1: Reddit Post (structured)
    prompt_parts.append("=== REDDIT POST TO RESPOND TO ===")
    if subreddit:
        prompt_parts.append(f"Subreddit: r/{subreddit}")
    prompt_parts.append(f"Title: {post_title}")
    prompt_parts.append(f"\nPost Content:\n{post_content[:2000]}")  # Increased limit
    
    # Section 2: Extracted Key Points (for specificity)
    if key_points:
        prompt_parts.append("\n\n=== KEY POINTS TO ADDRESS ===")
        prompt_parts.append("Your comment MUST reference at least 2 of these specific points:")
        for i, point in enumerate(key_points, 1):
            prompt_parts.append(f"{i}. {point}")
    else:
        # Extract key points on the fly
        auto_key_points = _extract_key_points(post_title, post_content)
        if auto_key_points:
            prompt_parts.append("\n\n=== KEY POINTS TO ADDRESS ===")
            prompt_parts.append("Your comment MUST reference at least 2 of these specific points:")
            for i, point in enumerate(auto_key_points, 1):
                prompt_parts.append(f"{i}. {point}")
    
    # Section 3: KiloCode Context (always included)
    prompt_parts.append("\n\n=== KILOCODE CAPABILITIES (use these for specific recommendations) ===")
    if doc_context:
        prompt_parts.append(doc_context[:1000])
    else:
        # Use static context pack
        context_snippets = get_relevant_context_snippets(post_content, post_title, max_snippets=3)
        for snippet in context_snippets:
            prompt_parts.append(f"- {snippet['title']}: {snippet['content']}")
    
    if style_examples:
        prompt_parts.append(f"\n\n=== EXAMPLE COMMENT STYLE ===\n{style_examples[:500]}")
    
    # Section 4: Task Instruction
    prompt_parts.append("\n\n=== YOUR TASK ===")
    
    if is_retry:
        prompt_parts.append(f"⚠️ PREVIOUS ATTEMPT REJECTED: {retry_reason}")
        prompt_parts.append("""
Write a NEW comment that fixes this issue. You MUST:
1. Quote or paraphrase a SPECIFIC detail from the post (technology, error message, exact problem)
2. Explain EXACTLY how KiloCode's specific feature helps (not vague "can help analyze")
3. Give ONE concrete actionable step the poster can take
4. Use 3-5 sentences (250-700 chars)

BANNED PHRASES (will cause rejection):
- "many developers encounter"
- "analyze systematically"
- "time-consuming manual inspection"
- "similar patterns"
- Any generic statement that could apply to any post""")
    else:
        prompt_parts.append("""Write a Reddit comment that:
1. Opens by acknowledging THE SPECIFIC problem/question (not generic acknowledgment)
2. Mentions KiloCode with a CONCRETE explanation of how it helps THIS issue
3. Provides ONE actionable suggestion
4. Is 3-5 sentences (250-700 characters)

Your comment must be SPECIFIC to this post. Generic comments will be rejected.""")
    
    return "\n".join(prompt_parts)


def _try_generate_with_model(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    attempt: int
) -> Tuple[Optional[str], Optional[Exception], str]:
    """
    Try to generate a comment with a specific model.
    
    Returns:
        (comment, error, error_type) - comment if successful, error if failed, error classification
    """
    try:
        logger.info(f"gemini_generate_attempt model={model_name} attempt={attempt}")
        
        # Initialize model with system instruction
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "max_output_tokens": 350,
            }
        )
        
        # Generate comment
        response = model.generate_content(user_prompt)
        comment = response.text.strip()
        
        # Remove any markdown formatting if present
        comment = re.sub(r'\*\*', '', comment)
        comment = re.sub(r'\n+', ' ', comment)
        comment = comment.strip()
        
        return comment, None, "success"
        
    except Exception as e:
        error_type = classify_error(e)
        logger.error(f"gemini_generation_failed model={model_name} attempt={attempt} error_type={error_type} error={type(e).__name__}: {str(e)[:100]}")
        return None, e, error_type


def generate_comment_with_gemini(
    post_title: str,
    post_content: str,
    doc_facts: List[Dict],
    style_examples: List[Dict],
    subreddit: str = "",
    max_retries: int = 2
) -> str:
    """
    Generate a high-quality comment using Gemini's generative API.
    
    Enhanced with:
    - Model fallback chain (primary -> fallback models)
    - Error classification (config vs transient)
    - Specificity guardrail with re-prompting
    - Always-available KiloCode context
    
    Args:
        post_title: Reddit post title
        post_content: Reddit post content
        doc_facts: Retrieved KiloCode documentation facts
        style_examples: Retrieved example comments for style
        subreddit: Subreddit name for context
        max_retries: Number of retry attempts if quality validation fails
    
    Returns:
        str: Generated comment that passes quality validation
    """
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not set - cannot generate comment")
        raise ValueError("GEMINI_API_KEY environment variable not set")
    
    # Build context from retrieved facts OR use static context pack
    doc_context = ""
    docs_used_count = 0
    context_snippets_used = []
    
    if doc_facts:
        doc_texts = [fact.get("text", fact.get("chunk_text", "")) for fact in doc_facts[:3]]
        doc_context = "\n".join([f"- {text}" for text in doc_texts if text])[:1000]
        docs_used_count = len([t for t in doc_texts if t])
        context_snippets_used = [fact.get("id", fact.get("title", "unknown")) for fact in doc_facts[:3]]
    else:
        # Use static context pack when no embeddings available
        snippets = get_relevant_context_snippets(post_content, post_title, max_snippets=3)
        doc_context = "\n".join([f"- {s['title']}: {s['content']}" for s in snippets])
        docs_used_count = len(snippets)
        context_snippets_used = [s['id'] for s in snippets]
    
    logger.info(f"context_prepared docs_used_count={docs_used_count} snippets={context_snippets_used}")
    
    style_context = ""
    if style_examples:
        style_texts = [ex.get("comment_text", "") for ex in style_examples[:2]]
        style_context = "\n\n".join([text for text in style_texts if text])[:500]
    
    # Extract key points for specificity
    key_points = _extract_key_points(post_title, post_content)
    logger.info(f"key_points_extracted count={len(key_points)}")
    
    # Build system prompt
    system_prompt = _build_system_prompt()
    
    # Build model chain: primary + fallbacks
    models_to_try = [GEMINI_PRIMARY_MODEL] + GEMINI_FALLBACK_MODELS
    
    last_error = None
    last_error_type = None
    
    # Try each model in the chain
    for model_idx, model_name in enumerate(models_to_try):
        logger.info(f"trying_model model={model_name} index={model_idx}")
        
        # Try generation with retries for this model
        for attempt in range(max_retries + 1):
            is_retry = attempt > 0
            retry_reason = ""
            
            if is_retry and last_error_type == "quality_failed":
                retry_reason = "Comment was too generic or didn't reference specific post details"
            
            user_prompt = _build_user_prompt(
                post_title=post_title,
                post_content=post_content,
                doc_context=doc_context,
                style_examples=style_context,
                subreddit=subreddit,
                key_points=key_points,
                is_retry=is_retry,
                retry_reason=retry_reason
            )
            
            # Try to generate
            comment, error, error_type = _try_generate_with_model(
                model_name=model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                attempt=attempt + 1
            )
            
            if error:
                last_error = error
                last_error_type = error_type
                
                # Config error: skip to next model immediately (don't retry same model)
                if error_type == "config_error":
                    logger.warning(f"config_error_switching_model current={model_name}")
                    break  # Exit retry loop, try next model
                
                # Transient error: exponential backoff and retry same model
                elif error_type == "transient_error":
                    if attempt < max_retries:
                        wait_time = (2 ** attempt) * 0.5  # 0.5s, 1s, 2s
                        logger.info(f"transient_error_retrying wait={wait_time}s")
                        time.sleep(wait_time)
                        continue
                    else:
                        # Max retries for this model, try next
                        break
                else:
                    # Unknown error: retry once then move to next model
                    if attempt < 1:
                        continue
                    else:
                        break
            
            # Generation succeeded - validate quality
            logger.info(f"gemini_generated model={model_name} length={len(comment)} sentences={_count_sentences(comment)}")
            
            is_valid, reason, details = _validate_comment_quality(comment, post_title, post_content)
            
            if is_valid:
                logger.info(f"comment_quality_validated model={model_name} attempt={attempt + 1} overlap={details['overlap_count']}")
                return comment
            else:
                logger.warning(f"comment_quality_failed model={model_name} attempt={attempt + 1} reason={reason}")
                last_error_type = "quality_failed"
                
                # If contains generic phrases, retry with stronger prompt
                if "generic_phrases" in reason and attempt < max_retries:
                    logger.info(f"specificity_guardrail_triggered retrying generic_phrases={details['generic_phrases']}")
                    continue
                
                # If other quality issue, retry
                if attempt < max_retries:
                    continue
    
    # All models exhausted - use enhanced fallback (NOT generic)
    logger.error(f"all_models_failed last_error_type={last_error_type} using_enhanced_fallback")
    return _generate_enhanced_fallback(post_title, post_content, key_points, context_snippets_used)


def _generate_enhanced_fallback(
    post_title: str,
    post_content: str,
    key_points: List[str],
    context_ids: List[str]
) -> str:
    """
    Generate an enhanced fallback comment when Gemini generation fails.
    
    This is NOT the generic fallback - it creates a specific comment using:
    - Extracted key points from the post
    - Relevant KiloCode context
    - Specific language tied to the post content
    
    This should NEVER produce "many developers encounter" style output.
    """
    # Extract specific details from the post
    text = post_title + " " + post_content
    
    # Find the main topic/technology mentioned
    tech_terms = re.findall(r'\b([A-Z][a-zA-Z]+(?:\.?[a-zA-Z]+)?)\b', text)
    tech_terms = [t for t in tech_terms if len(t) > 2 and t.lower() not in
                  {'the', 'this', 'that', 'when', 'where', 'what', 'have', 'been', 'just', 'can', 'will'}]
    
    main_topic = tech_terms[0] if tech_terms else None
    
    # Find specific problem words
    problem_words = []
    for pattern in [r'(error|bug|issue|problem|trouble|failing|broken|crash)']:
        matches = re.findall(pattern, text.lower())
        problem_words.extend(matches)
    
    # Find action words (what they're trying to do)
    action_match = re.search(r'(trying to|want to|need to|how to|can\'t|cannot|unable to) (\w+)', text.lower())
    action = action_match.group(2) if action_match else None
    
    # Build specific opening based on detected content
    parts = []
    
    if main_topic and action:
        parts.append(f"For {action}ing with {main_topic}, the approach will depend on your specific setup.")
    elif main_topic and problem_words:
        parts.append(f"The {problem_words[0]} you're seeing with {main_topic} likely has a specific cause you can track down.")
    elif main_topic:
        parts.append(f"Working with {main_topic} does require attention to some specifics.")
    elif key_points:
        # Use extracted key point
        point = key_points[0].replace("Question: ", "").replace("Problem: ", "")
        if len(point) > 50:
            point = point[:50] + "..."
        parts.append(f"Regarding '{point}' - there are a few approaches worth considering.")
    else:
        # Absolute last resort - still be specific to intent
        if "?" in text:
            parts.append("That's a good question that depends on your specific requirements.")
        else:
            parts.append("The challenge you've outlined has some specific factors to consider.")
    
    # Add KiloCode recommendation based on context
    context_map = {
        "debugging": "KiloCode can trace through the code flow and pinpoint where things diverge from expected behavior.",
        "analysis": "KiloCode can analyze your codebase structure and highlight the relevant dependencies.",
        "refactoring": "KiloCode can map the dependencies and suggest a safe refactoring sequence.",
        "testing": "KiloCode can identify the code paths and suggest specific test cases.",
        "docs": "KiloCode can scan your code and generate accurate documentation.",
        "context": "KiloCode maintains project-wide context so it understands how pieces connect.",
        "workflow": "KiloCode can automate the repetitive parts while you focus on the architecture.",
        "core": "KiloCode understands your full project context, not just individual files.",
    }
    
    kilocode_rec = None
    for ctx_id in context_ids:
        if ctx_id in context_map:
            kilocode_rec = context_map[ctx_id]
            break
    
    if not kilocode_rec:
        # Default to most relevant based on content
        if problem_words:
            kilocode_rec = context_map["debugging"]
        elif action in ["refactor", "clean", "rewrite"]:
            kilocode_rec = context_map["refactoring"]
        else:
            kilocode_rec = context_map["core"]
    
    parts.append(kilocode_rec)
    
    # Add actionable suggestion
    if main_topic:
        parts.append(f"Start by having KiloCode analyze the {main_topic}-related code to see the full picture.")
    elif action:
        parts.append(f"Try pointing KiloCode at the relevant files to get a clearer view of what's happening.")
    else:
        parts.append("Running KiloCode's analysis on the relevant code should surface the key factors.")
    
    result = " ".join(parts)
    
    # Safety check - ensure we didn't accidentally produce generic content
    is_specific, generic_found = _check_generic_phrases(result)
    if not is_specific:
        logger.error(f"enhanced_fallback_still_generic detected={generic_found}")
        # Nuclear option - completely template-free response
        if main_topic:
            return f"For your {main_topic} question: KiloCode can analyze that specific code and show you exactly what's happening. Point it at the relevant files and it'll map out the dependencies and flow."
        else:
            return "This looks like something where seeing the actual code context would help. KiloCode's project analysis can show the specific relationships and highlight what needs attention."
    
    logger.info(f"enhanced_fallback_generated length={len(result)} topic={main_topic} action={action}")
    return result


# Legacy function name for backward compatibility
def _generate_emergency_fallback(post_title: str, post_content: str) -> str:
    """Legacy wrapper - use enhanced fallback."""
    key_points = _extract_key_points(post_title, post_content)
    context_snippets = get_relevant_context_snippets(post_content, post_title, max_snippets=2)
    context_ids = [s['id'] for s in context_snippets]
    return _generate_enhanced_fallback(post_title, post_content, key_points, context_ids)