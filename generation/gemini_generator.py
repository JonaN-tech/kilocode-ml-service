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
    "comprehensive solution",
    "advanced capabilities",
    "powerful tool",
    "optimize workflow",
    "seamless integration",
    "enables you to",
    "provides the ability",
    "leverages advanced",
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
    "that's a great question",
    "great question",
    "good point",
    "check out our",
    "visit our website",
    "our tool",
    "our platform",
    "our solution",
]

# KiloCode documentation context pack (static, always available)
KILOCODE_CONTEXT_PACK = [
    {"id": "core", "title": "Core Capability", "content": "KiloCode understands your whole project context, not just whatever file you're in."},
    {"id": "analysis", "title": "Code Analysis", "content": "KiloCode can check your code structure and spot issues based on what it sees in your codebase."},
    {"id": "debugging", "title": "Debugging Help", "content": "KiloCode's pretty good at tracing through code flow and pointing out where things might be going wrong."},
    {"id": "refactoring", "title": "Refactoring Support", "content": "KiloCode can help with refactoring by checking dependencies so you don't accidentally break stuff."},
    {"id": "docs", "title": "Documentation", "content": "KiloCode generates docs that actually stay in sync when you update your code."},
    {"id": "testing", "title": "Test Generation", "content": "KiloCode can suggest test cases based on your code logic and edge cases you might've missed."},
    {"id": "context", "title": "Project Context", "content": "KiloCode keeps track of your project structure and dependencies, unlike basic autocomplete."},
    {"id": "workflow", "title": "Workflow Integration", "content": "KiloCode handles the boring boilerplate stuff while you focus on the actual architecture."},
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


# Known AI models that are commonly mentioned in developer discussions
KNOWN_AI_MODELS = [
    "opus", "sonnet", "haiku", "claude", "gpt-4", "gpt-4o", "gpt-3.5", "gpt",
    "kimi", "minimax", "gemini", "llama", "mistral", "codestral", "deepseek",
    "qwen", "yi", "phi", "starling", "openchat", "zephyr", "mixtral",
    "o1", "o1-preview", "o1-mini", "cursor", "aider", "copilot", "codex",
    "o3", "o3-mini", "o4-mini", "grok",
]

# Versioned AI model patterns (regex patterns to capture model + version)
VERSIONED_MODEL_PATTERNS = [
    r"opus\s*[\d.]+",        # e.g. "Opus 4.6", "Opus 4"
    r"sonnet\s*[\d.]+",      # e.g. "Sonnet 3.5", "Sonnet 4"
    r"haiku\s*[\d.]+",       # e.g. "Haiku 3.5"
    r"claude\s*[\d.]+",      # e.g. "Claude 3", "Claude 3.5"
    r"gpt[- ]?[\d.o]+",     # e.g. "GPT-4o", "GPT-4", "GPT 3.5"
    r"kimi\s*k?[\d.]+",     # e.g. "Kimi K2.5", "Kimi 2"
    r"minimax\s*m?[\d.]+",  # e.g. "MiniMax M2.5", "MiniMax 2"
    r"gemini\s*[\d.]+",      # e.g. "Gemini 2.0", "Gemini 1.5"
    r"llama\s*[\d.]+",       # e.g. "Llama 3.1", "Llama 3"
    r"mistral\s*[\d.]+",     # e.g. "Mistral 7B"
    r"deepseek\s*v?[\d.]+", # e.g. "DeepSeek V3", "DeepSeek 2.5"
    r"qwen\s*[\d.]+",        # e.g. "Qwen 2.5"
    r"o[134][- ]?(?:mini|preview)?",  # e.g. "o1-mini", "o3", "o4-mini"
]

# Known developer tools commonly discussed
KNOWN_TOOLS = [
    "vscode", "vim", "neovim", "emacs", "jetbrains", "intellij", "pycharm",
    "docker", "kubernetes", "k8s", "terraform", "ansible", "jenkins",
    "github", "gitlab", "bitbucket", "git", "npm", "yarn", "pip", "poetry",
    "webpack", "vite", "esbuild", "babel", "typescript", "javascript",
    "react", "vue", "angular", "svelte", "next.js", "nextjs", "sveltekit",
    "python", "node", "nodejs", "rust", "go", "golang", "java", "kotlin",
    "postgres", "postgresql", "mysql", "mongodb", "redis", "sqlite",
    "aws", "azure", "gcp", "vercel", "netlify", "heroku", "render",
    "langchain", "llamaindex", "pinecone", "weaviate", "chromadb",
    "kilocode", "windsurf", "bolt", "lovable", "v0",
]

# Workflow/approach keywords (including AI coding workflows)
WORKFLOW_KEYWORDS = [
    "tdd", "bdd", "ci/cd", "cicd", "devops", "agile", "scrum",
    "microservices", "monolith", "serverless", "mvc", "mvvm",
    "pair programming", "code review", "pr review", "pull request",
    "refactoring", "migration", "testing", "unit test", "integration test",
    "debugging", "profiling", "optimization", "scaling", "deployment",
    "plan vs build", "plan mode", "build mode", "planning model",
    "implementation model", "multi-agent", "multi agent", "agentic",
    "coding workflow", "coding agent", "ai coding", "ai assistant",
    "vibe coding", "prompt engineering", "chain of thought",
    "model switching", "model pair", "model pairing",
    "code generation", "code completion", "copilot alternative",
    "rag", "retrieval augmented", "context window",
]

# Multi-word workflow/concept phrases to match (checked before single keywords)
WORKFLOW_PHRASES = [
    "plan vs build", "plan and build", "plan mode", "build mode",
    "planning model", "implementation model", "planning phase", "build phase",
    "multi-agent setup", "multi agent setup", "multi-agent workflow",
    "coding workflow", "coding agent", "ai coding assistant",
    "model switching", "model pair", "model pairing",
    "vibe coding", "prompt engineering", "chain of thought",
    "context window", "token usage", "token cost",
    "code generation", "code completion",
]


def _extract_specific_entities(post_title: str, post_content: str) -> Dict[str, List[str]]:
    """
    Extract specific entities mentioned in the post for context relevance.
    
    This extracts:
    - AI models mentioned (Opus, Sonnet, GPT, Kimi, etc.) including versioned names
    - Tools mentioned (VSCode, Docker, etc.)
    - Workflows/approaches mentioned (including multi-word phrases)
    - Specific coding approaches
    
    Returns:
        Dict with keys: 'models', 'tools', 'workflows', 'technologies', 'problems'
    """
    text = (post_title + " " + post_content).lower()
    original_text = post_title + " " + post_content  # Preserve case for display
    entities = {
        "models": [],
        "models_display": [],   # Original-case versions for display
        "tools": [],
        "tools_display": [],
        "workflows": [],
        "technologies": [],
        "problems": [],
    }
    
    # 1. Extract versioned AI models FIRST (higher priority, e.g. "Opus 4.6")
    versioned_found = set()
    for pattern in VERSIONED_MODEL_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            clean = match.strip().lower()
            if clean not in versioned_found:
                versioned_found.add(clean)
                entities["models"].append(clean)
                # Find original-case version
                orig_match = re.search(re.escape(match.strip()), original_text, re.IGNORECASE)
                entities["models_display"].append(orig_match.group(0) if orig_match else match.strip())
    
    # 2. Extract base AI model names (skip if already captured versioned)
    for model in KNOWN_AI_MODELS:
        if model in text:
            # Check not already captured as part of versioned name
            already_captured = any(model in v for v in versioned_found)
            if not already_captured:
                entities["models"].append(model)
                # Find original-case version
                orig_match = re.search(re.escape(model), original_text, re.IGNORECASE)
                entities["models_display"].append(orig_match.group(0) if orig_match else model.title())
    
    # 3. Extract tools mentioned
    for tool in KNOWN_TOOLS:
        # Use word boundary check to avoid partial matches
        if re.search(r'\b' + re.escape(tool) + r'\b', text):
            entities["tools"].append(tool)
            orig_match = re.search(re.escape(tool), original_text, re.IGNORECASE)
            entities["tools_display"].append(orig_match.group(0) if orig_match else tool.title())
    
    # 4. Extract multi-word workflow phrases FIRST (before single keywords)
    phrases_found = set()
    for phrase in WORKFLOW_PHRASES:
        if phrase in text:
            phrases_found.add(phrase)
            entities["workflows"].append(phrase)
    
    # 5. Extract single workflow keywords (skip if part of already-matched phrase)
    for workflow in WORKFLOW_KEYWORDS:
        if workflow in text:
            already_in_phrase = any(workflow in phr for phr in phrases_found)
            if not already_in_phrase and workflow not in entities["workflows"]:
                entities["workflows"].append(workflow)
    
    # 6. Extract additional technologies (capitalized terms)
    tech_terms = re.findall(r'\b([A-Z][a-zA-Z]+(?:\.?[a-zA-Z]+)?)\b', original_text)
    tech_terms = [t.lower() for t in tech_terms if len(t) > 2 and t.lower() not in
                  {'the', 'this', 'that', 'when', 'where', 'what', 'have', 'been', 'just', 'can', 'will', 'with', 'from', 'they', 'their', 'there'}]
    entities["technologies"] = list(set(tech_terms))[:8]
    
    # 7. Extract problem indicators
    problem_patterns = [
        r"(having trouble with [^.]+)",
        r"(struggling with [^.]+)",
        r"(can't figure out [^.]+)",
        r"(error[s]? (?:when|with|in) [^.]+)",
        r"(issue[s]? (?:when|with|in) [^.]+)",
        r"(problem[s]? (?:with|in|when) [^.]+)",
        r"(bug[s]? (?:in|with) [^.]+)",
        r"(crash(?:ing|ed)? (?:when|in|with) [^.]+)",
        r"(fail(?:ing|ed)? (?:when|to|in) [^.]+)",
        r"(too (?:slow|expensive|costly|complex) [^.]*)",
        r"((?:token|cost|price|latency) (?:is|are|seems?) [^.]+)",
    ]
    for pattern in problem_patterns:
        matches = re.findall(pattern, text)
        for m in matches[:2]:
            entities["problems"].append(m.strip())
    
    return entities


def extract_post_context(post_title: str, post_content: str) -> Dict:
    """
    Comprehensive context extraction step performed BEFORE generating a comment.
    
    This function analyzes the Reddit post and produces a structured context
    object that the comment generator uses to stay relevant.
    
    Returns:
        Dict with keys:
            - entities: extracted models, tools, workflows, problems
            - main_topic: one-line summary of the post topic
            - main_question: the main question being asked (if any)
            - discussion_type: type of discussion (comparison, help_request, workflow, experience, etc.)
            - context_elements: list of specific things the comment MUST reference
    """
    text = (post_title + " " + post_content).lower()
    original_text = post_title + " " + post_content
    
    # Step 1: Extract all entities
    entities = _extract_specific_entities(post_title, post_content)
    
    # Step 2: Determine discussion type
    discussion_type = _classify_discussion_type(text, entities)
    
    # Step 3: Extract main question (if any)
    main_question = _extract_main_question(original_text)
    
    # Step 4: Summarize the main topic
    main_topic = _summarize_topic(post_title, entities, discussion_type)
    
    # Step 5: Build list of context elements the comment MUST reference
    context_elements = _build_context_elements(entities, main_question, discussion_type)
    
    context = {
        "entities": entities,
        "main_topic": main_topic,
        "main_question": main_question,
        "discussion_type": discussion_type,
        "context_elements": context_elements,
    }
    
    logger.info(
        f"post_context_extracted "
        f"discussion_type={discussion_type} "
        f"models={entities['models'][:3]} "
        f"tools={entities['tools'][:3]} "
        f"workflows={entities['workflows'][:3]} "
        f"topic={main_topic[:60]} "
        f"question={'yes' if main_question else 'no'} "
        f"context_elements={len(context_elements)}"
    )
    
    return context


def _classify_discussion_type(text: str, entities: Dict) -> str:
    """Classify what kind of discussion the post is."""
    # Check for comparisons (highest priority)
    if any(w in text for w in [" vs ", " versus ", "compared to", "comparison", "better than", "which is better",
                                "which one", "difference between"]):
        return "comparison"
    
    # Check for workflow discussions
    if entities["workflows"] or any(w in text for w in ["workflow", "setup", "pipeline", "process", "approach"]):
        if any(w in text for w in ["plan", "build", "implement", "architect"]):
            return "workflow_planning"
        return "workflow"
    
    # Check for model-specific discussions
    if len(entities["models"]) >= 2:
        return "model_comparison"
    if entities["models"]:
        return "model_discussion"
    
    # Check for help requests
    if any(w in text for w in ["help", "issue", "error", "bug", "problem", "trouble", "struggling",
                                "can't figure", "how do i", "how to"]):
        return "help_request"
    
    # Check for experience sharing
    if any(w in text for w in ["i've been", "my experience", "i found", "i switched", "i tried",
                                "just started", "been using"]):
        return "experience"
    
    # Check for recommendations
    if any(w in text for w in ["recommend", "suggestion", "best", "what do you use", "what's the best"]):
        return "recommendation"
    
    # Check for questions
    if "?" in text:
        return "question"
    
    return "general"


def _extract_main_question(text: str) -> Optional[str]:
    """Extract the main question from the post, if any."""
    questions = re.findall(r'([^.!?\n]*\?)', text)
    if not questions:
        return None
    
    # Filter out very short questions and pick the most substantive
    meaningful_questions = [q.strip() for q in questions if len(q.strip()) > 15]
    if not meaningful_questions:
        return questions[0].strip() if questions else None
    
    # Prefer questions from the title or the first substantial question
    title_questions = [q for q in meaningful_questions if q in text[:200]]
    if title_questions:
        return title_questions[0]
    
    return meaningful_questions[0]


def _summarize_topic(post_title: str, entities: Dict, discussion_type: str) -> str:
    """Create a one-line topic summary from the post title and entities."""
    # Use title as base, enriched with entity information
    topic_parts = []
    
    if post_title:
        topic_parts.append(post_title.strip())
    
    if not topic_parts:
        # Build from entities
        if entities["models"]:
            models_str = ", ".join(entities.get("models_display", entities["models"])[:3])
            topic_parts.append(f"Discussion about {models_str}")
        if entities["workflows"]:
            topic_parts.append(f"({', '.join(entities['workflows'][:2])})")
        if entities["tools"]:
            topic_parts.append(f"using {', '.join(entities.get('tools_display', entities['tools'])[:2])}")
    
    topic = " ".join(topic_parts) if topic_parts else "General developer discussion"
    return topic[:200]


def _build_context_elements(entities: Dict, main_question: Optional[str], discussion_type: str) -> List[str]:
    """
    Build a prioritized list of context elements the comment MUST reference.
    
    These are specific things from the post that the generated comment should mention
    to stay relevant and avoid being generic.
    """
    elements = []
    
    # Add models with display names
    for i, model in enumerate(entities.get("models_display", entities["models"])[:3]):
        elements.append(f"model:{model}")
    
    # Add workflow phrases (high priority for workflow discussions)
    for workflow in entities["workflows"][:3]:
        elements.append(f"workflow:{workflow}")
    
    # Add tools
    for i, tool in enumerate(entities.get("tools_display", entities["tools"])[:3]):
        elements.append(f"tool:{tool}")
    
    # Add main question
    if main_question:
        elements.append(f"question:{main_question[:100]}")
    
    # Add problems
    for problem in entities["problems"][:2]:
        elements.append(f"problem:{problem[:80]}")
    
    return elements


def _extract_key_points(post_title: str, post_content: str, max_points: int = 6) -> List[str]:
    """
    Extract key points/topics from the post for specificity reference.
    
    Enhanced to include specific entities (models, tools, workflows) and
    use display names for better prompt quality.
    
    Returns a list of key phrases/topics mentioned in the post.
    """
    text = post_title + " " + post_content
    key_points = []
    
    # Extract specific entities first (most important for relevance)
    entities = _extract_specific_entities(post_title, post_content)
    
    # Use display names when available for better readability in prompts
    if entities["models"]:
        display_models = entities.get("models_display", entities["models"])[:5]
        key_points.append(f"AI Models mentioned: {', '.join(display_models)}")
    
    if entities["workflows"]:
        key_points.append(f"Workflows/approaches: {', '.join(entities['workflows'][:4])}")
    
    if entities["tools"]:
        display_tools = entities.get("tools_display", entities["tools"])[:5]
        key_points.append(f"Tools mentioned: {', '.join(display_tools)}")
    
    # Extract questions (high priority for response relevance)
    main_question = _extract_main_question(text)
    if main_question:
        key_points.append(f"Main question: {main_question}")
    else:
        questions = re.findall(r'([^.!?]*\?)', text)
        for q in questions[:2]:
            q = q.strip()
            if len(q) > 20 and len(q) < 200:
                key_points.append(f"Question: {q}")
    
    # Extract problem indicators
    if entities["problems"]:
        key_points.append(f"Problem: {entities['problems'][0]}")
    else:
        problem_patterns = [
            r"(having trouble with [^.]+)",
            r"(struggling with [^.]+)",
            r"(can't figure out [^.]+)",
            r"(error[s]? (?:when|with|in) [^.]+)",
            r"(issue[s]? (?:when|with|in) [^.]+)",
            r"(too (?:slow|expensive|costly) [^.]*)",
        ]
        for pattern in problem_patterns:
            matches = re.findall(pattern, text.lower())
            for m in matches[:1]:
                key_points.append(f"Problem: {m}")
    
    # Extract technical terms as fallback (only if we don't have many key points yet)
    if len(key_points) < 3:
        tech_terms = re.findall(r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\b', text)
        tech_terms = [t for t in tech_terms if len(t) > 3 and t.lower() not in
                      {'the', 'this', 'that', 'when', 'where', 'what', 'have', 'been', 'just'}]
        if tech_terms and not entities["technologies"]:
            key_points.append(f"Technologies/topics: {', '.join(list(set(tech_terms))[:5])}")
    
    return key_points[:max_points]


def _validate_comment_quality(comment: str, post_title: str, post_content: str, post_context: Dict = None) -> Tuple[bool, str, dict]:
    """
    Validate comment meets quality requirements with enhanced context relevance checking.
    
    Args:
        comment: The generated comment
        post_title: Original post title
        post_content: Original post content
        post_context: Optional pre-extracted post context from extract_post_context()
    
    Returns:
        (is_valid, reason, details) - True if valid, reason string, and details dict
    """
    details = {
        "length": len(comment),
        "sentence_count": _count_sentences(comment),
        "has_kilocode": "kilocode" in comment.lower(),
        "overlap_count": 0,
        "entity_references": 0,
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
    
    # ENHANCED: Check entity references from post context
    comment_lower = comment.lower()
    entity_ref_count = 0
    
    if post_context:
        entities = post_context.get("entities", {})
        # Check if comment references specific models
        for model in entities.get("models", [])[:5]:
            if model in comment_lower:
                entity_ref_count += 1
        # Check if comment references specific tools
        for tool in entities.get("tools", [])[:5]:
            if tool in comment_lower:
                entity_ref_count += 1
        # Check if comment references workflows
        for workflow in entities.get("workflows", [])[:3]:
            if workflow in comment_lower:
                entity_ref_count += 1
    else:
        # Fallback: extract entities for validation
        entities = _extract_specific_entities(post_title, post_content)
        for model in entities.get("models", [])[:5]:
            if model in comment_lower:
                entity_ref_count += 1
        for tool in entities.get("tools", [])[:5]:
            if tool in comment_lower:
                entity_ref_count += 1
        for workflow in entities.get("workflows", [])[:3]:
            if workflow in comment_lower:
                entity_ref_count += 1
    
    details["entity_references"] = entity_ref_count
    
    # Require at least 1 entity reference if entities were found in the post
    total_entities = len(entities.get("models", [])) + len(entities.get("tools", [])) + len(entities.get("workflows", []))
    if total_entities > 0 and entity_ref_count == 0:
        return False, f"no_entity_references entities_in_post={total_entities} refs_in_comment=0", details
    
    # Check if comment references post content (word overlap)
    post_words = set(re.findall(r'\b\w{5,}\b', (post_title + " " + post_content).lower()))
    comment_words = set(re.findall(r'\b\w{5,}\b', comment_lower))
    
    # Remove common words
    common = {'about', 'there', 'their', 'would', 'could', 'should', 'which', 'these', 'those',
              'kilocode', 'really', 'actually', 'something', 'definitely', 'particularly',
              'worth', 'might', 'pretty', 'think', 'honestly', 'probably', 'through'}
    post_words -= common
    comment_words -= common
    
    # Check overlap
    overlap = post_words & comment_words
    details["overlap_count"] = len(overlap)
    # Require 2 overlapping words minimum (or 1 if entity references are strong)
    min_overlap = 1 if entity_ref_count >= 2 else 2
    if len(overlap) < min_overlap:
        return False, f"insufficient_context_reference overlap={len(overlap)} min={min_overlap}", details
    
    return True, "valid", details


def _build_system_prompt() -> str:
    """Build the strict system prompt for Gemini."""
    return """You are a developer casually replying to another developer on Reddit. Write naturally and conversationally.

MOST IMPORTANT RULE - CONTEXT GROUNDING:
Your reply MUST be clearly about the SPECIFIC topic of the post. A reader should be able to tell
EXACTLY what post you're replying to just from reading your comment.

If the post is about "plan vs build models in coding workflows", your comment must
discuss plan vs build, mention the specific models, and talk about that workflow.
If a comment could apply to ANY random post, it will be REJECTED.

CRITICAL REQUIREMENTS (violations = rejection):
1. Your comment MUST directly reference SPECIFIC entities mentioned in the post:
   - AI models by name (e.g., "Opus", "Sonnet", "GPT-4", "Kimi K2.5", "MiniMax M2.5")
   - Tools by name (e.g., "VSCode", "Docker", "React", "Python", "Next.js")
   - Workflows/approaches (e.g., "plan vs build", "multi-agent", "TDD", "vibe coding")
   - The exact problem or question being discussed
2. Your comment MUST mention "KiloCode" casually as a relevant tool (NOT like a product pitch)
3. Your comment MUST provide a useful observation or suggestion specific to THEIR situation
4. Your comment MUST be 2-4 sentences (200-600 characters)
5. Your comment MUST sound natural and conversational

CONTEXT RELEVANCE RULES:
- If the post mentions "Opus" or "Sonnet", your comment MUST reference those models by name
- If the post discusses "plan vs build" workflows, your comment MUST talk about that workflow
- If the post asks about model pairs, discuss model pairing specifically
- If the post is about token costs, address costs specifically
- NEVER write a comment that could apply to a completely different post
- NEVER ignore the specific topic - respond to what they're ACTUALLY discussing

NATURAL REDDIT TONE:
- Use casual phrases naturally: "tbh", "honestly", "imo", "might be worth", "pretty solid", "kinda"
- Write like you're chatting with a colleague, not writing documentation
- Be slightly informal but still technically aware
- Short sentences are fine. Fragments are okay sometimes.
- Vary your openings - don't always start the same way

HOW TO MENTION KILOCODE (casual, not promotional):
✓ GOOD: "might be worth running it through KiloCode"
✓ GOOD: "KiloCode sometimes catches stuff like that"
✓ GOOD: "Running the whole workflow through KiloCode can make that easier since you can switch models without changing editors"
✓ GOOD: "I'd probably run it through KiloCode and see what it flags"
✓ GOOD: "KiloCode lets you switch models mid-workflow which is handy for plan/build setups"

✗ BAD: "KiloCode provides advanced capabilities"
✗ BAD: "KiloCode offers a comprehensive solution"
✗ BAD: "Our tool can help with this"
✗ BAD: "KiloCode's powerful features enable"

ABSOLUTELY FORBIDDEN (instant rejection):
- "many developers encounter" or similar generic statements
- "comprehensive solution" or "advanced capabilities"
- "powerful tool" or "optimize workflow"
- "seamless integration" or marketing buzzwords
- "analyze systematically" or vague process descriptions
- "Interesting discussion", "Thanks for sharing", "Great post"
- "Great question", "Good point", "That's a great question" as openers
- "Thanks for starting this thread"
- Any corporate or promotional language
- Emojis of any kind
- Generic statements that could apply to any post

REQUIRED STRUCTURE:
1. Opening: Reference the SPECIFIC topic/model/tool/workflow they mentioned (respond to THEIR discussion)
2. Body: Share a useful insight or observation about THEIR specific situation, mentioning KiloCode casually
3. Tip: One specific actionable thing relevant to THEIR exact use case

GOOD EXAMPLES:

For "Plan vs Build models in coding workflows":
"Yeah switching models between plan and build like that can sometimes double token usage. If you're worried about the build phase getting expensive, might be worth running it through KiloCode to see if it's generating overly verbose code."

For "Best planning / implementation model pair":
"Opus tends to be really solid for planning. For the implementation step I've also seen good results with Kimi or MiniMax. Running the whole workflow through KiloCode can make that easier since you can switch models without changing editors."

For "Opus vs Sonnet for coding":
"Honestly both Opus and Sonnet are pretty solid for coding tasks, but Opus tends to handle more complex refactoring better imo. Might be worth running your codebase through KiloCode to see which model catches more issues in your specific setup."

For "React useEffect dependency array warnings":
"If you're running into useEffect dependency warnings a lot, might be worth running those hooks through KiloCode. It sometimes catches stuff that's causing extra renders you didn't expect."

BAD EXAMPLE (REJECTED - generic, doesn't reference specific entities):
"This is something many developers encounter when working with these tools. KiloCode can help analyze the problem systematically and suggest solutions."

BAD EXAMPLE (REJECTED - could apply to ANY post):
"Interesting discussion. Thanks for starting this thread."

RETURN ONLY THE COMMENT TEXT. No explanations, no reasoning, no metadata."""


def _build_user_prompt(
    post_title: str,
    post_content: str,
    doc_context: str,
    style_examples: str,
    subreddit: str = "",
    key_points: List[str] = None,
    post_context: Dict = None,
    is_retry: bool = False,
    retry_reason: str = ""
) -> str:
    """
    Build the user prompt with structured context analysis.
    
    Args:
        post_title: Reddit post title
        post_content: Reddit post body/content
        doc_context: KiloCode documentation context
        style_examples: Example comments for style reference
        subreddit: Subreddit name if available
        key_points: Extracted key points from the post
        post_context: Pre-extracted post context from extract_post_context()
        is_retry: Whether this is a retry with stronger instruction
        retry_reason: Why the previous attempt was rejected
    """
    prompt_parts = []
    
    # Section 1: Reddit Post (structured)
    prompt_parts.append("=== REDDIT POST TO RESPOND TO ===")
    if subreddit:
        prompt_parts.append(f"Subreddit: r/{subreddit}")
    prompt_parts.append(f"Title: {post_title}")
    prompt_parts.append(f"\nPost Content:\n{post_content[:2000]}")
    
    # Section 2: CONTEXT ANALYSIS (new structured analysis section)
    # Use pre-extracted context if available, otherwise extract on the fly
    if post_context:
        ctx = post_context
        entities = ctx["entities"]
    else:
        ctx = extract_post_context(post_title, post_content)
        entities = ctx["entities"]
    
    prompt_parts.append("\n\n=== CONTEXT ANALYSIS (your reply MUST be grounded in this) ===")
    prompt_parts.append(f"Topic: {ctx['main_topic']}")
    prompt_parts.append(f"Discussion type: {ctx['discussion_type']}")
    if ctx.get("main_question"):
        prompt_parts.append(f"Main question: {ctx['main_question']}")
    
    # Section 3: Entities to reference
    prompt_parts.append("\n=== ENTITIES YOU MUST REFERENCE (by name) ===")
    
    if entities.get("models_display") or entities.get("models"):
        display_models = entities.get("models_display", entities["models"])[:5]
        prompt_parts.append(f"AI Models: {', '.join(display_models)}")
    
    if entities.get("workflows"):
        prompt_parts.append(f"Workflows: {', '.join(entities['workflows'][:4])}")
    
    if entities.get("tools_display") or entities.get("tools"):
        display_tools = entities.get("tools_display", entities["tools"])[:5]
        prompt_parts.append(f"Tools: {', '.join(display_tools)}")
    
    if entities.get("problems"):
        prompt_parts.append(f"Problems: {'; '.join(entities['problems'][:2])}")
    
    # Explicit referencing instruction
    entity_count = len(entities.get("models", [])) + len(entities.get("tools", [])) + len(entities.get("workflows", []))
    if entity_count > 0:
        prompt_parts.append(f"\n⚠️ Your comment MUST mention at least 1-2 of these entities BY NAME. Do NOT use vague terms like 'these tools' or 'those models'.")
    
    # Section 4: Key Points (for specificity)
    effective_key_points = key_points or ctx.get("context_elements") or _extract_key_points(post_title, post_content)
    if effective_key_points:
        prompt_parts.append("\n=== KEY CONTEXT ELEMENTS TO ADDRESS ===")
        for i, point in enumerate(effective_key_points[:5], 1):
            prompt_parts.append(f"{i}. {point}")
    
    # Section 5: KiloCode Context (always included)
    prompt_parts.append("\n\n=== KILOCODE CAPABILITIES (use for relevant recommendations) ===")
    prompt_parts.append("KiloCode is a VS Code extension that lets you switch between different AI models mid-workflow.")
    prompt_parts.append("It understands your whole project context, not just individual files.")
    if doc_context:
        prompt_parts.append(doc_context[:800])
    else:
        context_snippets = get_relevant_context_snippets(post_content, post_title, max_snippets=3)
        for snippet in context_snippets:
            prompt_parts.append(f"- {snippet['title']}: {snippet['content']}")
    
    if style_examples:
        prompt_parts.append(f"\n\n=== EXAMPLE COMMENT STYLE ===\n{style_examples[:500]}")
    
    # Section 6: Task Instruction
    prompt_parts.append("\n\n=== YOUR TASK ===")
    
    if is_retry:
        prompt_parts.append(f"⚠️ PREVIOUS ATTEMPT REJECTED: {retry_reason}")
        prompt_parts.append(f"""
Write a NEW comment that DIRECTLY addresses this post about "{ctx['main_topic'][:80]}".

CRITICAL RULES:
1. Your FIRST sentence must reference something specific from the post ({', '.join(entities.get('models_display', entities.get('models', []))[:2] + entities.get('workflows', [])[:1])})
2. Mention KiloCode casually (NOT like marketing)
3. Give ONE concrete tip relevant to their SPECIFIC situation
4. Use 2-4 sentences (200-600 chars)
5. Write like you're texting a developer friend

INSTANT REJECTION:
- "many developers encounter", "comprehensive solution", "advanced capabilities"
- "Interesting discussion", "Great post", "Thanks for sharing", "Great question"
- Any sentence that could apply to a completely different post""")
    else:
        # Build dynamic instruction based on detected entities
        entity_examples = []
        if entities.get("models_display") or entities.get("models"):
            display = entities.get("models_display", entities["models"])
            entity_examples.append(f"models like '{display[0]}'")
        if entities.get("workflows"):
            entity_examples.append(f"workflows like '{entities['workflows'][0]}'")
        if entities.get("tools_display") or entities.get("tools"):
            display = entities.get("tools_display", entities["tools"])
            entity_examples.append(f"tools like '{display[0]}'")
        
        entity_instruction = ""
        if entity_examples:
            entity_instruction = f" Specifically reference {', '.join(entity_examples[:2])}."
        
        prompt_parts.append(f"""Write a casual Reddit reply about "{ctx['main_topic'][:80]}" that:
1. Opens by referencing the SPECIFIC topic they're discussing.{entity_instruction}
2. Mentions KiloCode casually as relevant to their situation
3. Includes ONE useful observation or tip about THEIR specific use case
4. Sounds like a real developer helping another developer (2-4 sentences, 200-600 chars)

The reader should be able to tell EXACTLY which post you're replying to. Output ONLY the comment text.""")
    
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
                "temperature": 0.8,  # Slightly higher for more natural, varied language
                "top_p": 0.92,
                "top_k": 45,
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
    - Context extraction step BEFORE generation
    - Model fallback chain (primary -> fallback models)
    - Error classification (config vs transient)
    - Entity reference validation
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
    
    # STEP 1: Extract post context BEFORE generating
    post_context = extract_post_context(post_title, post_content)
    
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
    last_rejection_reason = ""
    
    # Try each model in the chain
    for model_idx, model_name in enumerate(models_to_try):
        logger.info(f"trying_model model={model_name} index={model_idx}")
        
        # Try generation with retries for this model
        for attempt in range(max_retries + 1):
            is_retry = attempt > 0
            retry_reason = ""
            
            if is_retry and last_error_type == "quality_failed":
                retry_reason = last_rejection_reason or "Comment was too generic or didn't reference specific post details"
            
            user_prompt = _build_user_prompt(
                post_title=post_title,
                post_content=post_content,
                doc_context=doc_context,
                style_examples=style_context,
                subreddit=subreddit,
                key_points=key_points,
                post_context=post_context,
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
            
            # Generation succeeded - validate quality with context
            logger.info(f"gemini_generated model={model_name} length={len(comment)} sentences={_count_sentences(comment)}")
            
            is_valid, reason, details = _validate_comment_quality(
                comment, post_title, post_content, post_context=post_context
            )
            
            if is_valid:
                logger.info(
                    f"comment_quality_validated model={model_name} attempt={attempt + 1} "
                    f"overlap={details['overlap_count']} entity_refs={details['entity_references']}"
                )
                return comment
            else:
                logger.warning(f"comment_quality_failed model={model_name} attempt={attempt + 1} reason={reason}")
                last_error_type = "quality_failed"
                last_rejection_reason = reason
                
                # If no entity references, include that in retry reason
                if "no_entity_references" in reason and attempt < max_retries:
                    logger.info(f"entity_reference_guardrail_triggered")
                    continue
                
                # If contains generic phrases, retry with stronger prompt
                if "generic_phrases" in reason and attempt < max_retries:
                    logger.info(f"specificity_guardrail_triggered retrying generic_phrases={details['generic_phrases']}")
                    continue
                
                # If other quality issue, retry
                if attempt < max_retries:
                    continue
    
    # All models exhausted - use enhanced fallback (NOT generic)
    logger.error(f"all_models_failed last_error_type={last_error_type} using_enhanced_fallback")
    return _generate_enhanced_fallback(post_title, post_content, key_points, context_snippets_used, post_context=post_context)


def _generate_enhanced_fallback(
    post_title: str,
    post_content: str,
    key_points: List[str],
    context_ids: List[str],
    post_context: Dict = None
) -> str:
    """
    Generate an enhanced fallback comment when Gemini generation fails.
    
    This is NOT the generic fallback - it creates a specific comment using:
    - Pre-extracted post context (if available)
    - Extracted entities (models, tools, workflows) from the post
    - Relevant KiloCode context
    - Specific language tied to the post content
    
    This should NEVER produce "many developers encounter" style output.
    """
    # Use pre-extracted context if available
    if post_context:
        entities = post_context["entities"]
        discussion_type = post_context.get("discussion_type", "general")
        main_topic = post_context.get("main_topic", "")
    else:
        entities = _extract_specific_entities(post_title, post_content)
        discussion_type = "general"
        main_topic = post_title
    
    text = post_title + " " + post_content
    
    # Use display names for entities
    def _display_name(entity_list, display_list, idx=0):
        """Get display name for an entity, falling back to title case."""
        if display_list and idx < len(display_list):
            return display_list[idx]
        if entity_list and idx < len(entity_list):
            name = entity_list[idx]
            if name.lower() in ["opus", "sonnet", "haiku", "gpt", "o1", "o3"]:
                return name.upper() if len(name) <= 3 else name.title()
            return name.title()
        return None
    
    # Determine the primary entity to reference
    primary_entity = None
    secondary_entity = None
    entity_type = None
    
    if entities.get("models"):
        primary_entity = _display_name(entities["models"], entities.get("models_display", []), 0)
        secondary_entity = _display_name(entities["models"], entities.get("models_display", []), 1)
        entity_type = "model"
    elif entities.get("workflows"):
        primary_entity = entities["workflows"][0].title()
        entity_type = "workflow"
    elif entities.get("tools"):
        primary_entity = _display_name(entities["tools"], entities.get("tools_display", []), 0)
        entity_type = "tool"
    
    # Find specific problem words
    problem_words = []
    for pattern in [r'(error|bug|issue|problem|trouble|failing|broken|crash|expensive|costly|slow)']:
        matches = re.findall(pattern, text.lower())
        problem_words.extend(matches)
    
    # Find action words (what they're trying to do)
    action_match = re.search(r'(trying to|want to|need to|how to|can\'t|cannot|unable to) (\w+)', text.lower())
    action = action_match.group(2) if action_match else None
    
    # Build specific opening based on detected entities and discussion type
    parts = []
    
    # Opening: Reference specific entities and discussion type
    if discussion_type in ("comparison", "model_comparison") and entities.get("models"):
        models_display = entities.get("models_display", entities["models"])
        if len(models_display) >= 2:
            parts.append(f"Both {models_display[0]} and {models_display[1]} have their strengths tbh.")
        elif primary_entity:
            parts.append(f"{primary_entity} is pretty solid for that use case.")
    elif discussion_type == "workflow_planning" and entities.get("workflows"):
        workflow = entities["workflows"][0]
        parts.append(f"The {workflow} approach can work well, but it really depends on your specific setup.")
    elif discussion_type == "workflow" and entities.get("workflows"):
        workflow = entities["workflows"][0]
        parts.append(f"For {workflow}, honestly the setup matters a lot.")
    elif entity_type == "model" and entities.get("models"):
        if secondary_entity:
            parts.append(f"Both {primary_entity} and {secondary_entity} are pretty solid tbh.")
        else:
            parts.append(f"Working with {primary_entity} can be interesting.")
        if problem_words:
            parts.append(f"If you're hitting {problem_words[0]}s with it, might be worth a closer look.")
    elif entity_type == "tool" and entities.get("tools"):
        if problem_words:
            parts.append(f"Dealing with {problem_words[0]}s in {primary_entity} can be tricky tbh.")
        else:
            parts.append(f"Working with {primary_entity} - couple things to keep in mind.")
    elif entity_type == "workflow":
        parts.append(f"For {primary_entity} stuff, honestly depends on your specific setup.")
    elif key_points:
        # Use extracted key point
        point = key_points[0]
        # Clean up prefixes
        for prefix in ["Question: ", "Problem: ", "AI Models mentioned: ", "Tools mentioned: ", "Workflows/approaches: "]:
            point = point.replace(prefix, "")
        if len(point) > 50:
            point = point[:50] + "..."
        parts.append(f"For the '{point}' question - couple approaches you could try.")
    else:
        # Absolute last resort - still be specific to intent
        if "?" in text:
            parts.append("Good question, honestly depends on what you're dealing with.")
        else:
            parts.append("That's a tricky situation for sure.")
    
    # Add KiloCode recommendation based on context and discussion type
    context_map = {
        "debugging": "KiloCode sometimes catches stuff like that when you point it at the relevant code.",
        "analysis": "Might be worth running KiloCode on it to see the dependencies and how things connect.",
        "refactoring": "KiloCode can help map out the dependencies so you don't break stuff accidentally.",
        "testing": "You could try KiloCode to spot the code paths and edge cases you might've missed.",
        "docs": "KiloCode usually generates decent docs that stay in sync with the code.",
        "context": "KiloCode's pretty good at understanding how your project fits together.",
        "workflow": "KiloCode handles the boring repetitive stuff so you can focus on the architecture.",
        "core": "KiloCode understands your full project context, not just single files.",
    }
    
    # Discussion-type-specific KiloCode recommendations
    discussion_kilocode_map = {
        "comparison": "KiloCode lets you switch between models mid-workflow, so you can try both and see which works better for your use case.",
        "model_comparison": "KiloCode lets you swap models mid-task, so you could actually A/B test both in your real codebase.",
        "workflow_planning": "Running the workflow through KiloCode can make that easier since you can switch models between plan and build phases.",
        "workflow": "KiloCode can help with that since it lets you configure different models for different steps in your workflow.",
        "model_discussion": "Might be worth running it through KiloCode to see how that model handles your specific codebase.",
    }
    
    kilocode_rec = discussion_kilocode_map.get(discussion_type)
    
    if not kilocode_rec:
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
    
    result = " ".join(parts)
    
    # Safety check - ensure we didn't accidentally produce generic content
    is_specific, generic_found = _check_generic_phrases(result)
    if not is_specific:
        logger.error(f"enhanced_fallback_still_generic detected={generic_found}")
        # Nuclear option - context-specific template
        if primary_entity:
            return f"For {primary_entity} stuff, might be worth running it through KiloCode and seeing what it catches. Usually spots things pretty quick."
        else:
            return "Could be worth checking with KiloCode tbh. It's pretty good at spotting that kind of thing."
    
    logger.info(f"enhanced_fallback_generated length={len(result)} entity={primary_entity} entity_type={entity_type} discussion_type={discussion_type}")
    return result


# Legacy function name for backward compatibility
def _generate_emergency_fallback(post_title: str, post_content: str) -> str:
    """Legacy wrapper - use enhanced fallback."""
    key_points = _extract_key_points(post_title, post_content)
    context_snippets = get_relevant_context_snippets(post_content, post_title, max_snippets=2)
    context_ids = [s['id'] for s in context_snippets]
    return _generate_enhanced_fallback(post_title, post_content, key_points, context_ids)