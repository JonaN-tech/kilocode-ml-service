# Reddit Comment Context Relevance Improvements

## Overview

This document describes improvements made to the Reddit comment generation system to ensure comments stay highly relevant to the specific content of Reddit posts while maintaining a natural conversational tone.

## Problem Statement

Previously, generated comments could be too generic and didn't always reference specific entities mentioned in the original post. Comments like "Interesting discussion" or "Thanks for sharing" don't add value and don't respond to the post. Even when comments sounded natural, they sometimes lost contextual grounding - they could apply to any post, not specifically the one being replied to.

## Solution

### 1. Context Extraction Step (NEW)

Added a comprehensive [`extract_post_context()`](generation/gemini_generator.py:392) function that performs structured analysis of each post BEFORE generating a comment:

```python
context = extract_post_context(post_title, post_content)
# Returns:
# {
#   "entities": { models, tools, workflows, problems },
#   "main_topic": "Plan vs Build models in coding workflows",
#   "main_question": "What's the best model pair?",
#   "discussion_type": "workflow_planning",
#   "context_elements": ["model:Opus", "workflow:plan vs build", ...]
# }
```

This context object is passed through the entire generation pipeline and used for:
- Building the user prompt (tells Gemini exactly what to reference)
- Validating the output (checks that the comment references specific entities)
- Enhanced fallback generation (produces context-aware fallbacks)

### 2. Enhanced Entity Extraction

Updated [`_extract_specific_entities()`](generation/gemini_generator.py:293) with:

#### Versioned AI Models (NEW)
- **Pattern-based extraction**: Detects "Opus 4.6", "Kimi K2.5", "MiniMax M2.5", "GPT-4o", etc.
- **Display names preserved**: Stores original-case versions (e.g., "Opus 4.6" not "opus 4.6") for more natural prompt text
- **Deduplication**: Versioned models take priority over base name matching

#### Expanded Model List
- Added: o3, o3-mini, o4-mini, grok

#### Expanded Tool List
- Added: kilocode, windsurf, bolt, lovable, v0

#### Multi-word Workflow Phrases (NEW)
- Matches compound phrases FIRST: "plan vs build", "plan mode", "build mode", "multi-agent setup", "coding workflow", "model pairing", "vibe coding", "token usage", etc.
- Single keywords are only added if not already part of a matched phrase

#### AI Coding Workflow Keywords (NEW)
- Added: "plan vs build", "planning model", "implementation model", "multi-agent", "agentic", "coding workflow", "vibe coding", "prompt engineering", "model switching", "code generation", "context window", etc.

### 3. Discussion Type Classification (NEW)

Added [`_classify_discussion_type()`](generation/gemini_generator.py:447) that categorizes posts into:
- `comparison` - "Opus vs Sonnet", "X compared to Y"
- `workflow_planning` - "plan vs build", "planning model"
- `workflow` - general workflow discussions
- `model_comparison` - posts mentioning 2+ models
- `model_discussion` - posts about a specific model
- `help_request` - posts with errors/issues/problems
- `experience` - "I've been using...", "my experience with..."
- `recommendation` - "best...", "what do you recommend"
- `question` - general questions
- `general` - everything else

### 4. Stronger Context Grounding in Prompts

#### System Prompt ([`_build_system_prompt()`](generation/gemini_generator.py:737))
- **New top-level rule**: "A reader should be able to tell EXACTLY what post you're replying to just from reading your comment"
- Added examples matching task requirements (plan vs build, model pairs)
- Added more forbidden phrases: "That's a great question", "Great question", "Good point", "Thanks for starting this thread"
- Explicit instruction: "If a comment could apply to ANY random post, it will be REJECTED"

#### User Prompt ([`_build_user_prompt()`](generation/gemini_generator.py:828))
- **New CONTEXT ANALYSIS section**: Includes topic summary, discussion type, and main question
- **Entities section with display names**: Shows "Opus 4.6" not "opus 4.6"
- **Dynamic task instructions**: Reference specific entities by name in the instruction text
- **Stronger retry prompts**: Include the specific entities the comment MUST reference
- Added KiloCode capability description: "VS Code extension that lets you switch between different AI models mid-workflow"

### 5. Enhanced Quality Validation

Updated [`_validate_comment_quality()`](generation/gemini_generator.py:626) with:
- **Entity reference checking**: Validates that the comment actually mentions entities found in the post
- **Reject on zero entity references**: If the post mentions models/tools/workflows and the comment doesn't reference any, it's rejected
- **Adaptive overlap requirements**: If entity references are strong (2+), word overlap threshold is relaxed
- **Extended common words filter**: More words excluded from overlap counting for accuracy

### 6. Discussion-Aware Fallback Generator

Updated [`_generate_enhanced_fallback()`](generation/gemini_generator.py:1180) with:
- **Discussion type awareness**: Different templates for comparison, workflow_planning, model_comparison, etc.
- **Discussion-specific KiloCode recommendations**: e.g., for comparisons: "lets you switch between models mid-workflow"
- **Display names**: Uses original-case entity names
- **Secondary entity support**: References 2 models when a comparison is detected

### 7. Pipeline Integration

Updated [`comment_engine.py`](comment_engine.py:87) to:
- Import and call `extract_post_context()` before generation
- Log extracted context details (discussion type, models, workflows, topic)

## Quality Guardrails

### Forbidden Phrases (Never Allowed)
- "Interesting discussion"
- "Thanks for sharing"
- "Great post"
- "Nice thread"
- "Good topic"
- "That's a great question" (NEW)
- "Great question" (NEW)
- "Good point" (NEW)
- "Thanks for starting"

### Generic Phrases (Rejected by Quality Check)
- "many developers encounter"
- "comprehensive solution"
- "advanced capabilities"
- "powerful tool"
- "optimize workflow"
- "seamless integration"

### Entity Reference Requirement (NEW)
- If the post mentions models/tools/workflows, the comment MUST reference at least 1 by name
- Comments with zero entity references are rejected and retried

## Example Outputs

### Before (Generic)
> "This is something many developers encounter. KiloCode can help analyze the problem systematically."

### After (Context-Aware)
> "Yeah switching models between plan and build like that can sometimes double token usage. If you're worried about the build phase getting expensive, might be worth running it through KiloCode to see if it's generating overly verbose code."

### Before (Generic)
> "Great question."

### After (Context-Aware)
> "Opus tends to be really solid for planning. For the implementation step I've also seen good results with Kimi or MiniMax. Running the whole workflow through KiloCode can make that easier since you can switch models without changing editors."

## Testing

The test file [`test_context_relevance.py`](test_context_relevance.py) covers:

1. **Entity extraction** - models, versioned models, tools, workflow phrases, problems
2. **Post context extraction** - full `extract_post_context()` pipeline
3. **Discussion type classification** - comparison, workflow_planning, help_request, etc.
4. **Main question extraction** - finding the key question in a post
5. **Key points extraction** - with display names and workflow phrases
6. **Enhanced fallback** - context-aware fallback with discussion type
7. **No generic phrases** - guardrail validation
8. **No forbidden phrases** - strict prohibition validation
9. **Quality validation entity refs** - entity reference checking in validation
10. **Context elements building** - prioritized context element list

## Files Modified

1. [`generation/gemini_generator.py`](generation/gemini_generator.py) - Core changes: context extraction, entity detection, prompts, validation, fallback
2. [`comment_engine.py`](comment_engine.py) - Pipeline integration: imports `extract_post_context`, logs context
3. [`test_context_relevance.py`](test_context_relevance.py) - Expanded test suite

## New Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `extract_post_context()` | gemini_generator.py:392 | Comprehensive context extraction step |
| `_classify_discussion_type()` | gemini_generator.py:447 | Classify post discussion type |
| `_extract_main_question()` | gemini_generator.py:487 | Extract main question from post |
| `_summarize_topic()` | gemini_generator.py:506 | Create topic summary |
| `_build_context_elements()` | gemini_generator.py:528 | Build prioritized context elements list |

## New Constants

| Constant | Purpose |
|----------|---------|
| `VERSIONED_MODEL_PATTERNS` | Regex patterns for versioned model names |
| `WORKFLOW_PHRASES` | Multi-word workflow phrases |
| `WORKFLOW_KEYWORDS` (expanded) | AI coding workflow keywords |

## Benefits

1. **Contextual Grounding**: Comments now demonstrably respond to the specific post topic
2. **Entity References**: Comments reference specific models, tools, and workflows by name
3. **Discussion Awareness**: System understands whether a post is a comparison, help request, workflow discussion, etc.
4. **Versioned Model Support**: Correctly handles "Opus 4.6", "Kimi K2.5", "MiniMax M2.5"
5. **Workflow Phrase Matching**: Detects compound concepts like "plan vs build" and "multi-agent setup"
6. **No Generic Content**: Multiple guardrails prevent generic, low-value comments
7. **Stronger Validation**: Entity reference checking ensures the comment actually addresses the post
