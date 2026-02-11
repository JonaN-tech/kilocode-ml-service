# Gemini Comment Generation Regression Fix

## Summary

This document summarizes the fixes for the comment generation regression where comments became generic and repetitive across posts.

### Root Cause Analysis

The logs revealed:
```
[ML] ERROR gemini_generation_failed ... NotFound 404 models/gemini-2.0-flash-exp ... v1beta
[ML] INFO lightweight_comment_path title_len=6 content_len=4322
[ML] comment_generated ... docs_used=0 embeddings_used=false
```

**Problems identified:**
1. **Invalid model**: `gemini-2.0-flash-exp` returned 404 (not available on v1beta API)
2. **No retry with fallback model**: After Gemini failed, it went straight to lightweight fallback
3. **Generic fallback content**: Emergency fallback produced "many developers encounter..." boilerplate
4. **No context injection**: `docs_used=0` meant KiloCode context wasn't included for Reddit
5. **Title extraction bug**: `title_len=6` suggested broken title parsing
6. **No specificity guardrail**: No check to reject generic output and re-prompt

---

## Files Modified

### 1. `generation/gemini_generator.py`

**Changes:**
- **Model selection**: Changed from `gemini-2.0-flash-exp` to `gemini-2.0-flash` (stable)
- **Fallback chain**: Added `GEMINI_FALLBACK_MODELS = ["gemini-1.5-flash", "gemini-1.5-pro"]`
- **Error classification**: New function `classify_error()` distinguishes:
  - `config_error`: 404, invalid model, missing API key → switch model immediately
  - `transient_error`: timeout, rate limit, 5xx → exponential backoff retry
- **Specificity guardrail**: New `GENERIC_PHRASES` list and `_check_generic_phrases()` function
- **KiloCode context pack**: Static `KILOCODE_CONTEXT_PACK` with 8 relevant snippets
- **Context selection**: `get_relevant_context_snippets()` selects relevant docs based on keywords
- **Key point extraction**: `_extract_key_points()` extracts questions, tech terms, problems
- **Enhanced fallback**: `_generate_enhanced_fallback()` creates specific (not generic) comments
- **Improved prompts**: Stronger system and user prompts requiring specificity

### 2. `comment_engine.py`

**Changes:**
- **New Reddit path**: `generate_reddit_comment()` replaces lightweight path for Reddit
- **Context injection**: Now passes `doc_facts` from static context pack
- **Subreddit extraction**: `extract_subreddit()` helper for additional context
- **Better logging**: Logs `context_snippets` IDs showing which docs were used

### 3. `fetchers.py`

**Changes:**
- **Reddit-specific extraction**: New `extract_reddit_title()` with multiple strategies:
  1. JSON-LD structured data
  2. og:title meta tag
  3. Reddit-specific HTML elements
  4. URL fallback
- **Enhanced URL parsing**: Better handling of underscores, dashes, URL encoding
- **Logging**: Added logging for title extraction method used

### 4. `tests/test_gemini_regression_fix.py` (NEW)

**Tests cover:**
- Error classification (config vs transient)
- Model configuration (no experimental models)
- Generic phrase detection
- Key point extraction
- Context injection
- Title extraction
- Enhanced fallback quality
- Golden test with sample Reddit post
- Integration tests with mocked Gemini API

---

## Key Changes Summary

| Issue | Before | After |
|-------|--------|-------|
| Model | `gemini-2.0-flash-exp` (404) | `gemini-2.0-flash` + fallbacks |
| On 404 error | Immediate fallback to generic | Switch to fallback model |
| Context for Reddit | `docs_used=0` | 2-3 relevant context snippets |
| Fallback output | "many developers encounter..." | Topic-specific recommendations |
| Title extraction | Sometimes 6 chars | Enhanced Reddit-specific parsing |
| Specificity check | None | Guardrail rejects generic phrases |

---

## Manual Test Plan

### Test 1: Basic Reddit URL
```bash
curl -X POST http://localhost:8000/ml/generate-comment \
  -H "Content-Type: application/json" \
  -d '{"post_url": "https://www.reddit.com/r/reactjs/comments/example/how_to_debug_useeffect_infinite_loop"}'
```

**Verify:**
- [ ] No 404 errors in logs
- [ ] Comment length > 200 chars
- [ ] Comment mentions KiloCode
- [ ] Comment references specific post topic (React, useEffect, etc.)
- [ ] No generic phrases ("many developers encounter", etc.)
- [ ] Logs show `docs_used > 0` or `context_snippets=[...]`

### Test 2: Check Model Fallback
```bash
# Set invalid primary model to force fallback
GEMINI_GEN_MODEL=invalid-model python -c "from generation.gemini_generator import generate_comment_with_gemini; print(generate_comment_with_gemini('Test', 'Test content', [], []))"
```

**Verify:**
- [ ] Logs show `config_error_switching_model`
- [ ] Eventually generates a comment (using fallback model)

### Test 3: Verify Title Extraction
```bash
python -c "from fetchers import extract_title_from_url; print(extract_title_from_url('https://reddit.com/r/python/comments/abc/debugging_async_code_in_python'))"
```

**Verify:**
- [ ] Output: "Debugging Async Code In Python" (or similar)
- [ ] Length > 10 characters

### Test 4: Check Specificity Guardrail
```python
from generation.gemini_generator import _check_generic_phrases

# Should fail (contains generic phrase)
result = _check_generic_phrases("This is something many developers encounter.")
assert result[0] == False

# Should pass (specific)
result = _check_generic_phrases("The useEffect hook is causing issues because of missing dependencies.")
assert result[0] == True
```

### Test 5: Run Unit Tests
```bash
pytest tests/test_gemini_regression_fix.py -v
```

**Verify:**
- [ ] All tests pass
- [ ] No import errors

### Test 6: Full Integration Test
```bash
# Start the server
uvicorn app:app --reload

# In another terminal, test with a real Reddit URL
curl -X POST http://localhost:8000/ml/generate-comment \
  -H "Content-Type: application/json" \
  -d '{"post_url": "https://www.reddit.com/r/learnpython/comments/actual_post_id/actual_title_here"}'
```

**Check logs for:**
```
[ML] INFO gemini_generator_configured primary_model=gemini-2.0-flash fallbacks=['gemini-1.5-flash', 'gemini-1.5-pro']
[ML] INFO context_prepared docs_used_count=3 snippets=['debugging', 'analysis', 'core']
[ML] INFO gemini_generate_attempt model=gemini-2.0-flash attempt=1
[ML] INFO gemini_generated model=gemini-2.0-flash length=XXX sentences=X
[ML] INFO comment_quality_validated model=gemini-2.0-flash attempt=1 overlap=X
```

**NOT expected (regression indicators):**
```
[ML] ERROR gemini_generation_failed ... NotFound 404
[ML] INFO lightweight_comment_path  # For Reddit
docs_used=0
"many developers encounter"
```

---

## Rollback Plan

If issues arise, revert:
```bash
git checkout HEAD~1 -- generation/gemini_generator.py comment_engine.py fetchers.py
```

Or set environment variable to force legacy model:
```bash
export GEMINI_GEN_MODEL=gemini-1.5-flash
```

---

## Monitoring

After deployment, monitor for:
1. `gemini_generation_failed` with `config_error` → model needs updating
2. `specificity_guardrail_triggered` frequently → prompts need tuning
3. `comment_quality_failed` with `contains_generic_phrases` → model output degraded
4. `title_extraction_failed all_methods_exhausted` → fetch pattern changed

Add alerts for:
- Error rate > 10% on `/ml/generate-comment`
- Average comment length < 150 chars
- `docs_used=0` for Reddit posts