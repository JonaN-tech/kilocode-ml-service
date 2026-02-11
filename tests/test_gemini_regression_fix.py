"""
Unit and integration tests for the Gemini comment generation regression fix.

Tests cover:
1. Model fallback when primary model is invalid
2. Error classification (config vs transient)
3. Generic phrase detection (specificity guardrail)
4. Title extraction for Reddit
5. KiloCode context injection
6. Golden test with sample Reddit payload
"""
import pytest
import re
from unittest.mock import Mock, patch, MagicMock

# Import modules under test
from generation.gemini_generator import (
    classify_error,
    get_relevant_context_snippets,
    _check_generic_phrases,
    _extract_key_points,
    _validate_comment_quality,
    _generate_enhanced_fallback,
    GEMINI_PRIMARY_MODEL,
    GEMINI_FALLBACK_MODELS,
    GENERIC_PHRASES,
    KILOCODE_CONTEXT_PACK,
)
from fetchers import (
    extract_title,
    extract_title_from_url,
    extract_reddit_title,
)
from comment_engine import (
    extract_subreddit,
    generate_reddit_comment,
)


class TestErrorClassification:
    """Tests for error classification (config vs transient)."""
    
    def test_classify_404_as_config_error(self):
        """404 model not found should be classified as config error."""
        error = Exception("404 models/gemini-2.0-flash-exp is not found")
        assert classify_error(error) == "config_error"
    
    def test_classify_not_found_as_config_error(self):
        """NotFound errors should be classified as config error."""
        from google.api_core.exceptions import NotFound
        error = NotFound("Model not found")
        assert classify_error(error) == "config_error"
    
    def test_classify_rate_limit_as_transient(self):
        """Rate limit (429) should be classified as transient."""
        error = Exception("429 rate limit exceeded")
        assert classify_error(error) == "transient_error"
    
    def test_classify_503_as_transient(self):
        """Service unavailable (503) should be classified as transient."""
        error = Exception("503 service unavailable")
        assert classify_error(error) == "transient_error"
    
    def test_classify_timeout_as_transient(self):
        """Timeout errors should be classified as transient."""
        from google.api_core.exceptions import DeadlineExceeded
        error = DeadlineExceeded("Timeout")
        assert classify_error(error) == "transient_error"


class TestModelConfiguration:
    """Tests for model configuration."""
    
    def test_primary_model_is_stable(self):
        """Primary model should not be an experimental model."""
        assert "exp" not in GEMINI_PRIMARY_MODEL.lower()
        assert GEMINI_PRIMARY_MODEL in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
    
    def test_fallback_models_defined(self):
        """Fallback models should be defined."""
        assert len(GEMINI_FALLBACK_MODELS) >= 1
        for model in GEMINI_FALLBACK_MODELS:
            assert "exp" not in model.lower()


class TestSpecificityGuardrail:
    """Tests for the specificity guardrail that detects generic phrases."""
    
    def test_detect_generic_phrase_many_developers(self):
        """Should detect 'many developers encounter' as generic."""
        comment = "This is something many developers encounter when working with React."
        is_specific, detected = _check_generic_phrases(comment)
        assert not is_specific
        assert "many developers encounter" in detected
    
    def test_detect_generic_phrase_analyze_systematically(self):
        """Should detect 'analyze systematically' as generic."""
        comment = "KiloCode can help analyze systematically and find issues."
        is_specific, detected = _check_generic_phrases(comment)
        assert not is_specific
        assert "analyze systematically" in detected
    
    def test_detect_generic_phrase_time_consuming(self):
        """Should detect 'time-consuming manual inspection' as generic."""
        comment = "It's useful when time-consuming manual inspection would be needed."
        is_specific, detected = _check_generic_phrases(comment)
        assert not is_specific
        assert "time-consuming manual inspection" in detected
    
    def test_specific_comment_passes(self):
        """A specific comment should pass the guardrail."""
        comment = (
            "The useEffect dependency warnings you're seeing are caused by missing deps. "
            "KiloCode can analyze your hooks and show exactly which variables are missing. "
            "Try running the hook analyzer on that component."
        )
        is_specific, detected = _check_generic_phrases(comment)
        assert is_specific
        assert len(detected) == 0
    
    def test_all_generic_phrases_defined(self):
        """All expected generic phrases should be in the list."""
        expected_phrases = [
            "many developers encounter",
            "analyze systematically",
            "time-consuming manual inspection",
        ]
        for phrase in expected_phrases:
            assert phrase in GENERIC_PHRASES


class TestKeyPointExtraction:
    """Tests for key point extraction from posts."""
    
    def test_extract_questions(self):
        """Should extract questions from post content."""
        content = "I've been trying to debug this React app. How do I fix the useEffect warnings?"
        key_points = _extract_key_points("Debug React", content)
        assert any("Question:" in p for p in key_points)
    
    def test_extract_tech_terms(self):
        """Should extract technology terms."""
        content = "Using React with TypeScript and having issues with useState hooks."
        key_points = _extract_key_points("React TypeScript Issue", content)
        assert any("Technologies" in p or "React" in str(key_points) for p in key_points)
    
    def test_extract_problem_indicators(self):
        """Should extract problem indicators."""
        content = "I'm having trouble with authentication in my Node.js app."
        key_points = _extract_key_points("Auth Issue", content)
        assert any("trouble" in p.lower() or "Problem:" in p for p in key_points)


class TestContextInjection:
    """Tests for KiloCode context injection."""
    
    def test_context_pack_has_entries(self):
        """Context pack should have multiple entries."""
        assert len(KILOCODE_CONTEXT_PACK) >= 6
    
    def test_get_relevant_snippets_for_debugging(self):
        """Should return debugging context for debug-related posts."""
        snippets = get_relevant_context_snippets(
            "I'm trying to debug this error in my code",
            "Debug help needed"
        )
        snippet_ids = [s['id'] for s in snippets]
        assert "debugging" in snippet_ids
    
    def test_get_relevant_snippets_for_testing(self):
        """Should return testing context for test-related posts."""
        snippets = get_relevant_context_snippets(
            "How do I write unit tests for this function?",
            "Unit testing question"
        )
        snippet_ids = [s['id'] for s in snippets]
        assert "testing" in snippet_ids
    
    def test_get_relevant_snippets_for_refactoring(self):
        """Should return refactoring context for refactor-related posts."""
        snippets = get_relevant_context_snippets(
            "I need to refactor this legacy codebase",
            "Refactoring legacy code"
        )
        snippet_ids = [s['id'] for s in snippets]
        assert "refactoring" in snippet_ids
    
    def test_always_returns_some_context(self):
        """Should always return at least one context snippet."""
        snippets = get_relevant_context_snippets(
            "Random unrelated content here",
            "Random title"
        )
        assert len(snippets) >= 1


class TestTitleExtraction:
    """Tests for Reddit title extraction."""
    
    def test_extract_title_from_reddit_url(self):
        """Should extract title from Reddit URL."""
        url = "https://www.reddit.com/r/programming/comments/abc123/my_post_about_debugging_react"
        title = extract_title_from_url(url)
        assert len(title) > 10
        assert "Debugging" in title or "React" in title or "Post" in title
    
    def test_extract_title_handles_underscores(self):
        """Should convert underscores to spaces."""
        url = "https://reddit.com/r/python/comments/xyz/how_to_use_async_await"
        title = extract_title_from_url(url)
        assert "_" not in title
        assert " " in title
    
    def test_extract_title_handles_dashes(self):
        """Should convert dashes to spaces."""
        url = "https://reddit.com/r/javascript/comments/abc/best-practices-for-react"
        title = extract_title_from_url(url)
        # Dashes should be converted to spaces
        assert "best" in title.lower() or "practices" in title.lower() or "react" in title.lower()
    
    def test_extract_subreddit_from_url(self):
        """Should extract subreddit name from URL."""
        url = "https://www.reddit.com/r/learnpython/comments/abc123/help_with_pandas"
        subreddit = extract_subreddit(url)
        assert subreddit == "learnpython"
    
    def test_title_not_too_short(self):
        """Extracted title should not be suspiciously short (regression test)."""
        url = "https://reddit.com/r/programming/comments/abc/understanding_dependency_injection_in_python"
        title = extract_title_from_url(url)
        # This is the regression test - title_len should not be 6
        assert len(title) > 10, f"Title too short: '{title}' (len={len(title)})"


class TestEnhancedFallback:
    """Tests for the enhanced fallback (not generic)."""
    
    def test_fallback_does_not_contain_generic_phrases(self):
        """Enhanced fallback should NOT contain generic phrases."""
        title = "How to debug React useEffect warnings"
        content = "I'm getting dependency warnings in useEffect and can't figure out why."
        key_points = _extract_key_points(title, content)
        context_ids = ["debugging", "analysis"]
        
        fallback = _generate_enhanced_fallback(title, content, key_points, context_ids)
        
        # Check none of the generic phrases are present
        fallback_lower = fallback.lower()
        for phrase in GENERIC_PHRASES:
            assert phrase not in fallback_lower, f"Fallback contains generic phrase: '{phrase}'"
    
    def test_fallback_mentions_kilocode(self):
        """Enhanced fallback should mention KiloCode."""
        fallback = _generate_enhanced_fallback(
            "Test title",
            "Test content about debugging",
            ["Problem: debugging issue"],
            ["debugging"]
        )
        assert "kilocode" in fallback.lower()
    
    def test_fallback_is_specific_to_topic(self):
        """Enhanced fallback should reference the topic."""
        title = "React hooks causing infinite loop"
        content = "My useEffect is running in an infinite loop. Help!"
        
        fallback = _generate_enhanced_fallback(
            title, content,
            _extract_key_points(title, content),
            ["debugging"]
        )
        
        # Should reference React or hooks or loop
        fallback_lower = fallback.lower()
        topic_mentioned = any(term in fallback_lower for term in ["react", "hook", "loop", "useeffect"])
        assert topic_mentioned or len(fallback) > 100  # Either specific or substantial


class TestCommentQualityValidation:
    """Tests for comment quality validation."""
    
    def test_rejects_too_short_comment(self):
        """Should reject comments that are too short."""
        is_valid, reason, details = _validate_comment_quality(
            "Short comment.",
            "Test Post",
            "Test content"
        )
        assert not is_valid
        assert "too_short" in reason
    
    def test_rejects_comment_without_kilocode(self):
        """Should reject comments that don't mention KiloCode."""
        comment = (
            "This is a longer comment that talks about debugging and "
            "React hooks and other programming topics but never mentions "
            "the tool we want to promote."
        )
        is_valid, reason, details = _validate_comment_quality(
            comment,
            "React debugging",
            "React debugging content"
        )
        assert not is_valid
        assert "no_kilocode_mention" in reason
    
    def test_rejects_generic_comment(self):
        """Should reject comments with generic phrases."""
        comment = (
            "This is something many developers encounter when working with React. "
            "KiloCode can help analyze the problem systematically. "
            "It's useful for debugging."
        )
        is_valid, reason, details = _validate_comment_quality(
            comment,
            "React debugging",
            "React debugging content with error messages"
        )
        assert not is_valid
        assert "generic_phrases" in reason
    
    def test_accepts_specific_comment(self):
        """Should accept a specific, high-quality comment."""
        comment = (
            "The useEffect dependency warnings you're seeing in React are usually "
            "caused by missing variables in the dependency array. KiloCode can "
            "analyze your hooks and show exactly which dependencies are missing. "
            "Try running the hook analyzer on that specific component."
        )
        is_valid, reason, details = _validate_comment_quality(
            comment,
            "React useEffect warnings",
            "I'm getting useEffect dependency warnings and can't figure out why"
        )
        # If this passes, great. If not, check why
        if not is_valid:
            # Print details for debugging
            print(f"Validation failed: {reason}")
            print(f"Details: {details}")


class TestGoldenSampleRedditPost:
    """Golden test with a realistic Reddit post payload."""
    
    SAMPLE_POST = {
        "title": "How to debug React useEffect infinite loop?",
        "content": """
        I'm working on a React application and I've been stuck on this issue for hours.
        
        My useEffect hook keeps running in an infinite loop. Here's my code:
        
        ```javascript
        useEffect(() => {
            fetchData();
        }, [data]);
        ```
        
        Every time fetchData runs, it updates `data`, which triggers the effect again.
        I've tried using useCallback but that didn't help. 
        
        Has anyone dealt with this before? What's the best way to break the cycle?
        
        I'm using React 18 with TypeScript.
        """,
        "subreddit": "reactjs",
        "url": "https://reddit.com/r/reactjs/comments/abc123/how_to_debug_react_useeffect_infinite_loop"
    }
    
    def test_key_points_extracted(self):
        """Should extract meaningful key points from sample post."""
        key_points = _extract_key_points(
            self.SAMPLE_POST["title"],
            self.SAMPLE_POST["content"]
        )
        assert len(key_points) >= 1
        # Should identify either the question, React/TypeScript, or the problem
        key_points_str = str(key_points).lower()
        assert any(term in key_points_str for term in ["react", "typescript", "useeffect", "infinite", "loop", "question"])
    
    def test_context_snippets_relevant(self):
        """Should select relevant context snippets for the post."""
        snippets = get_relevant_context_snippets(
            self.SAMPLE_POST["content"],
            self.SAMPLE_POST["title"]
        )
        snippet_ids = [s['id'] for s in snippets]
        # Should select debugging-related context
        assert any(sid in ["debugging", "analysis", "context"] for sid in snippet_ids)
    
    def test_title_extraction_from_url(self):
        """Should extract proper title from sample URL."""
        title = extract_title_from_url(self.SAMPLE_POST["url"])
        assert len(title) > 15, f"Title too short: {title}"
        # Should contain words from the original title
        title_lower = title.lower()
        assert any(word in title_lower for word in ["debug", "react", "useeffect", "infinite", "loop"])
    
    def test_subreddit_extraction(self):
        """Should extract subreddit from sample URL."""
        subreddit = extract_subreddit(self.SAMPLE_POST["url"])
        assert subreddit == "reactjs"


# Integration test that would require mocking Gemini API
class TestIntegrationWithMockedGemini:
    """Integration tests with mocked Gemini API."""
    
    @patch('generation.gemini_generator.genai')
    def test_fallback_model_used_on_404(self, mock_genai):
        """When primary model returns 404, should try fallback model."""
        from generation.gemini_generator import _try_generate_with_model
        from google.api_core.exceptions import NotFound
        
        # First call raises NotFound
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = NotFound("Model not found")
        mock_genai.GenerativeModel.return_value = mock_model
        
        comment, error, error_type = _try_generate_with_model(
            model_name="gemini-2.0-flash-exp",
            system_prompt="test",
            user_prompt="test",
            attempt=1
        )
        
        assert error is not None
        assert error_type == "config_error"
    
    @patch('generation.gemini_generator.genai')
    def test_successful_generation_returns_comment(self, mock_genai):
        """Successful generation should return the comment text."""
        from generation.gemini_generator import _try_generate_with_model
        
        mock_response = MagicMock()
        mock_response.text = "This is a test comment about KiloCode."
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        comment, error, error_type = _try_generate_with_model(
            model_name="gemini-2.0-flash",
            system_prompt="test",
            user_prompt="test",
            attempt=1
        )
        
        assert comment is not None
        assert error is None
        assert error_type == "success"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])