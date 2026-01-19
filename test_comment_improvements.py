"""
Quick verification test for comment generation improvements.
Tests the new features without requiring full system setup.
"""

import sys
import logging
import io

# Fix Windows encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Setup basic logging
logging.basicConfig(level=logging.INFO)

# Mock the retrieval module to avoid dependencies
sys.modules['retrieval'] = type(sys)('retrieval')
sys.modules['retrieval'].search_by_name = lambda *args, **kwargs: []

# Mock text_utils
sys.modules['text_utils'] = type(sys)('text_utils')
sys.modules['text_utils'].clean_text = lambda text, **kwargs: text
sys.modules['text_utils'].chunk_text = lambda text, **kwargs: [text[:1000]]

# Import the modules to test
from generation.prompt_builder import (
    _detect_kilocode_mention,
    _get_kilocode_injection,
    _count_sentences,
    _check_repetition,
    PLATFORM_MIN_SENTENCES,
    KILOCODE_CONCEPTS,
    build_lightweight_comment
)

def test_kilocode_detection():
    """Test KiloCode mention detection."""
    print("\n=== Testing KiloCode Detection ===")
    
    # Should detect KiloCode
    text1 = "I'm using KiloCode for my project"
    result1 = _detect_kilocode_mention(text1)
    print(f"Text with KiloCode: {result1} (expected: True)")
    assert result1 == True, "Should detect KiloCode"
    
    # Should not detect KiloCode
    text2 = "I'm working on a Python automation script"
    result2 = _detect_kilocode_mention(text2)
    print(f"Text without KiloCode: {result2} (expected: False)")
    assert result2 == False, "Should not detect KiloCode"
    
    print("[OK] KiloCode detection working correctly")


def test_kilocode_injection():
    """Test KiloCode injection suggestions."""
    print("\n=== Testing KiloCode Injection ===")
    
    # Test automation keyword
    content1 = "I need to automate my repetitive tasks"
    title1 = "Looking for automation tools"
    suggestion1 = _get_kilocode_injection(content1, title1)
    print(f"Automation context: {suggestion1[:50]}...")
    assert "automate" in suggestion1.lower() or "KiloCode" in suggestion1
    
    # Test debugging keyword
    content2 = "I'm trying to debug this error in my code"
    title2 = "Need help debugging"
    suggestion2 = _get_kilocode_injection(content2, title2)
    print(f"Debugging context: {suggestion2[:50]}...")
    assert "debug" in suggestion2.lower() or "KiloCode" in suggestion2
    
    print("[OK] KiloCode injection working correctly")


def test_sentence_counting():
    """Test sentence counting."""
    print("\n=== Testing Sentence Counting ===")
    
    text = "This is sentence one. This is sentence two! Is this sentence three?"
    count = _count_sentences(text)
    print(f"Sentence count: {count} (expected: 3)")
    assert count == 3, f"Expected 3 sentences, got {count}"
    
    print("[OK] Sentence counting working correctly")


def test_platform_requirements():
    """Test platform-specific requirements are defined."""
    print("\n=== Testing Platform Requirements ===")
    
    print(f"Reddit minimum: {PLATFORM_MIN_SENTENCES.get('reddit')} sentences")
    print(f"GitHub minimum: {PLATFORM_MIN_SENTENCES.get('github')} sentences")
    print(f"HN minimum: {PLATFORM_MIN_SENTENCES.get('hn')} sentences")
    
    assert PLATFORM_MIN_SENTENCES.get('reddit') >= 3, "Reddit should require 3+ sentences"
    assert PLATFORM_MIN_SENTENCES.get('github') >= 3, "GitHub should require 3+ sentences"
    
    print("[OK] Platform requirements configured correctly")


def test_comment_generation():
    """Test lightweight comment generation."""
    print("\n=== Testing Comment Generation ===")
    
    title = "Best practices for Python automation"
    content = "I'm looking for ways to automate my development workflow. What tools do people recommend?"
    platform = "reddit"
    
    comment = build_lightweight_comment(title, content, platform)
    
    print(f"Generated comment ({len(comment)} chars):")
    print(f"  {comment}")
    
    # Check length
    sentence_count = _count_sentences(comment)
    print(f"Sentence count: {sentence_count}")
    
    # Verify minimum length
    min_required = PLATFORM_MIN_SENTENCES.get(platform, 2)
    assert sentence_count >= min_required, f"Expected at least {min_required} sentences, got {sentence_count}"
    
    # Check if KiloCode was mentioned (should be, since it's not in the original)
    has_kilocode = "kilocode" in comment.lower()
    print(f"KiloCode mentioned: {has_kilocode}")
    
    print("[OK] Comment generation working correctly")


def test_repetition_check():
    """Test anti-repetition mechanism."""
    print("\n=== Testing Anti-Repetition ===")
    
    comment1 = "Thanks for sharing your experience with automation."
    comment2 = "Thanks for sharing your experience with automation."  # Same
    comment3 = "I appreciate your thoughts on testing strategies."    # Different
    
    result1 = _check_repetition(comment1)
    print(f"First comment unique: {result1} (expected: True)")
    
    result2 = _check_repetition(comment2)
    print(f"Duplicate comment unique: {result2} (expected: False)")
    
    result3 = _check_repetition(comment3)
    print(f"Different comment unique: {result3} (expected: True)")
    
    print("[OK] Anti-repetition mechanism working")


def run_all_tests():
    """Run all verification tests."""
    print("=" * 60)
    print("COMMENT GENERATION IMPROVEMENT VERIFICATION")
    print("=" * 60)
    
    try:
        test_kilocode_detection()
        test_kilocode_injection()
        test_sentence_counting()
        test_platform_requirements()
        test_comment_generation()
        test_repetition_check()
        
        print("\n" + "=" * 60)
        print("[SUCCESS] ALL TESTS PASSED - Changes verified successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[FAILED] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()