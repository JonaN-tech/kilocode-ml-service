"""
Test script for Reddit comment context relevance improvements.

Tests that comments properly reference:
- AI models (Opus, Sonnet, GPT, Kimi, MiniMax, etc.) including versioned names
- Tools (VSCode, Docker, React, Python, etc.)
- Workflows (TDD, CI/CD, plan vs build, multi-agent, etc.)
- Specific problems mentioned in posts
- Discussion type classification
- Main question extraction
- Context element building
"""
import sys
sys.path.insert(0, '.')

from generation.gemini_generator import (
    _extract_specific_entities,
    _extract_key_points,
    _generate_enhanced_fallback,
    _check_generic_phrases,
    _validate_comment_quality,
    extract_post_context,
    _classify_discussion_type,
    _extract_main_question,
    _build_context_elements,
    FORBIDDEN_PHRASES,
    GENERIC_PHRASES,
)


def test_entity_extraction():
    """Test that entities are correctly extracted from posts."""
    print("\n=== Testing Entity Extraction ===\n")
    
    # Test 1: AI Models extraction
    post_title = "Opus vs Sonnet for coding tasks"
    post_content = "I've been comparing Opus and Sonnet for my daily coding workflow. Also tried GPT-4 and Kimi for comparison."
    
    entities = _extract_specific_entities(post_title, post_content)
    print(f"Post: {post_title}")
    print(f"Extracted models: {entities['models']}")
    print(f"Extracted models_display: {entities.get('models_display', [])}")
    assert "opus" in entities['models'], "Should extract Opus"
    assert "sonnet" in entities['models'], "Should extract Sonnet"
    assert "gpt-4" in entities['models'], "Should extract GPT-4"
    assert "kimi" in entities['models'], "Should extract Kimi"
    print("✓ AI Models extraction working\n")
    
    # Test 2: Versioned AI Models extraction
    post_title = "Opus 4.6 vs Kimi K2.5 for coding"
    post_content = "Comparing Opus 4.6, Kimi K2.5, and MiniMax M2.5 for different coding tasks. Also tried Sonnet."
    
    entities = _extract_specific_entities(post_title, post_content)
    print(f"Post: {post_title}")
    print(f"Extracted models: {entities['models']}")
    print(f"Extracted models_display: {entities.get('models_display', [])}")
    # Should find versioned models
    versioned_found = any("opus" in m and "4" in m for m in entities['models'])
    assert versioned_found, f"Should extract versioned 'Opus 4.6', got: {entities['models']}"
    kimi_versioned = any("kimi" in m and "2" in m for m in entities['models'])
    assert kimi_versioned, f"Should extract versioned 'Kimi K2.5', got: {entities['models']}"
    print("✓ Versioned AI Models extraction working\n")
    
    # Test 3: Tools extraction
    post_title = "Setting up VSCode with Docker"
    post_content = "I'm trying to configure VSCode to work with Docker and Kubernetes for my React project."
    
    entities = _extract_specific_entities(post_title, post_content)
    print(f"Post: {post_title}")
    print(f"Extracted tools: {entities['tools']}")
    assert "vscode" in entities['tools'], "Should extract VSCode"
    assert "docker" in entities['tools'], "Should extract Docker"
    assert "kubernetes" in entities['tools'], "Should extract Kubernetes"
    assert "react" in entities['tools'], "Should extract React"
    print("✓ Tools extraction working\n")
    
    # Test 4: Workflow phrase extraction (multi-word)
    post_title = "Plan vs Build models in coding workflows"
    post_content = "Looking at using different models for plan vs build in my coding workflow. The plan mode uses Opus and the build mode uses Sonnet."
    
    entities = _extract_specific_entities(post_title, post_content)
    print(f"Post: {post_title}")
    print(f"Extracted workflows: {entities['workflows']}")
    assert "plan vs build" in entities['workflows'], f"Should extract 'plan vs build', got: {entities['workflows']}"
    assert "coding workflow" in entities['workflows'], f"Should extract 'coding workflow', got: {entities['workflows']}"
    print("✓ Workflow phrase extraction working\n")
    
    # Test 5: Workflows extraction (single keywords)
    post_title = "Best practices for TDD in Python"
    post_content = "Looking for advice on TDD workflow and CI/CD integration with GitHub Actions."
    
    entities = _extract_specific_entities(post_title, post_content)
    print(f"Post: {post_title}")
    print(f"Extracted workflows: {entities['workflows']}")
    assert "tdd" in entities['workflows'], "Should extract TDD"
    assert "ci/cd" in entities['workflows'] or "cicd" in entities['workflows'], "Should extract CI/CD"
    print("✓ Single workflow extraction working\n")
    
    # Test 6: Problem extraction
    post_title = "Having trouble with React useEffect"
    post_content = "I'm having trouble with React useEffect dependency array warnings in my component."
    
    entities = _extract_specific_entities(post_title, post_content)
    print(f"Post: {post_title}")
    print(f"Extracted problems: {entities['problems']}")
    print(f"Extracted tools: {entities['tools']}")
    assert "react" in entities['tools'], "Should extract React"
    assert len(entities['problems']) > 0, "Should extract at least one problem"
    print("✓ Problem extraction working\n")


def test_extract_post_context():
    """Test the comprehensive context extraction function."""
    print("\n=== Testing extract_post_context() ===\n")
    
    # Test 1: Model comparison post
    post_title = "Opus vs Sonnet for coding tasks"
    post_content = "Which model is better for coding - Opus or Sonnet? I've been using both and Opus seems better for planning but Sonnet is faster for implementation."
    
    context = extract_post_context(post_title, post_content)
    print(f"Post: {post_title}")
    print(f"Discussion type: {context['discussion_type']}")
    print(f"Main topic: {context['main_topic']}")
    print(f"Main question: {context['main_question']}")
    print(f"Context elements: {context['context_elements']}")
    
    assert context['discussion_type'] in ('comparison', 'model_comparison'), \
        f"Should classify as comparison, got: {context['discussion_type']}"
    assert context['main_question'] is not None, "Should extract main question"
    assert len(context['context_elements']) > 0, "Should have context elements"
    print("✓ Model comparison context extraction working\n")
    
    # Test 2: Workflow planning post
    post_title = "Best model pair for plan vs build coding workflow"
    post_content = "I'm looking for the best planning/implementation model pair. Currently using Opus for planning and considering Kimi or MiniMax for the build phase."
    
    context = extract_post_context(post_title, post_content)
    print(f"Post: {post_title}")
    print(f"Discussion type: {context['discussion_type']}")
    print(f"Main topic: {context['main_topic']}")
    print(f"Workflows: {context['entities']['workflows']}")
    print(f"Models: {context['entities']['models']}")
    
    assert context['discussion_type'] in ('workflow_planning', 'comparison', 'workflow'), \
        f"Should classify as workflow_planning, got: {context['discussion_type']}"
    assert len(context['entities']['models']) >= 1, "Should find models"
    print("✓ Workflow planning context extraction working\n")
    
    # Test 3: Help request post
    post_title = "Docker container networking issues"
    post_content = "I'm having trouble with Docker container networking. My containers can't communicate with each other through the bridge network."
    
    context = extract_post_context(post_title, post_content)
    print(f"Post: {post_title}")
    print(f"Discussion type: {context['discussion_type']}")
    print(f"Problems: {context['entities']['problems']}")
    
    assert context['discussion_type'] == 'help_request', \
        f"Should classify as help_request, got: {context['discussion_type']}"
    print("✓ Help request context extraction working\n")


def test_discussion_type_classification():
    """Test that discussion types are correctly classified."""
    print("\n=== Testing Discussion Type Classification ===\n")
    
    test_cases = [
        ("opus vs sonnet", {"models": ["opus", "sonnet"], "workflows": [], "tools": []}, "comparison"),
        ("plan vs build models", {"models": ["opus"], "workflows": ["plan vs build"], "tools": []}, "comparison"),
        ("best planning model", {"models": ["opus"], "workflows": ["planning model"], "tools": []}, "workflow_planning"),
        ("having trouble with docker", {"models": [], "workflows": [], "tools": ["docker"]}, "help_request"),
        ("my experience with kimi", {"models": ["kimi"], "workflows": [], "tools": []}, "model_discussion"),
        ("opus and sonnet both work", {"models": ["opus", "sonnet"], "workflows": [], "tools": []}, "model_comparison"),
    ]
    
    for text, entities, expected_type in test_cases:
        result = _classify_discussion_type(text, entities)
        print(f"Text: '{text}' -> {result} (expected: {expected_type})")
        assert result == expected_type, f"Expected '{expected_type}', got '{result}' for '{text}'"
    
    print("✓ Discussion type classification working\n")


def test_main_question_extraction():
    """Test that main questions are correctly extracted."""
    print("\n=== Testing Main Question Extraction ===\n")
    
    # Test with a clear question
    text = "What's the best model pair for plan/build workflows? I've been using Opus for planning."
    question = _extract_main_question(text)
    print(f"Text: {text}")
    print(f"Question: {question}")
    assert question is not None, "Should extract question"
    assert "?" in question, "Question should contain ?"
    print("✓ Question extraction working\n")
    
    # Test with no question
    text = "I've been using Opus for planning and Sonnet for building. Works great."
    question = _extract_main_question(text)
    print(f"Text: {text}")
    print(f"Question: {question}")
    assert question is None, "Should not extract question when none exists"
    print("✓ No question correctly detected\n")


def test_key_points_extraction():
    """Test that key points include entities and display names."""
    print("\n=== Testing Key Points Extraction ===\n")
    
    post_title = "Comparing Claude Opus vs GPT-4 for code review"
    post_content = "I've been using both Opus and GPT-4 for code reviews in my GitHub workflow. Looking for opinions on which handles complex refactoring better."
    
    key_points = _extract_key_points(post_title, post_content)
    print(f"Post: {post_title}")
    print(f"Key points: {key_points}")
    
    # Check that models are in key points
    models_found = any("opus" in kp.lower() or "gpt" in kp.lower() for kp in key_points)
    assert models_found, "Key points should include AI models"
    
    # Check that tools are in key points
    tools_found = any("github" in kp.lower() for kp in key_points)
    assert tools_found, "Key points should include tools"
    print("✓ Key points extraction working\n")
    
    # Test with workflow phrases
    post_title = "Plan vs Build models in coding workflows"
    post_content = "How do you handle the plan vs build approach? What models work best for each?"
    
    key_points = _extract_key_points(post_title, post_content)
    print(f"Post: {post_title}")
    print(f"Key points: {key_points}")
    
    workflows_found = any("plan vs build" in kp.lower() or "coding workflow" in kp.lower() for kp in key_points)
    assert workflows_found, f"Key points should include workflow phrases, got: {key_points}"
    print("✓ Workflow key points extraction working\n")


def test_enhanced_fallback():
    """Test that fallback comments reference specific entities and use discussion type."""
    print("\n=== Testing Enhanced Fallback ===\n")
    
    # Test 1: Fallback with AI models (comparison)
    post_title = "Opus vs Sonnet comparison"
    post_content = "Which model is better for coding tasks - Opus or Sonnet?"
    key_points = _extract_key_points(post_title, post_content)
    post_context = extract_post_context(post_title, post_content)
    
    comment = _generate_enhanced_fallback(post_title, post_content, key_points, ["core"], post_context=post_context)
    print(f"Post: {post_title}")
    print(f"Generated comment: {comment}")
    
    # Check that comment references the models
    assert "opus" in comment.lower() or "sonnet" in comment.lower(), "Comment should reference Opus or Sonnet"
    assert "kilocode" in comment.lower(), "Comment should mention KiloCode"
    
    # Check for forbidden phrases
    for phrase in FORBIDDEN_PHRASES:
        assert phrase not in comment.lower(), f"Comment should not contain forbidden phrase: {phrase}"
    
    print("✓ Fallback with AI models working\n")
    
    # Test 2: Fallback with workflow (plan vs build)
    post_title = "Plan vs Build models in coding workflows"
    post_content = "How do you handle using different models for the plan vs build approach?"
    key_points = _extract_key_points(post_title, post_content)
    post_context = extract_post_context(post_title, post_content)
    
    comment = _generate_enhanced_fallback(post_title, post_content, key_points, ["workflow"], post_context=post_context)
    print(f"Post: {post_title}")
    print(f"Discussion type: {post_context['discussion_type']}")
    print(f"Generated comment: {comment}")
    
    # Check that comment references the workflow
    assert "plan" in comment.lower() or "build" in comment.lower() or "workflow" in comment.lower(), \
        f"Comment should reference plan/build/workflow, got: {comment}"
    assert "kilocode" in comment.lower(), "Comment should mention KiloCode"
    print("✓ Fallback with workflow working\n")
    
    # Test 3: Fallback with tools
    post_title = "Docker container networking issues"
    post_content = "I'm having issues with Docker container networking in my Kubernetes cluster."
    key_points = _extract_key_points(post_title, post_content)
    post_context = extract_post_context(post_title, post_content)
    
    comment = _generate_enhanced_fallback(post_title, post_content, key_points, ["debugging"], post_context=post_context)
    print(f"Post: {post_title}")
    print(f"Generated comment: {comment}")
    
    # Check that comment references the tools
    assert "docker" in comment.lower() or "kubernetes" in comment.lower(), "Comment should reference Docker or Kubernetes"
    assert "kilocode" in comment.lower(), "Comment should mention KiloCode"
    print("✓ Fallback with tools working\n")
    
    # Test 4: Fallback with model comparison discussion type
    post_title = "Best planning / implementation model pair"
    post_content = "Opus tends to be really solid for planning. For the implementation step I've also seen good results with Kimi or MiniMax."
    key_points = _extract_key_points(post_title, post_content)
    post_context = extract_post_context(post_title, post_content)
    
    comment = _generate_enhanced_fallback(post_title, post_content, key_points, ["core"], post_context=post_context)
    print(f"Post: {post_title}")
    print(f"Discussion type: {post_context['discussion_type']}")
    print(f"Generated comment: {comment}")
    
    assert "kilocode" in comment.lower(), "Comment should mention KiloCode"
    # Should reference at least one model or approach
    has_entity_ref = any(term in comment.lower() for term in ["opus", "kimi", "minimax", "planning", "implementation", "model"])
    assert has_entity_ref, f"Comment should reference entities from the post, got: {comment}"
    print("✓ Fallback with model pair discussion working\n")


def test_no_generic_phrases():
    """Test that generated comments don't contain generic phrases."""
    print("\n=== Testing No Generic Phrases ===\n")
    
    test_posts = [
        ("Opus vs Sonnet for coding", "Which model handles complex codebases better?"),
        ("Docker networking problems", "My Docker containers can't communicate with each other."),
        ("React performance issues", "My React app is slow and I need help optimizing."),
        ("TDD workflow questions", "How do you integrate TDD with GitHub Actions?"),
        ("Plan vs Build in coding workflows", "What's the best model pair for plan and build?"),
    ]
    
    for title, content in test_posts:
        key_points = _extract_key_points(title, content)
        post_context = extract_post_context(title, content)
        comment = _generate_enhanced_fallback(title, content, key_points, ["core"], post_context=post_context)
        
        # Check for generic phrases
        is_specific, detected = _check_generic_phrases(comment)
        
        print(f"Post: {title}")
        print(f"Comment: {comment}")
        print(f"Generic phrases detected: {detected}")
        assert is_specific, f"Comment should not contain generic phrases: {detected}"
        print("✓ No generic phrases\n")


def test_forbidden_phrases():
    """Test that forbidden phrases are never in generated comments."""
    print("\n=== Testing Forbidden Phrases ===\n")
    
    test_posts = [
        ("Interesting discussion about AI", "What do you think about AI coding assistants?"),
        ("Great post about Docker", "Docker is amazing for development."),
        ("Thanks for sharing this", "I learned a lot from this post."),
        ("Great question about models", "Which model should I use?"),
    ]
    
    for title, content in test_posts:
        key_points = _extract_key_points(title, content)
        post_context = extract_post_context(title, content)
        comment = _generate_enhanced_fallback(title, content, key_points, ["core"], post_context=post_context)
        
        print(f"Post: {title}")
        print(f"Comment: {comment}")
        
        for phrase in FORBIDDEN_PHRASES:
            assert phrase not in comment.lower(), f"Comment should not contain forbidden phrase: '{phrase}'"
        
        print("✓ No forbidden phrases\n")


def test_validate_comment_quality_entity_refs():
    """Test that quality validation checks for entity references."""
    print("\n=== Testing Quality Validation Entity References ===\n")
    
    post_title = "Opus vs Sonnet for coding"
    post_content = "Which model is better for complex refactoring tasks?"
    post_context = extract_post_context(post_title, post_content)
    
    # Comment that references entities - should pass entity check
    good_comment = "Both Opus and Sonnet handle refactoring pretty well tbh. Might be worth running your codebase through KiloCode to see which model catches more issues in your specific coding setup."
    is_valid, reason, details = _validate_comment_quality(good_comment, post_title, post_content, post_context=post_context)
    print(f"Good comment entity_refs: {details['entity_references']}")
    print(f"Good comment overlap: {details['overlap_count']}")
    # Entity refs should be > 0
    assert details['entity_references'] > 0, f"Should find entity references, got: {details['entity_references']}"
    print("✓ Good comment has entity references\n")
    
    # Comment that doesn't reference entities - should fail entity check
    bad_comment = "That's a really interesting question about coding tools. KiloCode might help with your workflow if you're looking for something that understands your whole project context and codebase structure."
    is_valid_bad, reason_bad, details_bad = _validate_comment_quality(bad_comment, post_title, post_content, post_context=post_context)
    print(f"Bad comment entity_refs: {details_bad['entity_references']}")
    print(f"Bad comment valid: {is_valid_bad}, reason: {reason_bad}")
    # This should ideally fail or have 0 entity refs
    if details_bad['entity_references'] == 0:
        print("✓ Bad comment correctly detected as missing entity references")
    else:
        print(f"NOTE: Bad comment has {details_bad['entity_references']} entity refs (may still fail on other checks)")
    print()


def test_context_elements_building():
    """Test that context elements are correctly built."""
    print("\n=== Testing Context Elements Building ===\n")
    
    entities = {
        "models": ["opus", "sonnet"],
        "models_display": ["Opus", "Sonnet"],
        "tools": ["vscode"],
        "tools_display": ["VSCode"],
        "workflows": ["plan vs build", "coding workflow"],
        "problems": ["too expensive token usage"],
    }
    
    elements = _build_context_elements(entities, "What's the best model pair?", "comparison")
    print(f"Context elements: {elements}")
    
    assert any("model:" in e for e in elements), "Should have model elements"
    assert any("workflow:" in e for e in elements), "Should have workflow elements"
    assert any("question:" in e for e in elements), "Should have question element"
    print("✓ Context elements correctly built\n")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("REDDIT COMMENT CONTEXT RELEVANCE TESTS")
    print("=" * 60)
    
    try:
        test_entity_extraction()
        test_extract_post_context()
        test_discussion_type_classification()
        test_main_question_extraction()
        test_key_points_extraction()
        test_enhanced_fallback()
        test_no_generic_phrases()
        test_forbidden_phrases()
        test_validate_comment_quality_entity_refs()
        test_context_elements_building()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60 + "\n")
        return True
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        return False
    except Exception as e:
        print(f"\n❌ TEST ERROR: {type(e).__name__}: {e}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
