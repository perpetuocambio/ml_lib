"""Integration tests for refactored PromptAnalyzer with Strategy Pattern.

Tests the full integration of PromptAnalyzer with all strategies working together.
"""

import pytest
from unittest.mock import Mock

from ml_lib.diffusion.domain.services.prompt_analyzer import PromptAnalyzer
from ml_lib.diffusion.domain.strategies import (
    RuleBasedConceptExtraction,
    RuleBasedIntentDetection,
    SimpleTokenization,
    StableDiffusionTokenization,
    AdvancedTokenization,
)
from ml_lib.diffusion.domain.interfaces.analysis_strategies import (
    ArtisticStyle,
    ContentType,
    QualityLevel,
)


class TestPromptAnalyzerBasicIntegration:
    """Test basic PromptAnalyzer functionality with default strategies."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer with LLM disabled."""
        return PromptAnalyzer(use_llm=False)

    def test_analyze_simple_prompt(self, analyzer):
        """Test analyzing a simple prompt."""
        result = analyzer.analyze("beautiful anime girl")

        assert result.original_prompt == "beautiful anime girl"
        assert len(result.tokens) > 0
        assert result.detected_concepts.concept_count > 0
        assert result.intent is not None
        assert 0.0 <= result.complexity_score <= 1.0
        assert result.emphasis_map.count > 0

    def test_analyze_complex_prompt(self, analyzer):
        """Test analyzing a complex prompt with multiple elements."""
        prompt = "masterpiece, best quality, 1girl, long hair, blue eyes, detailed face, photorealistic"
        result = analyzer.analyze(prompt)

        assert result.detected_concepts.concept_count >= 2
        assert result.intent.quality_level in [QualityLevel.HIGH, QualityLevel.MASTERPIECE]
        assert result.complexity_score > 0.3

    def test_analyze_with_emphasis(self, analyzer):
        """Test analyzing prompt with emphasis syntax."""
        prompt = "(beautiful girl), ((masterpiece)), [watermark]"
        result = analyzer.analyze(prompt)

        # Should detect emphasis
        assert result.emphasis_map.count >= 2
        # Check specific emphases
        assert result.emphasis_map.has_keyword("beautiful girl")
        assert result.emphasis_map.has_keyword("masterpiece")

    def test_analyze_detects_anime_style(self, analyzer):
        """Test that anime style is correctly detected."""
        result = analyzer.analyze("anime girl with blue hair")

        assert result.intent.artistic_style == ArtisticStyle.ANIME

    def test_analyze_detects_photorealistic_style(self, analyzer):
        """Test that photorealistic style is correctly detected."""
        result = analyzer.analyze("photorealistic portrait of a woman")

        assert result.intent.artistic_style == ArtisticStyle.PHOTOREALISTIC

    def test_analyze_detects_portrait_content(self, analyzer):
        """Test that portrait content is correctly detected."""
        result = analyzer.analyze("portrait of a woman, close-up face")

        assert result.intent.content_type == ContentType.PORTRAIT

    def test_analyze_detects_character_content(self, analyzer):
        """Test that character content is correctly detected."""
        result = analyzer.analyze("beautiful girl standing in forest")

        assert result.intent.content_type in [ContentType.CHARACTER, ContentType.SCENE]


class TestPromptAnalyzerOptimization:
    """Test prompt optimization with different model architectures."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer with LLM disabled."""
        return PromptAnalyzer(use_llm=False)

    def test_optimize_for_sdxl(self, analyzer):
        """Test SDXL optimization."""
        pos, neg = analyzer.optimize_for_model(
            prompt="beautiful girl",
            negative_prompt="blurry",
            base_model_architecture="SDXL",
            quality="high",
        )

        # SDXL appends quality tags
        assert "beautiful girl" in pos
        assert "masterpiece" in pos or "high quality" in pos
        assert "blurry" in neg
        assert "low quality" in neg

    def test_optimize_for_pony(self, analyzer):
        """Test Pony V6 optimization."""
        pos, neg = analyzer.optimize_for_model(
            prompt="cute girl",
            negative_prompt="",
            base_model_architecture="Pony V6",
            quality="balanced",
        )

        # Pony prepends score tags
        assert pos.startswith("score_")
        assert "cute girl" in pos
        # Pony-specific anatomical negatives
        assert "bad anatomy" in neg
        assert "score_4" in neg

    def test_optimize_for_sd15(self, analyzer):
        """Test SD 1.5 optimization."""
        pos, neg = analyzer.optimize_for_model(
            prompt="landscape scene",
            negative_prompt="ugly",
            base_model_architecture="SD 1.5",
            quality="ultra",
        )

        # SD 1.5 prepends quality tags
        assert "masterpiece" in pos
        assert "landscape scene" in pos
        assert "ugly" in neg
        assert "low quality" in neg

    def test_optimize_quality_levels(self, analyzer):
        """Test different quality levels produce different results."""
        prompt = "test prompt"

        fast_pos, _ = analyzer.optimize_for_model(
            prompt, "", "SDXL", quality="fast"
        )
        ultra_pos, _ = analyzer.optimize_for_model(
            prompt, "", "SDXL", quality="ultra"
        )

        # Ultra should have more quality tags
        assert len(ultra_pos) > len(fast_pos)
        assert "8k" in ultra_pos or "ultra" in ultra_pos

    def test_optimize_unknown_architecture_fallback(self, analyzer):
        """Test that unknown architecture falls back to SDXL."""
        pos, neg = analyzer.optimize_for_model(
            prompt="test",
            negative_prompt="",
            base_model_architecture="UnknownModel",
            quality="balanced",
        )

        # Should use SDXL defaults (appends quality)
        assert "test" in pos
        assert "quality" in pos.lower()


class TestPromptAnalyzerCustomStrategies:
    """Test PromptAnalyzer with custom strategy injection."""

    def test_custom_tokenization_strategy(self):
        """Test using custom tokenization strategy."""
        custom_tokenizer = SimpleTokenization()
        analyzer = PromptAnalyzer(
            use_llm=False,
            tokenization_strategy=custom_tokenizer,
        )

        result = analyzer.analyze("word1, word2, word3")

        # Simple tokenization just splits by commas
        assert "word1" in result.tokens
        assert "word2" in result.tokens
        assert "word3" in result.tokens

    def test_custom_concept_strategy(self):
        """Test using custom concept extraction strategy."""
        custom_extractor = RuleBasedConceptExtraction()
        analyzer = PromptAnalyzer(
            use_llm=False,
            concept_strategy=custom_extractor,
        )

        result = analyzer.analyze("beautiful anime girl")

        assert result.detected_concepts.concept_count > 0

    def test_custom_intent_strategy(self):
        """Test using custom intent detection strategy."""
        custom_detector = RuleBasedIntentDetection()
        analyzer = PromptAnalyzer(
            use_llm=False,
            intent_strategy=custom_detector,
        )

        result = analyzer.analyze("painting of landscape")

        assert result.intent.artistic_style == ArtisticStyle.PAINTING
        assert result.intent.content_type in [ContentType.SCENE, ContentType.CHARACTER]

    def test_advanced_tokenization_with_weights(self):
        """Test advanced tokenization with explicit weights."""
        advanced_tokenizer = AdvancedTokenization()
        analyzer = PromptAnalyzer(
            use_llm=False,
            tokenization_strategy=advanced_tokenizer,
        )

        result = analyzer.analyze("(beautiful:1.5), ((masterpiece)), test")

        # Should extract explicit weights
        emphasis = result.emphasis_map
        assert emphasis.has_keyword("beautiful")
        assert emphasis.has_keyword("masterpiece")


class TestPromptAnalyzerEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer with LLM disabled."""
        return PromptAnalyzer(use_llm=False)

    def test_empty_prompt(self, analyzer):
        """Test handling of empty prompt."""
        result = analyzer.analyze("")

        # Should handle gracefully
        assert result.original_prompt == ""
        assert result.detected_concepts.concept_count >= 1  # May have defaults

    def test_single_word_prompt(self, analyzer):
        """Test handling of single word prompt."""
        result = analyzer.analyze("test")

        assert result.original_prompt == "test"
        assert len(result.tokens) >= 1

    def test_very_long_prompt(self, analyzer):
        """Test handling of very long prompt."""
        # Create a 200-word prompt
        long_prompt = ", ".join([f"word{i}" for i in range(200)])
        result = analyzer.analyze(long_prompt)

        assert len(result.tokens) > 100
        # Complexity depends on semantic content, not just length
        # Generic words have lower complexity
        assert result.complexity_score > 0.2

    def test_prompt_with_special_characters(self, analyzer):
        """Test handling of special characters."""
        result = analyzer.analyze("(test), [word], {item}, word:1.5")

        # Should handle SD syntax
        assert result.emphasis_map.count >= 1

    def test_prompt_with_unicode(self, analyzer):
        """Test handling of unicode characters."""
        result = analyzer.analyze("美しい anime girl")

        # Should handle without crashing
        assert result.original_prompt == "美しい anime girl"


class TestPromptAnalyzerComplexityCalculation:
    """Test complexity score calculation."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer."""
        return PromptAnalyzer(use_llm=False)

    def test_simple_prompt_low_complexity(self, analyzer):
        """Test that simple prompts have low complexity."""
        result = analyzer.analyze("girl")

        assert result.complexity_score < 0.3

    def test_detailed_prompt_high_complexity(self, analyzer):
        """Test that detailed prompts have high complexity."""
        result = analyzer.analyze(
            "masterpiece, best quality, highly detailed, intricate, "
            "photorealistic, complex composition, elaborate background, "
            "1girl, long hair, detailed face, detailed eyes"
        )

        assert result.complexity_score > 0.5

    def test_complexity_increases_with_tokens(self, analyzer):
        """Test that complexity increases with token count."""
        simple = analyzer.analyze("girl")
        complex = analyzer.analyze("beautiful girl, long hair, blue eyes, detailed")

        assert complex.complexity_score > simple.complexity_score


class TestPromptAnalyzerStrategyIntegration:
    """Test integration between different strategies."""

    def test_tokenization_feeds_concept_extraction(self):
        """Test that tokenization output is used by concept extraction."""
        analyzer = PromptAnalyzer(use_llm=False)
        result = analyzer.analyze("anime girl, photorealistic, masterpiece")

        # Tokenization should split by comma
        # Concepts should find "anime", "photorealistic", "masterpiece"
        concepts = result.detected_concepts
        assert concepts.concept_count >= 2

    def test_concepts_feed_intent_detection(self):
        """Test that extracted concepts influence intent detection."""
        analyzer = PromptAnalyzer(use_llm=False)
        result = analyzer.analyze("anime portrait masterpiece")

        # Should detect anime style from concepts
        assert result.intent.artistic_style == ArtisticStyle.ANIME
        # Should detect portrait from concepts
        assert result.intent.content_type == ContentType.PORTRAIT
        # Should detect high quality from concepts
        assert result.intent.quality_level in [QualityLevel.HIGH, QualityLevel.MASTERPIECE]

    def test_all_strategies_work_together(self):
        """Test that all strategies work together correctly."""
        analyzer = PromptAnalyzer(use_llm=False)
        prompt = "(masterpiece:1.5), best quality, anime girl, portrait, detailed face"
        result = analyzer.analyze(prompt)

        # Tokenization
        assert len(result.tokens) > 0

        # Concept extraction
        assert result.detected_concepts.concept_count > 0

        # Intent detection
        assert result.intent.artistic_style == ArtisticStyle.ANIME
        assert result.intent.content_type in [ContentType.CHARACTER, ContentType.PORTRAIT]
        assert result.intent.quality_level in [QualityLevel.HIGH, QualityLevel.MASTERPIECE]

        # Emphasis extraction
        # Note: explicit weight syntax keeps the full token
        assert (result.emphasis_map.has_keyword("masterpiece:1.5") or
                result.emphasis_map.has_keyword("masterpiece"))

        # Complexity
        assert result.complexity_score > 0.3


class TestPromptAnalyzerBackwardCompatibility:
    """Test backward compatibility with old usage patterns."""

    def test_basic_usage_still_works(self):
        """Test that basic usage pattern still works."""
        analyzer = PromptAnalyzer(use_llm=False)
        result = analyzer.analyze("test prompt")

        assert result is not None
        assert hasattr(result, 'original_prompt')
        assert hasattr(result, 'tokens')
        assert hasattr(result, 'detected_concepts')
        assert hasattr(result, 'intent')
        assert hasattr(result, 'complexity_score')
        assert hasattr(result, 'emphasis_map')

    def test_optimize_for_model_still_works(self):
        """Test that optimize_for_model still works."""
        analyzer = PromptAnalyzer(use_llm=False)
        pos, neg = analyzer.optimize_for_model(
            "test",
            "blurry",
            "SDXL",
            "balanced"
        )

        assert isinstance(pos, str)
        assert isinstance(neg, str)
        assert len(pos) > 0
        assert len(neg) > 0

    def test_constructor_parameters_backward_compatible(self):
        """Test that old constructor parameters still work."""
        # Old usage pattern
        analyzer = PromptAnalyzer(
            ollama_url="http://localhost:11434",
            model_name="llama2",
            use_llm=False,
        )

        result = analyzer.analyze("test")
        assert result is not None
