"""Tests for Tokenization Strategies - Parse prompts with SD syntax.

Tests the Strategy pattern implementation for tokenization.
"""

import pytest

from ml_lib.diffusion.domain.strategies.tokenization import (
    SimpleTokenization,
    StableDiffusionTokenization,
    AdvancedTokenization,
)


class TestSimpleTokenization:
    """Tests for simple tokenization strategy."""

    @pytest.fixture
    def strategy(self):
        """Create simple tokenization strategy."""
        return SimpleTokenization()

    def test_tokenize_simple_prompt(self, strategy):
        """Test tokenizing a simple comma-separated prompt."""
        tokens = strategy.tokenize("beautiful girl, long hair, blue eyes")
        assert tokens == ["beautiful girl", "long hair", "blue eyes"]

    def test_tokenize_single_token(self, strategy):
        """Test tokenizing prompt with single token."""
        tokens = strategy.tokenize("beautiful landscape")
        assert tokens == ["beautiful landscape"]

    def test_tokenize_strips_whitespace(self, strategy):
        """Test that tokenization strips whitespace."""
        tokens = strategy.tokenize("  girl  ,  hair  ,  eyes  ")
        assert tokens == ["girl", "hair", "eyes"]

    def test_tokenize_filters_empty(self, strategy):
        """Test that empty tokens are filtered."""
        tokens = strategy.tokenize("girl,,,hair,,eyes")
        assert tokens == ["girl", "hair", "eyes"]

    def test_tokenize_empty_prompt(self, strategy):
        """Test tokenizing empty prompt."""
        tokens = strategy.tokenize("")
        assert tokens == []

    def test_extract_emphasis_returns_empty(self, strategy):
        """Test that simple tokenization doesn't extract emphasis."""
        emphasis = strategy.extract_emphasis_map("(emphasized), normal")
        assert emphasis == {}


class TestStableDiffusionTokenization:
    """Tests for Stable Diffusion tokenization strategy."""

    @pytest.fixture
    def strategy(self):
        """Create SD tokenization strategy."""
        return StableDiffusionTokenization()

    # Basic Tokenization Tests
    def test_tokenize_simple_prompt(self, strategy):
        """Test tokenizing simple SD prompt."""
        tokens = strategy.tokenize("1girl, long hair, blue eyes")
        assert tokens == ["1girl", "long hair", "blue eyes"]

    def test_tokenize_preserves_emphasis_syntax(self, strategy):
        """Test that emphasis syntax is preserved in tokens."""
        tokens = strategy.tokenize("(beautiful girl), [low quality]")
        assert "(beautiful girl)" in tokens
        assert "[low quality]" in tokens

    def test_tokenize_normalizes_whitespace(self, strategy):
        """Test that extra whitespace is normalized."""
        tokens = strategy.tokenize("girl  ,   hair   ,   eyes")
        assert tokens == ["girl", "hair", "eyes"]

    def test_tokenize_handles_and_blending(self, strategy):
        """Test handling of AND keyword for blending."""
        tokens = strategy.tokenize("cat AND dog, playing")
        assert "cat" in tokens
        assert "dog" in tokens
        assert "playing" in tokens
        assert len(tokens) == 3

    def test_tokenize_and_case_insensitive(self, strategy):
        """Test that AND is case insensitive."""
        tokens = strategy.tokenize("cat and dog")
        assert "cat" in tokens
        assert "dog" in tokens

    def test_tokenize_filters_empty(self, strategy):
        """Test that empty tokens are filtered."""
        tokens = strategy.tokenize("girl,,,hair")
        assert tokens == ["girl", "hair"]

    # Emphasis Extraction Tests
    def test_extract_single_emphasis(self, strategy):
        """Test extracting single level emphasis."""
        emphasis = strategy.extract_emphasis_map("(beautiful) girl")
        assert emphasis["beautiful"] == 1.1

    def test_extract_double_emphasis(self, strategy):
        """Test extracting double level emphasis."""
        emphasis = strategy.extract_emphasis_map("((masterpiece)) quality")
        assert emphasis["masterpiece"] == 1.2

    def test_extract_triple_emphasis(self, strategy):
        """Test extracting triple level emphasis."""
        emphasis = strategy.extract_emphasis_map("(((ultra detailed))) image")
        assert emphasis["ultra detailed"] == pytest.approx(1.3, rel=0.01)

    def test_extract_single_deemphasis(self, strategy):
        """Test extracting single level de-emphasis."""
        emphasis = strategy.extract_emphasis_map("[watermark] removed")
        assert emphasis["watermark"] == 0.9

    def test_extract_double_deemphasis(self, strategy):
        """Test extracting double level de-emphasis."""
        emphasis = strategy.extract_emphasis_map("[[blur]] reduced")
        assert emphasis["blur"] == pytest.approx(0.8, rel=0.01)

    def test_extract_mixed_emphasis(self, strategy):
        """Test extracting mixed emphasis and de-emphasis."""
        emphasis = strategy.extract_emphasis_map("(beautiful) girl, [blur], ((detailed))")

        assert emphasis["beautiful"] == 1.1
        assert emphasis["blur"] == 0.9
        assert emphasis["detailed"] == 1.2

    def test_extract_emphasis_strips_whitespace(self, strategy):
        """Test that emphasis extraction strips whitespace."""
        emphasis = strategy.extract_emphasis_map("(  beautiful  )")
        assert "beautiful" in emphasis

    def test_extract_emphasis_minimum_weight(self, strategy):
        """Test that de-emphasis doesn't go below 0.1."""
        # Deep nesting [[[[[[word]]]]]] would be 1.0 - 6*0.1 = 0.4
        emphasis = strategy.extract_emphasis_map("[[[[[[word]]]]]]")
        assert emphasis["word"] >= 0.1

    def test_extract_emphasis_ignores_unmatched(self, strategy):
        """Test that unmatched brackets are ignored."""
        emphasis = strategy.extract_emphasis_map("(word, normal text")
        # Should not extract emphasis for malformed syntax
        assert len(emphasis) == 0 or "word, normal text" not in emphasis


class TestAdvancedTokenization:
    """Tests for advanced tokenization strategy."""

    @pytest.fixture
    def strategy(self):
        """Create advanced tokenization strategy."""
        return AdvancedTokenization()

    # Basic Tokenization Tests
    def test_tokenize_simple_prompt(self, strategy):
        """Test tokenizing simple prompt."""
        tokens = strategy.tokenize("girl, hair, eyes")
        assert tokens == ["girl", "hair", "eyes"]

    def test_tokenize_and_blending(self, strategy):
        """Test AND blending tokenization."""
        tokens = strategy.tokenize("cat AND dog, playing")
        assert "cat" in tokens
        assert "dog" in tokens
        assert "playing" in tokens

    def test_tokenize_alternating_syntax(self, strategy):
        """Test alternating syntax [word1|word2]."""
        tokens = strategy.tokenize("[happy|sad] expression")
        # The alternating syntax extracts the alternates
        assert "happy" in tokens
        assert "sad" in tokens
        # Note: "expression" is in the same comma-delimited token as [happy|sad]

    def test_tokenize_multiple_alternates(self, strategy):
        """Test multiple alternating words."""
        tokens = strategy.tokenize("[red|blue|green] color")
        # Should extract all alternates
        assert "red" in tokens or "blue" in tokens or "green" in tokens

    def test_tokenize_mixed_syntax(self, strategy):
        """Test mixed syntax in one prompt."""
        tokens = strategy.tokenize("girl AND boy, [happy|sad], smiling")
        assert "girl" in tokens
        assert "boy" in tokens
        assert "smiling" in tokens
        # Should have alternates
        assert len(tokens) >= 4

    # Emphasis Extraction Tests
    def test_extract_explicit_weight(self, strategy):
        """Test extracting explicit weight syntax (word:1.5)."""
        emphasis = strategy.extract_emphasis_map("(beautiful:1.5) girl")
        assert emphasis["beautiful"] == 1.5

    def test_extract_explicit_weight_decimal(self, strategy):
        """Test extracting decimal weights."""
        emphasis = strategy.extract_emphasis_map("(detailed:1.25) image")
        assert emphasis["detailed"] == 1.25

    def test_extract_explicit_weight_low(self, strategy):
        """Test extracting low explicit weight."""
        emphasis = strategy.extract_emphasis_map("(background:0.3)")
        assert emphasis["background"] == 0.3

    def test_extract_standard_parentheses_emphasis(self, strategy):
        """Test standard parentheses emphasis without explicit weight."""
        emphasis = strategy.extract_emphasis_map("(beautiful) girl")
        assert emphasis["beautiful"] == 1.1

    def test_extract_double_parentheses_emphasis(self, strategy):
        """Test double parentheses emphasis."""
        emphasis = strategy.extract_emphasis_map("((masterpiece))")
        assert emphasis["masterpiece"] == 1.2

    def test_extract_bracket_deemphasis(self, strategy):
        """Test bracket de-emphasis."""
        emphasis = strategy.extract_emphasis_map("[watermark]")
        assert emphasis["watermark"] == 0.9

    def test_extract_step_scheduling(self, strategy):
        """Test step scheduling syntax [word:0.5]."""
        emphasis = strategy.extract_emphasis_map("[background:0.3] scene")
        assert emphasis["background"] == 0.3

    def test_extract_mixed_weight_types(self, strategy):
        """Test mixed explicit and implicit weights."""
        emphasis = strategy.extract_emphasis_map(
            "(beautiful:1.5), ((detailed)), [blur], (face:0.8)"
        )

        assert emphasis["beautiful"] == 1.5
        assert emphasis["detailed"] == 1.2
        assert emphasis["blur"] == 0.9
        assert emphasis["face"] == 0.8

    def test_extract_ignores_alternating(self, strategy):
        """Test that alternating syntax is ignored in emphasis."""
        emphasis = strategy.extract_emphasis_map("[happy|sad] expression")
        # Should not extract emphasis for alternating syntax
        assert "happy|sad" not in emphasis

    def test_extract_explicit_weight_priority(self, strategy):
        """Test that explicit weight takes priority over nesting."""
        emphasis = strategy.extract_emphasis_map("((word:1.5))")
        # Explicit weight should be used, not 1.2 from double parens
        assert emphasis["word"] == 1.5


class TestTokenizationIntegration:
    """Integration tests for tokenization strategies."""

    def test_all_strategies_produce_tokens(self):
        """Test that all strategies produce valid tokens."""
        prompt = "beautiful girl, long hair, blue eyes"

        simple = SimpleTokenization()
        sd = StableDiffusionTokenization()
        advanced = AdvancedTokenization()

        for strategy in [simple, sd, advanced]:
            tokens = strategy.tokenize(prompt)
            assert len(tokens) > 0
            assert all(isinstance(token, str) for token in tokens)

    def test_sd_and_advanced_handle_emphasis(self):
        """Test that SD and Advanced strategies handle emphasis."""
        prompt = "(beautiful) girl, ((masterpiece)), [blur]"

        sd = StableDiffusionTokenization()
        advanced = AdvancedTokenization()

        for strategy in [sd, advanced]:
            emphasis = strategy.extract_emphasis_map(prompt)
            assert "beautiful" in emphasis
            assert "masterpiece" in emphasis
            assert "blur" in emphasis
            assert emphasis["beautiful"] == 1.1
            assert emphasis["masterpiece"] == 1.2
            assert emphasis["blur"] == 0.9

    def test_advanced_handles_explicit_weights(self):
        """Test that advanced strategy handles explicit weights."""
        prompt = "(detailed:1.5), (face:0.8)"

        advanced = AdvancedTokenization()
        emphasis = advanced.extract_emphasis_map(prompt)

        assert emphasis["detailed"] == 1.5
        assert emphasis["face"] == 0.8

    def test_complex_prompt_tokenization(self):
        """Test tokenization of complex realistic prompt."""
        prompt = (
            "masterpiece, best quality, (beautiful:1.3), 1girl AND 1cat, "
            "long hair, [watermark], detailed face, outdoor scene"
        )

        advanced = AdvancedTokenization()
        tokens = advanced.tokenize(prompt)

        # Should have multiple tokens
        assert len(tokens) >= 5
        # Should handle AND blending
        assert "1girl" in tokens
        assert "1cat" in tokens

    def test_emphasis_extraction_comprehensive(self):
        """Test comprehensive emphasis extraction."""
        prompt = (
            "(ultra detailed:1.5), ((masterpiece)), (beautiful), "
            "[watermark], [blur:0.2], normal text"
        )

        advanced = AdvancedTokenization()
        emphasis = advanced.extract_emphasis_map(prompt)

        # Should extract all emphasis types
        assert emphasis["ultra detailed"] == 1.5
        assert emphasis["masterpiece"] == 1.2
        assert emphasis["beautiful"] == 1.1
        assert emphasis["watermark"] == 0.9
        assert emphasis["blur"] == 0.2

    def test_tokenization_preserves_important_syntax(self):
        """Test that tokenization preserves important SD syntax."""
        prompt = "(1girl:1.3), long hair AND short hair, [bad quality]"

        sd = StableDiffusionTokenization()
        tokens = sd.tokenize(prompt)

        # Tokens should preserve the original syntax for later processing
        assert any("1girl" in token for token in tokens)
        assert "long hair" in tokens
        assert "short hair" in tokens

    def test_edge_case_empty_parens(self):
        """Test handling of empty parentheses."""
        prompt = "(), girl, []"

        advanced = AdvancedTokenization()
        tokens = advanced.tokenize(prompt)
        emphasis = advanced.extract_emphasis_map(prompt)

        # Should handle gracefully
        assert "girl" in tokens
        assert len(emphasis) == 0  # Empty parens shouldn't create emphasis

    def test_edge_case_nested_mixed(self):
        """Test deeply nested and mixed emphasis."""
        prompt = "(((word1))), [[[word2]]], ((word3:1.5))"

        advanced = AdvancedTokenization()
        emphasis = advanced.extract_emphasis_map(prompt)

        assert emphasis["word1"] == pytest.approx(1.3, rel=0.01)
        assert emphasis["word2"] >= 0.1  # Capped at minimum
        assert emphasis["word3"] == 1.5  # Explicit weight
