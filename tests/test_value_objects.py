"""Tests for Value Objects."""

import pytest
from ml_lib.diffusion.domain.value_objects.weights import (
    LoRAWeight,
    PromptWeight,
    ConfidenceScore,
    CFGScale,
)


class TestLoRAWeight:
    """Tests for LoRAWeight Value Object."""

    def test_create_valid_weight(self):
        """Test creating valid weight."""
        weight = LoRAWeight(0.8)
        assert weight.value == 0.8

    def test_reject_negative_weight(self):
        """Test rejecting negative weights."""
        with pytest.raises(ValueError, match="must be between"):
            LoRAWeight(-0.5)

    def test_reject_too_high_weight(self):
        """Test rejecting weights above max."""
        with pytest.raises(ValueError, match="must be between"):
            LoRAWeight(3.0)

    def test_default_weight(self):
        """Test default weight is 1.0."""
        weight = LoRAWeight.default()
        assert weight.value == 1.0

    def test_from_float_clamps(self):
        """Test from_float clamps to valid range."""
        weight = LoRAWeight.from_float(5.0)
        assert weight.value == 2.0  # Clamped to max

        weight = LoRAWeight.from_float(-1.0)
        assert weight.value == 0.0  # Clamped to min

    def test_scale_by(self):
        """Test scaling weight."""
        weight = LoRAWeight(1.0)
        scaled = weight.scale_by(0.8)
        assert scaled.value == 0.8

    def test_immutability(self):
        """Test that LoRAWeight is immutable."""
        weight = LoRAWeight(1.0)
        with pytest.raises(Exception):  # Frozen dataclass raises on modification
            weight.value = 0.5

    def test_float_conversion(self):
        """Test converting to float."""
        weight = LoRAWeight(0.75)
        assert float(weight) == 0.75

    def test_string_representation(self):
        """Test string formatting."""
        weight = LoRAWeight(0.856)
        assert str(weight) == "0.86"


class TestConfidenceScore:
    """Tests for ConfidenceScore Value Object."""

    def test_create_valid_score(self):
        """Test creating valid score."""
        score = ConfidenceScore(0.75)
        assert score.value == 0.75

    def test_reject_invalid_scores(self):
        """Test rejecting out-of-range scores."""
        with pytest.raises(ValueError):
            ConfidenceScore(-0.1)

        with pytest.raises(ValueError):
            ConfidenceScore(1.1)

    def test_from_percentage(self):
        """Test creating from percentage."""
        score = ConfidenceScore.from_percentage(75.0)
        assert score.value == 0.75

    def test_to_percentage(self):
        """Test converting to percentage."""
        score = ConfidenceScore(0.75)
        assert score.to_percentage() == 75.0

    def test_confidence_levels(self):
        """Test confidence level checks."""
        low = ConfidenceScore(0.3)
        medium = ConfidenceScore(0.5)
        high = ConfidenceScore(0.8)

        assert low.is_low()
        assert not low.is_medium()
        assert not low.is_high()

        assert not medium.is_low()
        assert medium.is_medium()
        assert not medium.is_high()

        assert not high.is_low()
        assert not high.is_medium()
        assert high.is_high()

    def test_factory_methods(self):
        """Test factory methods for common values."""
        assert ConfidenceScore.low().value == 0.3
        assert ConfidenceScore.medium().value == 0.5
        assert ConfidenceScore.high().value == 0.8
        assert ConfidenceScore.very_high().value == 0.95

    def test_string_representation(self):
        """Test string formatting."""
        score = ConfidenceScore(0.753)
        assert "75" in str(score)  # Should show as percentage


class TestPromptWeight:
    """Tests for PromptWeight Value Object."""

    def test_create_valid_weight(self):
        """Test creating valid prompt weight."""
        weight = PromptWeight(1.1)
        assert weight.value == 1.1

    def test_default_weight(self):
        """Test default is 1.0."""
        weight = PromptWeight.default()
        assert weight.value == 1.0

    def test_emphasis_helpers(self):
        """Test emphasis helper methods."""
        emphasized = PromptWeight.emphasized()
        assert emphasized.value == 1.1

        strong = PromptWeight.strongly_emphasized()
        assert strong.value == 1.21

        de_emph = PromptWeight.de_emphasized()
        assert de_emph.value == 0.9

    def test_is_emphasized(self):
        """Test checking if weight is emphasis."""
        normal = PromptWeight.default()
        emphasized = PromptWeight.emphasized()

        assert not normal.is_emphasized()
        assert emphasized.is_emphasized()

    def test_is_default(self):
        """Test checking if weight is default."""
        default = PromptWeight.default()
        custom = PromptWeight(1.5)

        assert default.is_default()
        assert not custom.is_default()


class TestCFGScale:
    """Tests for CFGScale Value Object."""

    def test_create_valid_cfg(self):
        """Test creating valid CFG scale."""
        cfg = CFGScale(7.0)
        assert cfg.value == 7.0

    def test_reject_invalid_cfg(self):
        """Test rejecting out-of-range CFG."""
        with pytest.raises(ValueError):
            CFGScale(0.5)

        with pytest.raises(ValueError):
            CFGScale(50.0)

    def test_default_for_model(self):
        """Test model-specific defaults."""
        sdxl_cfg = CFGScale.default_for_model("SDXL")
        assert sdxl_cfg.value == 7.0

        pony_cfg = CFGScale.default_for_model("Pony")
        assert pony_cfg.value == 6.0

        sd15_cfg = CFGScale.default_for_model("SD 1.5")
        assert sd15_cfg.value == 7.5

    def test_is_low_high(self):
        """Test checking if CFG is low/high."""
        low = CFGScale(3.0)
        normal = CFGScale(7.0)
        high = CFGScale(15.0)

        assert low.is_low()
        assert not low.is_high()

        assert not normal.is_low()
        assert not normal.is_high()

        assert not high.is_low()
        assert high.is_high()
