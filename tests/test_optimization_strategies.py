"""Tests for Optimization Strategies - Model-specific prompt optimization.

Tests the Strategy pattern implementation for prompt optimization.
"""

import pytest

from ml_lib.diffusion.domain.strategies.optimization import (
    SDXLOptimizationStrategy,
    PonyV6OptimizationStrategy,
    SD15OptimizationStrategy,
    OptimizationStrategyFactory,
)
from ml_lib.diffusion.domain.interfaces.analysis_strategies import QualityLevel


class TestSDXLOptimizationStrategy:
    """Tests for SDXL optimization strategy."""

    @pytest.fixture
    def strategy(self):
        """Create SDXL strategy."""
        return SDXLOptimizationStrategy()

    def test_get_supported_architecture(self, strategy):
        """Test that strategy reports correct architecture."""
        assert strategy.get_supported_architecture() == "SDXL"

    def test_optimize_adds_quality_tags(self, strategy):
        """Test that quality tags are appended for SDXL."""
        result = strategy.optimize(
            prompt="a beautiful landscape",
            negative_prompt="",
            quality_level=QualityLevel.HIGH,
        )

        assert "high quality" in result.optimized_prompt
        assert "detailed" in result.optimized_prompt
        assert "a beautiful landscape" in result.optimized_prompt

    def test_optimize_masterpiece_quality(self, strategy):
        """Test masterpiece quality tags."""
        result = strategy.optimize(
            prompt="test prompt",
            negative_prompt="",
            quality_level=QualityLevel.MASTERPIECE,
        )

        assert "masterpiece" in result.optimized_prompt
        assert "best quality" in result.optimized_prompt
        assert "ultra detailed" in result.optimized_prompt
        assert "8k uhd" in result.optimized_prompt

    def test_optimize_adds_negatives(self, strategy):
        """Test that SDXL negatives are added."""
        result = strategy.optimize(
            prompt="test",
            negative_prompt="",
            quality_level=QualityLevel.MEDIUM,
        )

        assert "low quality" in result.optimized_negative
        assert "worst quality" in result.optimized_negative
        assert "blurry" in result.optimized_negative

    def test_optimize_preserves_existing_negative(self, strategy):
        """Test that existing negative prompt is preserved."""
        result = strategy.optimize(
            prompt="test",
            negative_prompt="existing negative",
            quality_level=QualityLevel.MEDIUM,
        )

        assert "existing negative" in result.optimized_negative
        assert "low quality" in result.optimized_negative

    def test_optimize_normalizes_weights(self, strategy):
        """Test that extreme weights are normalized."""
        result = strategy.optimize(
            prompt="((((((test))))))",  # 6x nesting = 1.6x weight
            negative_prompt="",
            quality_level=QualityLevel.MEDIUM,
        )

        # Should be capped to 1.4x max (4 levels)
        assert result.optimized_prompt.count("(") <= 4

    def test_modifications_list(self, strategy):
        """Test that modifications are tracked."""
        result = strategy.optimize(
            prompt="test",
            negative_prompt="",
            quality_level=QualityLevel.HIGH,
        )

        assert len(result.modifications) > 0
        assert any("quality tags" in mod for mod in result.modifications)


class TestPonyV6OptimizationStrategy:
    """Tests for Pony V6 optimization strategy."""

    @pytest.fixture
    def strategy(self):
        """Create Pony V6 strategy."""
        return PonyV6OptimizationStrategy()

    def test_get_supported_architecture(self, strategy):
        """Test that strategy reports correct architecture."""
        assert strategy.get_supported_architecture() == "Pony V6"

    def test_optimize_prepends_score_tags(self, strategy):
        """Test that Pony score tags are prepended."""
        result = strategy.optimize(
            prompt="a beautiful character",
            negative_prompt="",
            quality_level=QualityLevel.HIGH,
        )

        # Score tags should be at the beginning
        assert result.optimized_prompt.startswith("score_")
        assert "score_8" in result.optimized_prompt
        assert "a beautiful character" in result.optimized_prompt

    def test_optimize_masterpiece_score_tags(self, strategy):
        """Test masterpiece quality score tags."""
        result = strategy.optimize(
            prompt="test",
            negative_prompt="",
            quality_level=QualityLevel.MASTERPIECE,
        )

        assert "score_9" in result.optimized_prompt
        assert "score_8_up" in result.optimized_prompt
        assert "score_7_up" in result.optimized_prompt

    def test_optimize_adds_anatomical_negatives(self, strategy):
        """Test that Pony anatomical negatives are added."""
        result = strategy.optimize(
            prompt="test",
            negative_prompt="",
            quality_level=QualityLevel.MEDIUM,
        )

        assert "bad anatomy" in result.optimized_negative
        assert "bad hands" in result.optimized_negative
        assert "score_4" in result.optimized_negative
        assert "score_3" in result.optimized_negative

    def test_optimize_weight_cap_15x(self, strategy):
        """Test that weights are capped at 1.5x for Pony."""
        result = strategy.optimize(
            prompt="((((((test))))))",  # 6x nesting
            negative_prompt="",
            quality_level=QualityLevel.MEDIUM,
        )

        # Should be capped to 1.5x max (5 levels)
        assert result.optimized_prompt.count("(") <= 5


class TestSD15OptimizationStrategy:
    """Tests for SD 1.5 optimization strategy."""

    @pytest.fixture
    def strategy(self):
        """Create SD 1.5 strategy."""
        return SD15OptimizationStrategy()

    def test_get_supported_architecture(self, strategy):
        """Test that strategy reports correct architecture."""
        assert strategy.get_supported_architecture() == "SD 1.5"

    def test_optimize_prepends_quality_tags(self, strategy):
        """Test that quality tags are prepended for SD 1.5."""
        result = strategy.optimize(
            prompt="a beautiful scene",
            negative_prompt="",
            quality_level=QualityLevel.HIGH,
        )

        # Quality tags should be at the beginning
        assert result.optimized_prompt.startswith("high quality")
        assert "a beautiful scene" in result.optimized_prompt

    def test_optimize_masterpiece_quality(self, strategy):
        """Test masterpiece quality tags."""
        result = strategy.optimize(
            prompt="test",
            negative_prompt="",
            quality_level=QualityLevel.MASTERPIECE,
        )

        assert "masterpiece" in result.optimized_prompt
        assert "best quality" in result.optimized_prompt
        assert "highly detailed" in result.optimized_prompt

    def test_optimize_adds_sd15_negatives(self, strategy):
        """Test that SD 1.5 negatives are added."""
        result = strategy.optimize(
            prompt="test",
            negative_prompt="",
            quality_level=QualityLevel.MEDIUM,
        )

        assert "low quality" in result.optimized_negative
        assert "worst quality" in result.optimized_negative
        assert "bad anatomy" in result.optimized_negative


class TestOptimizationStrategyFactory:
    """Tests for OptimizationStrategyFactory."""

    def test_create_sdxl_strategy(self):
        """Test creating SDXL strategy."""
        strategy = OptimizationStrategyFactory.create("SDXL")
        assert isinstance(strategy, SDXLOptimizationStrategy)

    def test_create_pony_strategy(self):
        """Test creating Pony V6 strategy."""
        strategy = OptimizationStrategyFactory.create("Pony V6")
        assert isinstance(strategy, PonyV6OptimizationStrategy)

    def test_create_sd15_strategy(self):
        """Test creating SD 1.5 strategy."""
        strategy = OptimizationStrategyFactory.create("SD 1.5")
        assert isinstance(strategy, SD15OptimizationStrategy)

    def test_create_sd15_alias(self):
        """Test creating SD 1.5 with alias."""
        strategy = OptimizationStrategyFactory.create("SD15")
        assert isinstance(strategy, SD15OptimizationStrategy)

    def test_create_unsupported_architecture_raises(self):
        """Test that unsupported architecture raises error."""
        with pytest.raises(ValueError, match="Unsupported architecture"):
            OptimizationStrategyFactory.create("InvalidModel")

    def test_get_supported_architectures(self):
        """Test getting list of supported architectures."""
        architectures = OptimizationStrategyFactory.get_supported_architectures()

        assert "SDXL" in architectures
        assert "Pony V6" in architectures
        assert "SD 1.5" in architectures
        assert len(architectures) >= 3

    def test_register_custom_strategy(self):
        """Test registering custom strategy."""

        class CustomStrategy(SDXLOptimizationStrategy):
            def get_supported_architecture(self) -> str:
                return "Custom"

        OptimizationStrategyFactory.register("Custom", CustomStrategy)

        strategy = OptimizationStrategyFactory.create("Custom")
        assert isinstance(strategy, CustomStrategy)
        assert strategy.get_supported_architecture() == "Custom"


class TestOptimizationIntegration:
    """Integration tests for optimization strategies."""

    def test_all_strategies_produce_valid_output(self):
        """Test that all strategies produce valid output."""
        architectures = ["SDXL", "Pony V6", "SD 1.5"]
        prompt = "a beautiful anime girl with long hair"
        negative = "blurry"

        for arch in architectures:
            strategy = OptimizationStrategyFactory.create(arch)
            result = strategy.optimize(
                prompt=prompt,
                negative_prompt=negative,
                quality_level=QualityLevel.HIGH,
            )

            # Basic validation
            assert result.optimized_prompt
            assert result.optimized_negative
            assert len(result.modifications) > 0
            assert prompt in result.optimized_prompt or "anime" in result.optimized_prompt

    def test_quality_levels_affect_output(self):
        """Test that different quality levels produce different outputs."""
        strategy = OptimizationStrategyFactory.create("SDXL")

        low_result = strategy.optimize("test", "", QualityLevel.LOW)
        high_result = strategy.optimize("test", "", QualityLevel.HIGH)
        master_result = strategy.optimize("test", "", QualityLevel.MASTERPIECE)

        # Higher quality should add more tags
        assert len(master_result.optimized_prompt) > len(high_result.optimized_prompt)
        assert len(high_result.optimized_prompt) >= len(low_result.optimized_prompt)
