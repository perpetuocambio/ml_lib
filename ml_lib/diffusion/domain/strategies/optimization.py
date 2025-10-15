"""Prompt Optimization Strategies - Model-specific optimizations.

Each model architecture (SDXL, Pony V6, SD 1.5) has different requirements
for optimal prompt structure and quality tags.
"""

import re
from typing import Optional

from ml_lib.diffusion.domain.interfaces.analysis_strategies import (
    IOptimizationStrategy,
    OptimizationResult,
    QualityLevel,
)


class SDXLOptimizationStrategy(IOptimizationStrategy):
    """
    Optimization strategy for Stable Diffusion XL.

    SDXL characteristics:
    - Natural language friendly
    - Quality tags work best when appended
    - Supports longer prompts (75 tokens vs 77)
    - Weight cap: 1.4x recommended
    """

    def optimize(
        self,
        prompt: str,
        negative_prompt: str,
        quality_level: QualityLevel,
    ) -> OptimizationResult:
        """Optimize for SDXL."""
        modifications = []

        # Add quality tags at end (SDXL reads prompts left-to-right)
        quality_tags = self._get_quality_tags(quality_level)
        if quality_tags:
            optimized_prompt = f"{prompt}, {quality_tags}"
            modifications.append(f"Added quality tags: {quality_tags}")
        else:
            optimized_prompt = prompt

        # Normalize weights to SDXL's sweet spot (1.4x max)
        optimized_prompt = self._normalize_weights(optimized_prompt, max_weight=1.4)
        if "normalized weights" not in modifications:
            modifications.append("Normalized weights to 1.4x max")

        # Add standard SDXL negatives if not present
        sdxl_negatives = "low quality, worst quality, blurry, jpeg artifacts"
        if negative_prompt:
            optimized_negative = f"{negative_prompt}, {sdxl_negatives}"
        else:
            optimized_negative = sdxl_negatives
        modifications.append("Added SDXL quality negatives")

        return OptimizationResult(
            optimized_prompt=optimized_prompt,
            optimized_negative=optimized_negative,
            modifications=modifications,
        )

    def get_supported_architecture(self) -> str:
        """Return supported architecture."""
        return "SDXL"

    def _get_quality_tags(self, quality: QualityLevel) -> str:
        """Get quality tags for SDXL."""
        quality_map = {
            QualityLevel.MASTERPIECE: "masterpiece, best quality, ultra detailed, 8k uhd",
            QualityLevel.HIGH: "high quality, detailed, professional",
            QualityLevel.MEDIUM: "good quality",
            QualityLevel.LOW: "",
        }
        return quality_map.get(quality, "")

    def _normalize_weights(self, prompt: str, max_weight: float) -> str:
        """Normalize emphasis weights to max value."""

        def replace_weight(match):
            content = match.group(1)
            # Count parentheses depth
            depth = match.group(0).count("(")
            weight = 1.0 + (depth * 0.1)

            if weight > max_weight:
                # Cap at max_weight
                capped_depth = int((max_weight - 1.0) / 0.1)
                return "(" * capped_depth + content + ")" * capped_depth
            else:
                return match.group(0)

        # Replace multiple parentheses
        pattern = r"\(+([^()]+)\)+"
        return re.sub(pattern, replace_weight, prompt)


class PonyV6OptimizationStrategy(IOptimizationStrategy):
    """
    Optimization strategy for Pony Diffusion V6.

    Pony V6 characteristics:
    - Uses score tags (score_9, score_8_up, etc.)
    - Quality tags prepended for better effect
    - Strong anatomical understanding
    - Weight cap: 1.5x recommended
    """

    def optimize(
        self,
        prompt: str,
        negative_prompt: str,
        quality_level: QualityLevel,
    ) -> OptimizationResult:
        """Optimize for Pony V6."""
        modifications = []

        # Prepend Pony score tags
        score_tags = self._get_score_tags(quality_level)
        optimized_prompt = f"{score_tags}, {prompt}"
        modifications.append(f"Prepended Pony score tags: {score_tags}")

        # Normalize weights to Pony's tolerance (1.5x max)
        optimized_prompt = self._normalize_weights(optimized_prompt, max_weight=1.5)
        modifications.append("Normalized weights to 1.5x max")

        # Add Pony-specific anatomical negatives
        pony_negatives = (
            "score_4, score_3, score_2, score_1, "
            "bad anatomy, bad hands, missing fingers, extra fingers, "
            "blurry, low quality"
        )
        if negative_prompt:
            optimized_negative = f"{pony_negatives}, {negative_prompt}"
        else:
            optimized_negative = pony_negatives
        modifications.append("Added Pony anatomical negatives")

        return OptimizationResult(
            optimized_prompt=optimized_prompt,
            optimized_negative=optimized_negative,
            modifications=modifications,
        )

    def get_supported_architecture(self) -> str:
        """Return supported architecture."""
        return "Pony V6"

    def _get_score_tags(self, quality: QualityLevel) -> str:
        """Get Pony score tags."""
        quality_map = {
            QualityLevel.MASTERPIECE: "score_9, score_8_up, score_7_up",
            QualityLevel.HIGH: "score_8, score_7_up, score_6_up",
            QualityLevel.MEDIUM: "score_7, score_6_up",
            QualityLevel.LOW: "score_6",
        }
        return quality_map.get(quality, "score_7")

    def _normalize_weights(self, prompt: str, max_weight: float) -> str:
        """Normalize emphasis weights."""

        def replace_weight(match):
            content = match.group(1)
            depth = match.group(0).count("(")
            weight = 1.0 + (depth * 0.1)

            if weight > max_weight:
                capped_depth = int((max_weight - 1.0) / 0.1)
                return "(" * capped_depth + content + ")" * capped_depth
            else:
                return match.group(0)

        pattern = r"\(+([^()]+)\)+"
        return re.sub(pattern, replace_weight, prompt)


class SD15OptimizationStrategy(IOptimizationStrategy):
    """
    Optimization strategy for Stable Diffusion 1.5.

    SD 1.5 characteristics:
    - Older architecture, more conservative
    - Quality tags work best when prepended
    - Shorter token limit (77 tokens)
    - Weight cap: 1.5x recommended
    """

    def optimize(
        self,
        prompt: str,
        negative_prompt: str,
        quality_level: QualityLevel,
    ) -> OptimizationResult:
        """Optimize for SD 1.5."""
        modifications = []

        # Prepend quality tags (SD 1.5 emphasizes earlier tokens)
        quality_tags = self._get_quality_tags(quality_level)
        if quality_tags:
            optimized_prompt = f"{quality_tags}, {prompt}"
            modifications.append(f"Prepended quality tags: {quality_tags}")
        else:
            optimized_prompt = prompt

        # Normalize weights conservatively (1.5x max)
        optimized_prompt = self._normalize_weights(optimized_prompt, max_weight=1.5)
        modifications.append("Normalized weights to 1.5x max")

        # Add standard SD 1.5 negatives
        sd15_negatives = "low quality, worst quality, blurry, bad anatomy"
        if negative_prompt:
            optimized_negative = f"{negative_prompt}, {sd15_negatives}"
        else:
            optimized_negative = sd15_negatives
        modifications.append("Added SD 1.5 quality negatives")

        return OptimizationResult(
            optimized_prompt=optimized_prompt,
            optimized_negative=optimized_negative,
            modifications=modifications,
        )

    def get_supported_architecture(self) -> str:
        """Return supported architecture."""
        return "SD 1.5"

    def _get_quality_tags(self, quality: QualityLevel) -> str:
        """Get quality tags for SD 1.5."""
        quality_map = {
            QualityLevel.MASTERPIECE: "masterpiece, best quality, highly detailed",
            QualityLevel.HIGH: "high quality, detailed",
            QualityLevel.MEDIUM: "good quality",
            QualityLevel.LOW: "",
        }
        return quality_map.get(quality, "")

    def _normalize_weights(self, prompt: str, max_weight: float) -> str:
        """Normalize emphasis weights."""

        def replace_weight(match):
            content = match.group(1)
            depth = match.group(0).count("(")
            weight = 1.0 + (depth * 0.1)

            if weight > max_weight:
                capped_depth = int((max_weight - 1.0) / 0.1)
                return "(" * capped_depth + content + ")" * capped_depth
            else:
                return match.group(0)

        pattern = r"\(+([^()]+)\)+"
        return re.sub(pattern, replace_weight, prompt)


class OptimizationStrategyFactory:
    """
    Factory for creating optimization strategies based on model architecture.

    Usage:
        strategy = OptimizationStrategyFactory.create("SDXL")
        result = strategy.optimize(prompt, negative, quality)
    """

    _strategies: dict[str, type[IOptimizationStrategy]] = {
        "SDXL": SDXLOptimizationStrategy,
        "Pony V6": PonyV6OptimizationStrategy,
        "SD 1.5": SD15OptimizationStrategy,
        "SD15": SD15OptimizationStrategy,  # Alias
    }

    @classmethod
    def create(cls, architecture: str) -> IOptimizationStrategy:
        """
        Create optimization strategy for architecture.

        Args:
            architecture: Model architecture name

        Returns:
            Optimization strategy instance

        Raises:
            ValueError: If architecture not supported
        """
        strategy_class = cls._strategies.get(architecture)

        if strategy_class is None:
            raise ValueError(
                f"Unsupported architecture: {architecture}. "
                f"Supported: {list(cls._strategies.keys())}"
            )

        return strategy_class()

    @classmethod
    def register(cls, architecture: str, strategy_class: type[IOptimizationStrategy]):
        """
        Register custom optimization strategy.

        Args:
            architecture: Architecture name
            strategy_class: Strategy class to register
        """
        cls._strategies[architecture] = strategy_class

    @classmethod
    def get_supported_architectures(cls) -> list[str]:
        """Get list of supported architectures."""
        return list(cls._strategies.keys())
