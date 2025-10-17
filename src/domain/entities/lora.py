"""LoRA domain entity - Rich model with behavior.

This replaces the anemic LoRAInfo dataclass with a proper entity
that encapsulates both data and behavior.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from ml_lib.diffusion.domain.value_objects.weights import LoRAWeight, ConfidenceScore


@dataclass
class LoRA:
    """
    LoRA (Low-Rank Adaptation) entity.

    Rich domain entity that knows how to:
    - Validate itself
    - Compare with other LoRAs
    - Calculate relevance scores
    - Check compatibility
    """

    name: str
    path: Path
    base_model: str  # "SDXL", "SD15", "Pony", etc.
    weight: LoRAWeight
    trigger_words: list[str]
    tags: list[str]
    download_count: int = 0
    rating: float = 0.0

    def __post_init__(self):
        """Validate entity on construction."""
        if not self.name:
            raise ValueError("LoRA name cannot be empty")

        if not self.path.exists():
            raise ValueError(f"LoRA file not found: {self.path}")

        if not self.base_model:
            raise ValueError("LoRA must specify base_model")

        # Normalize lists
        if self.trigger_words is None:
            self.trigger_words = []
        if self.tags is None:
            self.tags = []

    @classmethod
    def create(
        cls,
        name: str,
        path: Path,
        base_model: str,
        weight: float = 1.0,
        trigger_words: list[str] | None = None,
        tags: list[str] | None = None,
        download_count: int = 0,
        rating: float = 0.0,
    ) -> "LoRA":
        """
        Factory method for creating LoRA with validation.

        Args:
            name: LoRA name
            path: Path to LoRA file
            base_model: Compatible base model
            weight: Alpha weight (default 1.0)
            trigger_words: Trigger words list
            tags: Tags list
            download_count: Download count (popularity)
            rating: Rating 0-5

        Returns:
            New LoRA entity
        """
        return cls(
            name=name,
            path=path,
            base_model=base_model,
            weight=LoRAWeight(weight),
            trigger_words=trigger_words or [],
            tags=tags or [],
            download_count=download_count,
            rating=rating,
        )

    def matches_prompt(self, prompt: str) -> bool:
        """
        Check if this LoRA is relevant for prompt.

        Args:
            prompt: Prompt text to check

        Returns:
            True if prompt contains trigger words or relevant tags
        """
        prompt_lower = prompt.lower()

        # Check trigger words (high relevance)
        for trigger in self.trigger_words:
            if trigger.lower() in prompt_lower:
                return True

        # Check tags (medium relevance)
        for tag in self.tags:
            if tag.lower() in prompt_lower:
                return True

        return False

    def calculate_relevance(self, prompt: str) -> ConfidenceScore:
        """
        Calculate relevance score for prompt.

        Args:
            prompt: Prompt text

        Returns:
            ConfidenceScore (0.0-1.0) indicating relevance
        """
        prompt_lower = prompt.lower()
        score = 0.0

        # Trigger words = high relevance
        trigger_matches = sum(
            1 for trigger in self.trigger_words
            if trigger.lower() in prompt_lower
        )
        score += trigger_matches * 0.3

        # Tags = medium relevance
        tag_matches = sum(
            1 for tag in self.tags
            if tag.lower() in prompt_lower
        )
        score += tag_matches * 0.15

        # Popularity bonus (normalized)
        import math
        if self.download_count > 0:
            popularity_score = min(math.log10(self.download_count) / 5, 0.2)
            score += popularity_score

        # Rating bonus (normalized)
        if self.rating > 0:
            rating_score = (self.rating / 5.0) * 0.1
            score += rating_score

        # Clamp to 0-1
        clamped_score = min(max(score, 0.0), 1.0)

        return ConfidenceScore(clamped_score)

    def is_compatible_with(self, base_model: str) -> bool:
        """
        Check if this LoRA is compatible with base model.

        Args:
            base_model: Base model architecture

        Returns:
            True if compatible
        """
        base_lower = base_model.lower()
        self_lower = self.base_model.lower()

        # Exact match
        if self_lower == base_lower:
            return True

        # SDXL variants are compatible
        if "sdxl" in self_lower and "sdxl" in base_lower:
            return True

        # Pony is based on SDXL but has specific requirements
        if "pony" in self_lower:
            return "pony" in base_lower or "sdxl" in base_lower

        # SD 1.5 variants
        if "1.5" in self_lower or "sd15" in self_lower:
            return "1.5" in base_lower or "sd15" in base_lower

        return False

    def scale_weight(self, factor: float) -> "LoRA":
        """
        Create new LoRA with scaled weight.

        Args:
            factor: Scale factor

        Returns:
            New LoRA with scaled weight
        """
        return LoRA(
            name=self.name,
            path=self.path,
            base_model=self.base_model,
            weight=self.weight.scale_by(factor),
            trigger_words=self.trigger_words,
            tags=self.tags,
            download_count=self.download_count,
            rating=self.rating,
        )

    def get_popularity_score(self) -> float:
        """
        Calculate popularity score (0-100).

        Combines downloads, rating.
        """
        import math

        # Normalize downloads (log scale, max ~100k)
        download_score = min(math.log10(max(self.download_count, 1)) / 5, 1.0) * 60

        # Rating (0-5 scale)
        rating_score = (self.rating / 5.0) * 40

        return download_score + rating_score

    def __str__(self) -> str:
        """Human-readable representation."""
        return f"LoRA({self.name}, weight={self.weight}, base={self.base_model})"

    def __repr__(self) -> str:
        """Debug representation."""
        return (
            f"LoRA(name='{self.name}', base_model='{self.base_model}', "
            f"weight={self.weight}, triggers={len(self.trigger_words)})"
        )


@dataclass
class LoRARecommendation:
    """
    Recommendation for using a LoRA.

    Contains the LoRA entity plus recommendation metadata.
    """

    lora: LoRA
    confidence: ConfidenceScore
    reasoning: str

    def __post_init__(self):
        """Validate recommendation."""
        if not self.lora:
            raise ValueError("Recommendation must have a LoRA")

        if not self.reasoning:
            raise ValueError("Recommendation must have reasoning")

    @classmethod
    def create(
        cls,
        lora: LoRA,
        prompt: str,
        reasoning: str | None = None,
    ) -> "LoRARecommendation":
        """
        Create recommendation from LoRA and prompt.

        Args:
            lora: LoRA entity
            prompt: Prompt that triggered recommendation
            reasoning: Optional custom reasoning

        Returns:
            New LoRARecommendation
        """
        confidence = lora.calculate_relevance(prompt)

        # Generate reasoning if not provided
        if reasoning is None:
            if lora.matches_prompt(prompt):
                triggers = [t for t in lora.trigger_words if t.lower() in prompt.lower()]
                if triggers:
                    reasoning = f"Matches trigger words: {', '.join(triggers)}"
                else:
                    reasoning = f"Relevant tags found in prompt"
            else:
                reasoning = f"High popularity (score: {lora.get_popularity_score():.0f})"

        return cls(
            lora=lora,
            confidence=confidence,
            reasoning=reasoning,
        )

    def is_confident(self) -> bool:
        """Check if confidence is high enough to use."""
        return self.confidence.is_high()

    def __str__(self) -> str:
        """Human-readable representation."""
        return f"{self.lora.name} (confidence: {self.confidence}, weight: {self.lora.weight})"
