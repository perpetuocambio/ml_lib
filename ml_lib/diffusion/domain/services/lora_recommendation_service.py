"""LoRA Recommendation Domain Service.

This is a Domain Service (not Application Service) because it contains
business logic that doesn't naturally belong to a single entity.

Responsibilities:
- Recommend LoRAs based on prompt analysis
- Filter and rank LoRAs
- Apply business rules for selection

Uses:
- LoRA entities (rich domain objects)
- IModelRepository (for fetching LoRAs)
"""

from typing import Protocol
from ml_lib.diffusion.domain.entities.lora import LoRA, LoRARecommendation
from ml_lib.diffusion.domain.value_objects.weights import ConfidenceScore


class ILoRARepository(Protocol):
    """Repository interface for LoRA access."""

    def get_all_loras(self) -> list[LoRA]:
        """Get all available LoRAs."""
        ...

    def get_loras_by_base_model(self, base_model: str) -> list[LoRA]:
        """Get LoRAs compatible with base model."""
        ...


class LoRARecommendationService:
    """
    Domain service for recommending LoRAs.

    Uses rich LoRA entities and delegates selection logic to them.
    This is much cleaner than the original LoRARecommender service.
    """

    def __init__(self, repository: ILoRARepository):
        """
        Initialize service.

        Args:
            repository: LoRA repository for data access
        """
        self.repository = repository

    def recommend(
        self,
        prompt: str,
        base_model: str,
        max_loras: int = 3,
        min_confidence: float = 0.5,
    ) -> list[LoRARecommendation]:
        """
        Recommend LoRAs for prompt.

        Args:
            prompt: User's text prompt
            base_model: Base model architecture
            max_loras: Maximum LoRAs to recommend
            min_confidence: Minimum confidence threshold (0-1)

        Returns:
            List of LoRARecommendation ordered by relevance
        """
        # Get compatible LoRAs (Repository handles DB access)
        compatible_loras = self.repository.get_loras_by_base_model(base_model)

        # Calculate recommendations using entity behavior
        recommendations = []
        for lora in compatible_loras:
            # Entity knows how to check if it matches
            if not lora.is_compatible_with(base_model):
                continue

            # Create recommendation (entity calculates relevance)
            rec = LoRARecommendation.create(lora=lora, prompt=prompt)

            # Filter by confidence threshold
            if rec.confidence.value >= min_confidence:
                recommendations.append(rec)

        # Sort by confidence (highest first)
        recommendations.sort(key=lambda r: r.confidence.value, reverse=True)

        # Return top N
        return recommendations[:max_loras]

    def recommend_top(
        self,
        prompt: str,
        base_model: str,
        count: int = 1,
    ) -> LoRARecommendation | None:
        """
        Get single best recommendation.

        Args:
            prompt: User's text prompt
            base_model: Base model architecture
            count: Always 1 for this method

        Returns:
            Best LoRARecommendation or None if no good match
        """
        recommendations = self.recommend(
            prompt=prompt,
            base_model=base_model,
            max_loras=count,
            min_confidence=0.3,  # Lower threshold for "best available"
        )

        return recommendations[0] if recommendations else None

    def filter_confident_recommendations(
        self,
        recommendations: list[LoRARecommendation],
    ) -> list[LoRARecommendation]:
        """
        Filter to only confident recommendations.

        Args:
            recommendations: List of recommendations

        Returns:
            Only recommendations where is_confident() is True
        """
        # Entity knows when it's confident
        return [r for r in recommendations if r.is_confident()]

    def get_recommendations_by_trigger_words(
        self,
        prompt: str,
        base_model: str,
    ) -> list[LoRARecommendation]:
        """
        Get LoRAs that explicitly match trigger words.

        Args:
            prompt: User's text prompt
            base_model: Base model architecture

        Returns:
            Recommendations where prompt contains trigger words
        """
        compatible = self.repository.get_loras_by_base_model(base_model)

        recommendations = []
        for lora in compatible:
            # Entity knows if it matches prompt
            if lora.matches_prompt(prompt):
                rec = LoRARecommendation.create(lora=lora, prompt=prompt)
                recommendations.append(rec)

        # Sort by confidence
        recommendations.sort(key=lambda r: r.confidence.value, reverse=True)

        return recommendations
