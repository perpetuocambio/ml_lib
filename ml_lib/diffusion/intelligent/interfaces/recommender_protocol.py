"""Protocol for LoRA recommender."""

from typing import Protocol, Optional, runtime_checkable


@runtime_checkable
class LoRARecommenderProtocol(Protocol):
    """Protocol for LoRA recommender implementations."""

    def recommend(
        self,
        prompt_analysis,
        base_model: str,
        max_loras: int = 5,
        min_confidence: float = 0.7
    ) -> list:
        """
        Recommend LoRAs based on prompt analysis.

        Args:
            prompt_analysis: Analysis of the prompt
            base_model: Base model being used
            max_loras: Maximum LoRAs to recommend
            min_confidence: Minimum confidence threshold

        Returns:
            List of LoRA recommendations
        """
        ...
