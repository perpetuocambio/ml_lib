"""Protocol for LoRA recommender."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ml_lib.diffusion.models.prompt import PromptAnalysis
from ml_lib.diffusion.models.registry import ModelMetadata


@dataclass
class LoRARecommendation:
    """A single LoRA recommendation with confidence score."""

    metadata: ModelMetadata
    confidence: float
    weight: float = 1.0


@runtime_checkable
class LoRARecommenderProtocol(Protocol):
    """Protocol for LoRA recommender implementations."""

    def recommend(
        self,
        prompt_analysis: PromptAnalysis,
        base_model: str,
        max_loras: int = 5,
        min_confidence: float = 0.7
    ) -> list[LoRARecommendation]:
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
