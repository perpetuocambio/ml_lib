"""
Respuesta estructurada de un LLM.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMResponse:
    """Respuesta estructurada de un LLM."""

    content: str
    usage_tokens: int
    model_name: str
    confidence_score: float = 1.0  # 0.0 a 1.0

    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """
        Checks if the LLM response has a high confidence score.

        Args:
            threshold (float): The minimum confidence score to be considered high confidence (default is 0.8).

        Returns:
            bool: True if the `confidence_score` meets or exceeds the threshold, False otherwise.
        """
        return self.confidence_score >= threshold
