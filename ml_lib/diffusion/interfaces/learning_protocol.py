"""Protocol for learning engine."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class LearningEngineProtocol(Protocol):
    """Protocol for learning engine implementations."""

    def record_feedback(self, feedback):
        """
        Record user feedback.

        Args:
            feedback: GenerationFeedback object
        """
        ...

    def get_lora_adjustment_factor(self, lora_name: str) -> float:
        """
        Get adjustment factor for a LoRA based on past feedback.

        Args:
            lora_name: Name of the LoRA

        Returns:
            Adjustment factor (1.0 = no change)
        """
        ...
