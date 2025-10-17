"""Protocol for prompt analyzer."""

from typing import Protocol, runtime_checkable

from ml_lib.diffusion.domain.value_objects_models.prompt import PromptAnalysis


@runtime_checkable
class PromptAnalyzerProtocol(Protocol):
    """Protocol for prompt analyzer implementations."""

    def analyze(self, prompt: str) -> PromptAnalysis:
        """
        Analyze a text prompt.

        Args:
            prompt: Text prompt to analyze

        Returns:
            PromptAnalysis object with semantic understanding
        """
        ...
