"""Protocol for prompt analyzer."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class PromptAnalyzerProtocol(Protocol):
    """Protocol for prompt analyzer implementations."""

    def analyze(self, prompt: str):
        """
        Analyze a text prompt.

        Args:
            prompt: Text prompt to analyze

        Returns:
            PromptAnalysis object with semantic understanding
        """
        ...
