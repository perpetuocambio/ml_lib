"""Prompt analysis interface."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class IPromptAnalyzer(Protocol):
    """
    Protocol for prompt analysis.

    Analyzes prompts to extract semantic information.
    """

    def analyze(self, prompt: str) -> any:  # Will be PromptAnalysis after migration
        """
        Analyze a prompt.

        Args:
            prompt: Text prompt to analyze

        Returns:
            PromptAnalysis with extracted information
        """
        ...
