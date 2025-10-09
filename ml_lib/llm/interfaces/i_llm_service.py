"""Interface for generic LLM service operations."""

from abc import ABC, abstractmethod


class ILLMService(ABC):
    """Interface for LLM service operations."""

    @abstractmethod
    async def analyze_code_quality(self, code_content: str, context: str) -> str:
        """Analyze code quality using LLM.

        Args:
            code_content: The source code to analyze
            context: Additional context for analysis

        Returns:
            str: Analysis results and recommendations
        """

    @abstractmethod
    async def suggest_refactoring(
        self, code_content: str, issue_description: str
    ) -> str:
        """Suggest refactoring approaches for code issues.

        Args:
            code_content: The problematic code
            issue_description: Description of the architectural issue

        Returns:
            str: Refactoring suggestions and implementation guidance
        """

    @abstractmethod
    async def evaluate_architectural_compliance(
        self, file_content: str, rules: str
    ) -> str:
        """Evaluate architectural compliance using LLM.

        Args:
            file_content: Content of the file to evaluate
            rules: Architectural rules to check against

        Returns:
            str: Compliance evaluation results
        """
