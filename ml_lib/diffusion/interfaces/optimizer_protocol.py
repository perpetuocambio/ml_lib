"""Protocol for parameter optimizer."""

from typing import Protocol, Optional, runtime_checkable

from ml_lib.diffusion.models.prompt import PromptAnalysis, OptimizedParameters


@runtime_checkable
class ParameterOptimizerProtocol(Protocol):
    """Protocol for parameter optimizer implementations."""

    def optimize(
        self,
        prompt_analysis: PromptAnalysis,
        constraints: Optional[dict[str, int | float | str]] = None
    ) -> OptimizedParameters:
        """
        Optimize generation parameters based on prompt analysis.

        Args:
            prompt_analysis: Analysis of the prompt
            constraints: Optional parameter constraints

        Returns:
            OptimizedParameters object
        """
        ...
