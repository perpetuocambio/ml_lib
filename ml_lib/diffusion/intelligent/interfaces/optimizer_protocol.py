"""Protocol for parameter optimizer."""

from typing import Protocol, Optional, runtime_checkable


@runtime_checkable
class ParameterOptimizerProtocol(Protocol):
    """Protocol for parameter optimizer implementations."""

    def optimize(self, prompt_analysis, constraints: Optional[dict] = None):
        """
        Optimize generation parameters based on prompt analysis.

        Args:
            prompt_analysis: Analysis of the prompt
            constraints: Optional parameter constraints

        Returns:
            OptimizedParameters object
        """
        ...
