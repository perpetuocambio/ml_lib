"""AR model fitting result entity."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ArModelFitResult:
    """Result of AR model fitting."""

    parameters: list[float]
    fitted_values: list[float]
    residuals: list[float]
