"""Input data for statistical analysis."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StatisticalDataInput:
    """Input data container for statistical analysis."""

    variable_name: str
    data: list[float]

    def __post_init__(self):
        """Validate input data."""
        if len(self.data) == 0:
            raise ValueError("Data list cannot be empty")
