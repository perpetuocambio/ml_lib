"""Time series data input entity."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TimeSeriesDataInput:
    """Input data container for time series analysis."""

    series_name: str
    values: list[float]
    timestamps: list[str] | None = None
    frequency: str | None = None

    def __post_init__(self):
        """Validate input data."""
        if len(self.values) == 0:
            raise ValueError("Time series data cannot be empty")

        if self.timestamps is not None and len(self.timestamps) != len(self.values):
            raise ValueError("Timestamps and values must have same length")
