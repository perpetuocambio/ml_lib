"""Infrastructure-specific configuration for data processing operations."""

from dataclasses import dataclass


@dataclass
class InfraDataProcessingConfig:
    """Infrastructure-specific configuration for data processing operations."""

    missing_strategy: str = "drop"  # "drop", "fill_mean", "fill_zero"
    outlier_method: str = "zscore"  # "zscore", "iqr", "none"
    outlier_threshold: float = 3.0
    duplicate_columns: list[str] = None

    def __post_init__(self) -> None:
        if self.duplicate_columns is None:
            self.duplicate_columns = []
