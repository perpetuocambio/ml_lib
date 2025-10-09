"""
Data cleaning operation Infrastructure DTO.

Infrastructure version - no Domain dependencies.
"""

from dataclasses import dataclass
from datetime import datetime

from infrastructure.data.extractors.processing.data_cleaning_configuration import (
    DataCleaningConfiguration,
)


@dataclass
class InfraDataCleaningOperation:
    """Individual cleaning operation applied to the dataset."""

    method: str  # Infrastructure uses string instead of enum
    applied_at: datetime
    rows_affected: int
    columns_affected: list[str]
    configuration: DataCleaningConfiguration
    success: bool
    error_message: str = ""

    def get_operation_summary(self) -> str:
        """Get summary of the cleaning operation."""
        if self.success:
            return f"{self.method}: {self.rows_affected} rows, {len(self.columns_affected)} columns affected"
        else:
            return f"{self.method}: FAILED - {self.error_message}"
