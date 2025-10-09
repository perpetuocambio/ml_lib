"""
Data cleaning result Infrastructure DTO.

Infrastructure version - no Domain dependencies.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from infrastructure.data.extractors.processing.data_cleaning_operation import (
    InfraDataCleaningOperation,
)


@dataclass(frozen=True)
class InfraDataCleaningResult:
    """Result of data cleaning operations on a dataset."""

    # Core identification
    cleaning_id: str
    dataset_id: str
    source_file_path: Path
    output_file_path: Path

    # Cleaning configuration
    methods_applied: list[InfraDataCleaningOperation]

    # Results
    original_row_count: int
    cleaned_row_count: int
    original_column_count: int
    cleaned_column_count: int

    # Quality metrics
    duplicate_rows_removed: int
    missing_values_handled: int
    outliers_detected: int
    outliers_removed: int
    data_quality_score: float  # 0.0 to 1.0

    # Metadata
    cleaning_timestamp: datetime
    processing_duration_seconds: float
    confidence_score: float  # Infrastructure uses float instead of ConfidenceMetric

    # Issues and warnings
    warnings: list[str]
    errors: list[str]

    def get_rows_removed_count(self) -> int:
        """Get total number of rows removed during cleaning."""
        return self.original_row_count - self.cleaned_row_count

    def get_rows_removed_percentage(self) -> float:
        """Get percentage of rows removed during cleaning."""
        if self.original_row_count == 0:
            return 0.0
        return (self.get_rows_removed_count() / self.original_row_count) * 100

    def get_columns_modified_count(self) -> int:
        """Get number of unique columns that were modified."""
        all_columns = set()
        for operation in self.methods_applied:
            all_columns.update(operation.columns_affected)
        return len(all_columns)

    def was_successful(self) -> bool:
        """Check if the cleaning process was overall successful."""
        return (
            len(self.errors) == 0
            and any(op.success for op in self.methods_applied)
            and self.cleaned_row_count > 0
        )

    def get_cleaning_summary(self) -> str:
        """Get comprehensive summary of cleaning results."""
        summary_parts = [
            "DATA CLEANING SUMMARY",
            f"Dataset: {self.dataset_id}",
            "",
            f"OPERATIONS APPLIED: {len(self.methods_applied)}",
        ]

        for i, operation in enumerate(self.methods_applied, 1):
            status = "✅" if operation.success else "❌"
            summary_parts.append(f"  {i}. {status} {operation.get_operation_summary()}")

        summary_parts.extend(
            [
                "",
                "DATA CHANGES:",
                f"• Rows: {self.original_row_count} → {self.cleaned_row_count} ({self.get_rows_removed_percentage():.1f}% removed)",
                f"• Columns: {self.original_column_count} → {self.cleaned_column_count}",
                f"• Columns modified: {self.get_columns_modified_count()}",
                "",
                "QUALITY IMPROVEMENTS:",
                f"• Duplicates removed: {self.duplicate_rows_removed}",
                f"• Missing values handled: {self.missing_values_handled}",
                f"• Outliers detected: {self.outliers_detected}",
                f"• Outliers removed: {self.outliers_removed}",
                "",
                f"QUALITY SCORE: {self.data_quality_score:.2f}/1.0",
                f"PROCESSING TIME: {self.processing_duration_seconds:.2f}s",
            ]
        )

        if self.warnings:
            summary_parts.extend(["", f"WARNINGS ({len(self.warnings)}):"])
            for warning in self.warnings[:3]:  # Show first 3 warnings
                summary_parts.append(f"  ⚠️ {warning}")

        if self.errors:
            summary_parts.extend(["", f"ERRORS ({len(self.errors)}):"])
            for error in self.errors[:3]:  # Show first 3 errors
                summary_parts.append(f"  ❌ {error}")

        return "\\n".join(summary_parts)

    def get_quality_assessment(self) -> str:
        """Get quality assessment based on cleaning results."""
        if self.data_quality_score >= 0.9:
            return "Excellent"
        elif self.data_quality_score >= 0.8:
            return "Good"
        elif self.data_quality_score >= 0.7:
            return "Fair"
        elif self.data_quality_score >= 0.6:
            return "Poor"
        else:
            return "Critical"

    def needs_manual_review(self) -> bool:
        """Check if the dataset needs manual review after cleaning."""
        return (
            self.data_quality_score < 0.7
            or self.get_rows_removed_percentage() > 20
            or len(self.errors) > 0
            or len(self.warnings) > 5
        )
