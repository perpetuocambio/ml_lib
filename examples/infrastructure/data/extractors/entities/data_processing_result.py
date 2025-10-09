"""Infrastructure DTO for data processing results."""

from pathlib import Path


class DataProcessingResult:
    """Infrastructure DTO for data processing results."""

    def __init__(
        self,
        processing_id: str,
        dataset_id: str,
        source_file_path: Path,
        output_file_path: Path | None,
        original_row_count: int,
        final_row_count: int,
        original_column_count: int,
        final_column_count: int,
        execution_time_seconds: float,
        warnings: list[str],
        errors: list[str],
        operations_summary_text: str,
    ):
        self.processing_id = processing_id
        self.dataset_id = dataset_id
        self.source_file_path = source_file_path
        self.output_file_path = output_file_path
        self.original_row_count = original_row_count
        self.final_row_count = final_row_count
        self.original_column_count = original_column_count
        self.final_column_count = final_column_count
        self.execution_time_seconds = execution_time_seconds
        self.warnings = warnings
        self.errors = errors
        self.operations_summary_text = operations_summary_text

    def has_errors(self) -> bool:
        """Check if processing resulted in errors."""
        return len(self.errors) > 0

    def has_warnings(self) -> bool:
        """Check if processing resulted in warnings."""
        return len(self.warnings) > 0

    def is_successful(self) -> bool:
        """Check if processing completed successfully (no errors)."""
        return not self.has_errors()

    def get_rows_reduction_count(self) -> int:
        """Get number of rows reduced during processing."""
        return self.original_row_count - self.final_row_count

    def get_rows_reduction_percentage(self) -> float:
        """Get percentage of rows reduced during processing."""
        if self.original_row_count == 0:
            return 0.0
        return (self.get_rows_reduction_count() / self.original_row_count) * 100

    def get_columns_reduction_count(self) -> int:
        """Get number of columns reduced during processing."""
        return self.original_column_count - self.final_column_count

    def get_columns_reduction_percentage(self) -> float:
        """Get percentage of columns reduced during processing."""
        if self.original_column_count == 0:
            return 0.0
        return (self.get_columns_reduction_count() / self.original_column_count) * 100

    def get_data_reduction_summary(self) -> str:
        """Get human-readable summary of data reduction."""
        rows_change = self.get_rows_reduction_count()
        cols_change = self.get_columns_reduction_count()

        if rows_change == 0 and cols_change == 0:
            return "No data reduction"

        parts = []
        if rows_change > 0:
            parts.append(
                f"{rows_change} rows removed ({self.get_rows_reduction_percentage():.1f}%)"
            )
        elif rows_change < 0:
            parts.append(f"{abs(rows_change)} rows added")

        if cols_change > 0:
            parts.append(
                f"{cols_change} columns removed ({self.get_columns_reduction_percentage():.1f}%)"
            )
        elif cols_change < 0:
            parts.append(f"{abs(cols_change)} columns added")

        return ", ".join(parts)

    def get_performance_category(self) -> str:
        """Categorize processing performance based on execution time."""
        if self.execution_time_seconds < 1.0:
            return "Fast"
        elif self.execution_time_seconds < 10.0:
            return "Normal"
        elif self.execution_time_seconds < 60.0:
            return "Slow"
        else:
            return "Very Slow"

    def has_output_file(self) -> bool:
        """Check if processing generated an output file."""
        return self.output_file_path is not None

    def get_issue_count(self) -> int:
        """Get total count of warnings and errors."""
        return len(self.warnings) + len(self.errors)

    def get_processing_summary(self) -> str:
        """Get comprehensive processing summary."""
        status = "SUCCESS" if self.is_successful() else "FAILED"
        data_summary = self.get_data_reduction_summary()
        performance = self.get_performance_category()
        issues = self.get_issue_count()

        summary = f"Status: {status}, Performance: {performance} ({self.execution_time_seconds:.2f}s)"
        if data_summary != "No data reduction":
            summary += f", Data changes: {data_summary}"
        if issues > 0:
            summary += f", Issues: {issues} ({len(self.warnings)} warnings, {len(self.errors)} errors)"

        return summary
