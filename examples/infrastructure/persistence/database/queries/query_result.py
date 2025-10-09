"""Query result for database operations."""

from dataclasses import dataclass, field


@dataclass
class QueryResult:
    """Type-safe query result."""

    rows_affected: int = 0
    columns: list[str] = field(default_factory=list)
    data_rows: list[list[str]] = field(default_factory=list)
    success: bool = True
    error_message: str = ""

    def add_column(self, column_name: str) -> None:
        """Add column name."""
        self.columns.append(column_name)

    def add_data_row(self, row_data: list[str]) -> None:
        """Add data row."""
        self.data_rows.append(row_data)

    def get_row_count(self) -> int:
        """Get number of data rows."""
        return len(self.data_rows)

    def get_column_count(self) -> int:
        """Get number of columns."""
        return len(self.columns)

    def get_summary(self) -> str:
        """Get result summary."""
        if not self.success:
            return f"Error: {self.error_message}"

        return f"{self.get_row_count()} rows, {self.get_column_count()} columns"
