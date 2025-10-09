"""Query result value object for database operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from infrastructure.serialization.protocol_serializer import ProtocolSerializer

# Type-safe union for database values (eliminates generic dict)
DatabaseValue = str | int | float | bool | None
DatabaseRow = ProtocolSerializer.DatabaseRowType

T = TypeVar("T")


@dataclass(frozen=True)
class QueryResult:
    """Type-safe single query result."""

    data: DatabaseRow

    def __post_init__(self) -> None:
        """Validate query result data."""
        if not ProtocolSerializer.is_valid_database_row(self.data):
            raise ValueError("Query result data must be a valid database row")

    @classmethod
    def create(cls, data: DatabaseRow) -> QueryResult:
        """Create query result from database row."""
        return cls(data=data)

    def get(self, key: str) -> str | int | float | bool | None:
        """Get value by key."""
        return self.data.get(key)

    def convert_to(self, target_type: type[T]) -> T:
        """Convert query result to specific typed object."""
        # If target type is a dataclass, use field mapping
        if hasattr(target_type, "__dataclass_fields__"):
            return self._convert_to_dataclass(target_type)

        # If target type has from_database_row method
        if hasattr(target_type, "from_database_row"):
            return target_type.from_database_row(self.data)

        raise ValueError(f"Cannot convert QueryResult to {target_type}")

    def _convert_to_dataclass(self, target_type: type[T]) -> T:
        """Convert to dataclass using field mapping."""
        kwargs = {}
        for field_name, _field_info in target_type.__dataclass_fields__.items():
            if field_name in self.data:
                kwargs[field_name] = self.data[field_name]
        return target_type(**kwargs)

    def get_all_fields(self) -> list[tuple[str, DatabaseValue]]:
        """Get all fields as type-safe tuples."""
        return ProtocolSerializer.get_database_row_items(self.data)

    def get_field_names(self) -> list[str]:
        """Get all field names."""
        return ProtocolSerializer.get_database_row_keys(self.data)

    def get_field_values(self) -> list[DatabaseValue]:
        """Get all field values."""
        return ProtocolSerializer.get_database_row_values(self.data)
