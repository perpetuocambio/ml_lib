from dataclasses import dataclass, field


@dataclass
class TableInfo:
    """Información de una tabla extraída."""

    index: int
    table_type: str
    rows: int
    columns: int
    start_line: int = 0
    headers: list[str] = field(default_factory=list)
    has_data: bool = False
    data_preview: str = ""
