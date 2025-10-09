import datetime
from dataclasses import dataclass, field
from pathlib import Path

from infrastructure.data.extractors.enums.document_type import DocumentType


@dataclass
class DocumentMetadata:
    """Metadatos extraídos del documento."""

    file_path: Path
    file_size: int
    document_type: DocumentType
    title: str | None = None
    author: str | None = None
    creator: str | None = None
    subject: str | None = None
    keywords: list[str] = field(default_factory=list)
    creation_date: datetime.datetime | None = None
    modification_date: datetime.datetime | None = None
    page_count: int | None = None
    word_count: int | None = None
    language: str | None = None
    has_custom_metadata: bool = False
    custom_metadata_preview: str = ""
