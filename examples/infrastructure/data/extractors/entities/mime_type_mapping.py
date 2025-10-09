"""MIME type to document type mapping entity."""

from dataclasses import dataclass

from infrastructure.data.extractors.enums.document_type import DocumentType


@dataclass(frozen=True)
class MimeTypeMapping:
    """Individual MIME type to document type mapping."""

    mime_type: str
    document_type: DocumentType
