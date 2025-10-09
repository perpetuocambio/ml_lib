"""Extension to document type mapping entity."""

from dataclasses import dataclass

from infrastructure.data.extractors.enums.document_type import DocumentType


@dataclass(frozen=True)
class ExtensionMapping:
    """Individual extension to document type mapping."""

    extension: str
    document_type: DocumentType
