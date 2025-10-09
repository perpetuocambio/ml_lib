"""
Document type mappings - consolidated facade without dict violations.

Provides unified interface for extension and MIME type mappings
while maintaining architectural purity with typed collections.
"""

from infrastructure.data.extractors.entities.extension_mapping_collection import (
    ExtensionMappingCollection,
)
from infrastructure.data.extractors.entities.mime_type_mapping_collection import (
    MimeTypeMappingCollection,
)
from infrastructure.data.extractors.enums.document_type import DocumentType


class DocumentTypeMappings:
    """Consolidated document type mappings facade - no dict violations."""

    @classmethod
    def get_type_by_extension(cls, extension: str) -> DocumentType | None:
        """Get document type by file extension."""
        return ExtensionMappingCollection.get_type_by_extension(extension)

    @classmethod
    def get_type_by_mime_type(cls, mime_type: str) -> DocumentType | None:
        """Get document type by MIME type."""
        return MimeTypeMappingCollection.get_type_by_mime_type(mime_type)

    @classmethod
    def get_supported_extensions(cls) -> list[str]:
        """Get list of supported extensions."""
        return ExtensionMappingCollection.get_supported_extensions()

    @classmethod
    def get_supported_mime_types(cls) -> list[str]:
        """Get list of supported MIME types."""
        return MimeTypeMappingCollection.get_supported_mime_types()
