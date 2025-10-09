"""Extension mapping collection - replaces dict mappings with typed classes."""

from infrastructure.data.extractors.entities.extension_mapping import (
    ExtensionMapping,
)
from infrastructure.data.extractors.enums.document_type import DocumentType


class ExtensionMappingCollection:
    """Collection of extension mappings - replaces dict mappings with typed classes."""

    _MAPPINGS = [
        ExtensionMapping(".pdf", DocumentType.PDF),
        ExtensionMapping(".docx", DocumentType.DOCX),
        ExtensionMapping(".doc", DocumentType.DOC),
        ExtensionMapping(".pptx", DocumentType.PPTX),
        ExtensionMapping(".ppt", DocumentType.PPT),
        ExtensionMapping(".xlsx", DocumentType.XLSX),
        ExtensionMapping(".xls", DocumentType.XLS),
        ExtensionMapping(".html", DocumentType.HTML),
        ExtensionMapping(".htm", DocumentType.HTML),
        ExtensionMapping(".txt", DocumentType.TXT),
        ExtensionMapping(".md", DocumentType.MD),
        ExtensionMapping(".rtf", DocumentType.RTF),
        ExtensionMapping(".odt", DocumentType.ODT),
        ExtensionMapping(".odp", DocumentType.ODP),
        ExtensionMapping(".ods", DocumentType.ODS),
    ]

    @classmethod
    def get_type_by_extension(cls, extension: str) -> DocumentType | None:
        """Get document type by file extension."""
        normalized_ext = extension.lower()
        for mapping in cls._MAPPINGS:
            if mapping.extension == normalized_ext:
                return mapping.document_type
        return None

    @classmethod
    def get_supported_extensions(cls) -> list[str]:
        """Get list of supported extensions."""
        return [mapping.extension for mapping in cls._MAPPINGS]
