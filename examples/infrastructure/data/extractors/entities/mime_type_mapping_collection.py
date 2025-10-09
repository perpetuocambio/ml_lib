"""MIME type mapping collection - replaces dict mappings with typed classes."""

from infrastructure.data.extractors.entities.mime_type_mapping import (
    MimeTypeMapping,
)
from infrastructure.data.extractors.enums.document_type import DocumentType


class MimeTypeMappingCollection:
    """Collection of MIME type mappings - replaces dict mappings with typed classes."""

    _MAPPINGS = [
        MimeTypeMapping("application/pdf", DocumentType.PDF),
        MimeTypeMapping(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            DocumentType.DOCX,
        ),
        MimeTypeMapping("application/msword", DocumentType.DOC),
        MimeTypeMapping(
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            DocumentType.PPTX,
        ),
        MimeTypeMapping("application/vnd.ms-powerpoint", DocumentType.PPT),
        MimeTypeMapping(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            DocumentType.XLSX,
        ),
        MimeTypeMapping("application/vnd.ms-excel", DocumentType.XLS),
        MimeTypeMapping("text/html", DocumentType.HTML),
        MimeTypeMapping("text/plain", DocumentType.TXT),
        MimeTypeMapping("text/markdown", DocumentType.MD),
        MimeTypeMapping("application/rtf", DocumentType.RTF),
        MimeTypeMapping("application/vnd.oasis.opendocument.text", DocumentType.ODT),
        MimeTypeMapping(
            "application/vnd.oasis.opendocument.presentation", DocumentType.ODP
        ),
        MimeTypeMapping(
            "application/vnd.oasis.opendocument.spreadsheet", DocumentType.ODS
        ),
    ]

    @classmethod
    def get_type_by_mime_type(cls, mime_type: str) -> DocumentType | None:
        """Get document type by MIME type."""
        for mapping in cls._MAPPINGS:
            if mapping.mime_type == mime_type:
                return mapping.document_type
        return None

    @classmethod
    def get_supported_mime_types(cls) -> list[str]:
        """Get list of supported MIME types."""
        return [mapping.mime_type for mapping in cls._MAPPINGS]
