from docling.datamodel.base_models import InputFormat
from infrastructure.data.extractors.entities.document_format_mapping import (
    DocumentFormatMapping,
)
from infrastructure.data.extractors.enums.document_type import DocumentType


class DoclingFormatMapper:
    """Mapea tipos de documento a formatos de Docling."""

    def __init__(self):
        self._format_mappings: list[DocumentFormatMapping] = [
            DocumentFormatMapping(
                document_type=DocumentType.PDF, input_format=InputFormat.PDF
            ),
            DocumentFormatMapping(
                document_type=DocumentType.DOCX, input_format=InputFormat.DOCX
            ),
            DocumentFormatMapping(
                document_type=DocumentType.PPTX, input_format=InputFormat.PPTX
            ),
            DocumentFormatMapping(
                document_type=DocumentType.HTML, input_format=InputFormat.HTML
            ),
            DocumentFormatMapping(
                document_type=DocumentType.MD, input_format=InputFormat.MD
            ),
        ]

    def get_input_format(self, document_type: DocumentType) -> InputFormat | None:
        """Obtiene el formato de entrada de Docling para un tipo de documento."""
        for mapping in self._format_mappings:
            if mapping.document_type == document_type:
                return mapping.input_format
        return None

    def is_supported(self, document_type: DocumentType) -> bool:
        """Verifica si el tipo de documento es soportado."""
        return self.get_input_format(document_type) is not None
