from dataclasses import dataclass

from docling.datamodel.base_models import InputFormat
from infrastructure.data.extractors.enums.document_type import DocumentType


@dataclass
class DocumentFormatMapping:
    """Mapping between internal document types and Docling input formats."""

    document_type: DocumentType
    input_format: InputFormat
