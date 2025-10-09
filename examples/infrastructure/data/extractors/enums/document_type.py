from enum import Enum


class DocumentType(Enum):
    """Tipos de documentos soportados."""

    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    PPTX = "pptx"
    PPT = "ppt"
    XLSX = "xlsx"
    XLS = "xls"
    HTML = "html"
    TXT = "txt"
    MD = "md"
    RTF = "rtf"
    ODT = "odt"
    ODP = "odp"
    ODS = "ods"
    UNKNOWN = "unknown"
