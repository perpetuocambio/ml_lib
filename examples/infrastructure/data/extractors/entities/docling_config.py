"""
Configuración específica para el extractor Docling.
"""

from dataclasses import dataclass

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from infrastructure.data.extractors.entities.markdown_export_options import (
    MarkdownExportOptions,
)
from infrastructure.serialization.protocol_serializer import ProtocolSerializer


@dataclass
class DoclingConfig:
    """Configuración específica para Docling."""

    input_format: InputFormat | None = None
    pdf_options: PdfPipelineOptions | None = None
    ocr_enabled: bool = True
    table_structure_enabled: bool = True
    cell_matching_enabled: bool = True
    force_full_page_ocr: bool = False
    markdown_options: MarkdownExportOptions | None = None

    @classmethod
    def create_default(cls) -> "DoclingConfig":
        """Crea una configuración por defecto optimizada para markdown."""
        pdf_options = PdfPipelineOptions()
        pdf_options.do_ocr = True
        pdf_options.do_table_structure = True

        # Configure table structure options properly
        if hasattr(pdf_options, "table_structure_options"):
            pdf_options.table_structure_options.do_cell_matching = True

        return cls(
            pdf_options=pdf_options,
            ocr_enabled=True,
            table_structure_enabled=True,
            cell_matching_enabled=True,
            force_full_page_ocr=False,
            markdown_options=MarkdownExportOptions.create_default(),
        )

    @classmethod
    def create_fast_mode(cls) -> "DoclingConfig":
        """Crea una configuración optimizada para velocidad."""
        pdf_options = PdfPipelineOptions()
        pdf_options.do_ocr = False
        pdf_options.do_table_structure = False

        return cls(
            pdf_options=pdf_options,
            ocr_enabled=False,
            table_structure_enabled=False,
            cell_matching_enabled=False,
            force_full_page_ocr=False,
            markdown_options=MarkdownExportOptions.create_minimal(),
        )

    @classmethod
    def create_high_accuracy_mode(cls) -> "DoclingConfig":
        """Crea una configuración optimizada para máxima precisión en markdown."""
        pdf_options = PdfPipelineOptions()
        pdf_options.do_ocr = True
        pdf_options.do_table_structure = True

        # Configure table structure options for maximum accuracy
        if hasattr(pdf_options, "table_structure_options"):
            pdf_options.table_structure_options.do_cell_matching = True

        return cls(
            pdf_options=pdf_options,
            ocr_enabled=True,
            table_structure_enabled=True,
            cell_matching_enabled=True,
            force_full_page_ocr=True,
            markdown_options=MarkdownExportOptions.create_comprehensive(),
        )

    @classmethod
    def create_for_file_extensions(cls, extensions: set[str]) -> "DoclingConfig":
        """Crea configuración optimizada según las extensiones de archivo soportadas."""
        # For image files, enable OCR
        image_extensions = ProtocolSerializer.serialize_extension_set(
            {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".svg"}
        )
        needs_ocr = bool(extensions.intersection(image_extensions))

        # For office files, enable table structure
        office_extensions = ProtocolSerializer.serialize_extension_set(
            {
                ".pdf",
                ".docx",
                ".doc",
                ".pptx",
                ".ppt",
                ".xlsx",
                ".xls",
                ".odt",
                ".odp",
                ".ods",
            }
        )
        needs_tables = bool(extensions.intersection(office_extensions))

        pdf_options = PdfPipelineOptions()
        pdf_options.do_ocr = needs_ocr
        pdf_options.do_table_structure = needs_tables

        if needs_tables and hasattr(pdf_options, "table_structure_options"):
            pdf_options.table_structure_options.do_cell_matching = True

        # Choose appropriate markdown options based on file types
        if needs_tables:
            markdown_options = MarkdownExportOptions.create_comprehensive()
        else:
            markdown_options = MarkdownExportOptions.create_default()

        return cls(
            pdf_options=pdf_options,
            ocr_enabled=needs_ocr,
            table_structure_enabled=needs_tables,
            cell_matching_enabled=needs_tables,
            force_full_page_ocr=False,
            markdown_options=markdown_options,
        )
