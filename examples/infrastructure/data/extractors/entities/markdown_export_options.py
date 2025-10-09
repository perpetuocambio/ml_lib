"""
Opciones específicas para la exportación a markdown.
"""

from dataclasses import dataclass


@dataclass
class MarkdownExportOptions:
    """Opciones de configuración para la exportación a markdown."""

    include_tables: bool = True
    include_images: bool = True
    include_metadata: bool = False
    table_format: str = "pipe"
    preserve_formatting: bool = True
    page_number: int | None = None

    @classmethod
    def create_default(cls) -> "MarkdownExportOptions":
        """Crea opciones por defecto para markdown."""
        return cls(
            include_tables=True,
            include_images=True,
            include_metadata=False,
            table_format="pipe",
            preserve_formatting=True,
            page_number=None,
        )

    @classmethod
    def create_minimal(cls) -> "MarkdownExportOptions":
        """Crea opciones mínimas para markdown (solo texto)."""
        return cls(
            include_tables=False,
            include_images=False,
            include_metadata=False,
            table_format="pipe",
            preserve_formatting=False,
            page_number=None,
        )

    @classmethod
    def create_comprehensive(cls) -> "MarkdownExportOptions":
        """Crea opciones completas para markdown (todo incluido)."""
        return cls(
            include_tables=True,
            include_images=True,
            include_metadata=True,
            table_format="pipe",
            preserve_formatting=True,
            page_number=None,
        )
