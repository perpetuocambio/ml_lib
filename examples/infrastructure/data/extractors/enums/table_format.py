"""
Formatos de tabla disponibles para markdown.
"""

from enum import Enum


class TableFormat(Enum):
    """Formatos de tabla soportados en markdown."""

    PIPE = "pipe"  # Formato estándar con | (más compatible)
    GRID = "grid"  # Formato con líneas de grid
    SIMPLE = "simple"  # Formato simple sin bordes
    HTML = "html"  # Tablas en formato HTML dentro del markdown
