"""
Estilos de bloques de código disponibles para markdown.
"""

from enum import Enum


class CodeBlockStyle(Enum):
    """Estilos de bloques de código soportados en markdown."""

    FENCED = "fenced"  # Bloques de código con ``` (más legible)
    INDENTED = "indented"  # Bloques de código con indentación de 4 espacios
