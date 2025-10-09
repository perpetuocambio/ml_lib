from enum import Enum


class OutputFormat(Enum):
    """Formatos de salida disponibles."""

    MARKDOWN = "markdown"
    TEXT = "text"
    HTML = "html"
    PLAIN = "plain"
