import re

from infrastructure.data.extractors.entities.document_structure import (
    DocumentStructure,
)
from infrastructure.data.extractors.handlers.markdown_parser import MarkdownParser
from infrastructure.data.extractors.handlers.text_cleaner import TextCleaner


class ContentCleaner:
    """Limpiador y procesador de contenido extraído."""

    def __init__(self):
        self.text_cleaner = TextCleaner()
        self.markdown_parser = MarkdownParser()

    def clean_text(self, text: str) -> str:
        """Limpia el texto extraído."""
        return self.text_cleaner.clean_text(text)

    def extract_structure(self, text: str) -> DocumentStructure:
        """Extrae información estructural del texto."""
        return self.markdown_parser.count_markdown_elements(text)

    def markdown_to_text(self, markdown_text: str) -> str:
        """Convierte markdown a texto plano simple."""
        if not markdown_text:
            return ""

        text = markdown_text

        # Remover encabezados markdown
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

        # Remover formato de texto (negrita, cursiva)
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # negrita
        text = re.sub(r"\*(.*?)\*", r"\1", text)  # cursiva
        text = re.sub(r"__(.*?)__", r"\1", text)  # negrita alternativa
        text = re.sub(r"_(.*?)_", r"\1", text)  # cursiva alternativa

        # Remover enlaces
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

        # Remover código inline
        text = re.sub(r"`([^`]+)`", r"\1", text)

        # Remover bloques de código
        text = re.sub(r"```[\s\S]*?```", "", text)

        # Limpiar listas
        text = re.sub(r"^[\s]*[-*+]\s+", "• ", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

        return self.text_cleaner.clean_text(text)
