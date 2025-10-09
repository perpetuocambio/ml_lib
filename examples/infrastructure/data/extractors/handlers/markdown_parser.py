import re

from infrastructure.data.extractors.entities.document_structure import (
    DocumentStructure,
)
from infrastructure.data.extractors.entities.structural_element import (
    StructuralElement,
)


class MarkdownParser:
    """Parser para extraer información de texto markdown."""

    @staticmethod
    def extract_headers(text: str) -> list[StructuralElement]:
        """Extrae encabezados del texto markdown."""
        headers = []
        lines = text.split("\n")

        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith("#"):
                level = len(line) - len(line.lstrip("#"))
                header_text = line.lstrip("#").strip()

                if header_text:  # Solo si hay texto después de los #
                    element = StructuralElement(
                        element_type="header",
                        text_content=header_text,
                        position=i + 1,
                        level=level,
                    )
                    headers.append(element)

        return headers

    @staticmethod
    def extract_lists(text: str) -> list[StructuralElement]:
        """Extrae listas del texto markdown."""
        lists = []
        lines = text.split("\n")

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if line_stripped.startswith(("- ", "* ", "+ ")) or (
                line_stripped and line_stripped[0].isdigit() and ". " in line_stripped
            ):
                element = StructuralElement(
                    element_type="list_item", text_content=line_stripped, position=i + 1
                )
                lists.append(element)

        return lists

    @staticmethod
    def count_markdown_elements(text: str) -> DocumentStructure:
        """Cuenta elementos específicos de markdown."""
        structure = DocumentStructure()

        # Contar elementos básicos
        lines = text.split("\n")
        structure.line_count = len(lines)
        structure.word_count = len(text.split())
        structure.character_count = len(text)
        structure.paragraph_count = len([line for line in lines if line.strip()])

        # Contar elementos específicos de markdown
        structure.code_block_count = len(re.findall(r"```[\s\S]*?```", text))
        structure.link_count = len(re.findall(r"\[([^\]]+)\]\([^)]+\)", text))
        structure.bold_text_count = len(re.findall(r"\*\*[^*]+\*\*", text))
        structure.italic_text_count = len(re.findall(r"\*[^*]+\*", text))

        # Extraer elementos estructurales
        headers = MarkdownParser.extract_headers(text)
        lists = MarkdownParser.extract_lists(text)

        structure.header_count = len(headers)
        structure.list_count = len(lists)
        structure.elements = headers + lists

        return structure
