from dataclasses import dataclass, field

from infrastructure.data.extractors.entities.structural_element import (
    StructuralElement,
)


@dataclass
class DocumentStructure:
    """Estructura del documento extraído."""

    line_count: int = 0
    word_count: int = 0
    character_count: int = 0
    paragraph_count: int = 0
    header_count: int = 0
    list_count: int = 0
    code_block_count: int = 0
    link_count: int = 0
    bold_text_count: int = 0
    italic_text_count: int = 0
    elements: list[StructuralElement] = field(default_factory=list)
