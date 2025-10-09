from dataclasses import dataclass


@dataclass
class StructuralElement:
    """Elemento estructural del documento."""

    element_type: str
    text_content: str
    page_number: int = 0
    position: int = 0
    level: int = 0
