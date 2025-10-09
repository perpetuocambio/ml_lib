"""
Categorías de prompts según su función.
"""

from enum import Enum


class PromptCategory(Enum):
    """Categorías de prompts según su función."""

    SYSTEM = "system"  # Instrucciones de sistema/comportamiento
    USER = "user"  # Entrada del usuario/datos
    ASSISTANT = "assistant"  # Respuesta esperada del asistente
    FUNCTION = "function"  # Llamada a función específica
