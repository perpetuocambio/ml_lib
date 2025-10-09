"""
Tipos de proveedores LLM soportados.
"""

from enum import Enum


class LLMProviderType(Enum):
    """Tipos de proveedores LLM soportados."""

    OLLAMA = "ollama"
    AZURE_OPENAI = "azure_openai"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL_MODEL = "local_model"
