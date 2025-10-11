"""
Interfaz abstracta para proveedores de LLM.
"""

from abc import ABC, abstractmethod

from ml_lib.llm.config.llm_provider_config import LLMProviderConfig
from ml_lib.llm.entities.llm_prompt import LLMPrompt
from ml_lib.llm.entities.llm_provider_type import LLMProviderType
from ml_lib.llm.entities.llm_response import LLMResponse


class LLMProvider(ABC):
    """
    Interfaz abstracta para proveedores de LLM.

    Abstrae las diferentes implementaciones (Ollama, Azure OpenAI, etc.).
    """

    def __init__(self, configuration: LLMProviderConfig):
        self.configuration = configuration

    @abstractmethod
    def generate_response(self, prompt: LLMPrompt) -> LLMResponse:
        """
        Genera una respuesta usando el LLM.

        Args:
            prompt: Prompt estructurado para el LLM

        Returns:
            LLMResponse: Respuesta estructurada del LLM
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """
        Verifica si el proveedor está disponible.

        Returns:
            bool: True si el proveedor puede ser utilizado
        """
        ...

    @abstractmethod
    def get_supported_models(self) -> list[str]:
        """
        Obtiene la lista de modelos soportados.

        Returns:
            list[str]: Lista de nombres de modelos disponibles
        """
        ...

    def get_provider_type(self) -> LLMProviderType:
        """Obtiene el tipo de proveedor."""
        return self.configuration.provider_type
