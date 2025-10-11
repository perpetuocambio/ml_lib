"""
Factory para crear instancias de OllamaProvider.
"""

from ml_lib.llm.config.llm_provider_config import LLMProviderConfig
from ml_lib.llm.entities.llm_provider_type import LLMProviderType
from ml_lib.llm.providers.ollama_provider import OllamaProvider


class OllamaFactory:
    """Factory para crear instancias configuradas de OllamaProvider."""

    @staticmethod
    def create_provider(
        model_name: str = "llava:latest",
        api_endpoint: str = "http://localhost:11434",
        timeout_seconds: int = 30,
    ) -> OllamaProvider:
        """
        Crea un proveedor Ollama configurado.

        Args:
            model_name: Nombre del modelo Ollama (ej. "llava:latest", "qwen2.5vl:latest")
            api_endpoint: URL del endpoint de Ollama
            timeout_seconds: Timeout para las requests

        Returns:
            OllamaProvider: Proveedor configurado
        """
        configuration = LLMProviderConfig(
            provider_type=LLMProviderType.OLLAMA,
            model_name=model_name,
            api_endpoint=api_endpoint,
            api_key="",  # Ollama no requiere API key
            timeout_seconds=timeout_seconds,
            max_retries=2,
        )

        return OllamaProvider(configuration)

    @staticmethod
    def create_vision_provider(model_name: str = "llava:latest") -> OllamaProvider:
        """
        Crea un proveedor Ollama optimizado para visión.

        Args:
            model_name: Nombre del modelo de visión

        Returns:
            OllamaProvider: Proveedor configurado para tareas de visión
        """
        return OllamaFactory.create_provider(
            model_name=model_name,
            timeout_seconds=45,  # Más tiempo para modelos de visión
        )
