"""
LLM Provider Configuration - Configuración agnóstica para proveedores LLM.

Este módulo define la configuración base para cualquier proveedor LLM (Ollama, OpenAI, etc).
Diseñado para ser usado por múltiples módulos de la biblioteca.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class LLMProviderType(Enum):
    """Tipos de proveedores LLM soportados."""

    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"


@dataclass
class LLMProviderConfig:
    """
    Configuración para proveedores LLM.

    Esta clase es inmutable y thread-safe. Diseñada para ser compartida
    entre múltiples módulos sin efectos secundarios.

    Ejemplos:
        >>> # Ollama local
        >>> config = LLMProviderConfig(
        ...     provider_type=LLMProviderType.OLLAMA,
        ...     model_name="dolphin3",
        ...     api_endpoint="http://localhost:11434"
        ... )

        >>> # OpenAI con API key desde variable de entorno
        >>> config = LLMProviderConfig.from_env(
        ...     provider_type=LLMProviderType.OPENAI,
        ...     model_name="gpt-4"
        ... )
    """

    # Identificación del proveedor
    provider_type: LLMProviderType
    """Tipo de proveedor LLM."""

    model_name: str
    """Nombre del modelo a utilizar (ej: 'dolphin3', 'gpt-4')."""

    # Endpoints y autenticación
    api_endpoint: Optional[str] = None
    """URL del endpoint API. None = usar default del proveedor."""

    api_key: Optional[str] = None
    """API key si es necesaria. NUNCA hardcodear - usar variables de entorno."""

    # Parámetros de generación por defecto
    default_temperature: float = 0.7
    """Temperatura por defecto para generación (0.0-2.0)."""

    default_max_tokens: int = 4000
    """Número máximo de tokens por defecto."""

    default_top_p: float = 0.9
    """Top-p sampling por defecto."""

    # Configuración de timeout y reintentos
    timeout_seconds: int = 120
    """Timeout para requests en segundos."""

    max_retries: int = 3
    """Número máximo de reintentos en caso de fallo."""

    # Configuración avanzada
    custom_headers: dict[str, str] = field(default_factory=dict)
    """Headers HTTP personalizados."""

    verify_ssl: bool = True
    """Verificar certificados SSL."""

    # Metadatos
    description: str = ""
    """Descripción opcional de esta configuración."""

    def __post_init__(self):
        """Validaciones post-inicialización."""
        # Validar temperatura
        if not 0.0 <= self.default_temperature <= 2.0:
            raise ValueError(f"Temperature debe estar entre 0.0 y 2.0, got {self.default_temperature}")

        # Validar max_tokens
        if self.default_max_tokens <= 0:
            raise ValueError(f"max_tokens debe ser positivo, got {self.default_max_tokens}")

        # Validar top_p
        if not 0.0 <= self.default_top_p <= 1.0:
            raise ValueError(f"top_p debe estar entre 0.0 y 1.0, got {self.default_top_p}")

        # Endpoint por defecto según proveedor
        if self.api_endpoint is None:
            self.api_endpoint = self._get_default_endpoint()

    def _get_default_endpoint(self) -> str:
        """Obtiene el endpoint por defecto según el tipo de proveedor."""
        defaults = {
            LLMProviderType.OLLAMA: "http://localhost:11434",
            LLMProviderType.OPENAI: "https://api.openai.com/v1",
            LLMProviderType.ANTHROPIC: "https://api.anthropic.com/v1",
        }
        return defaults.get(self.provider_type, "")

    @classmethod
    def from_env(
        cls,
        provider_type: LLMProviderType,
        model_name: str,
        api_key_env_var: str = "LLM_API_KEY",
        endpoint_env_var: Optional[str] = None,
        **kwargs
    ) -> "LLMProviderConfig":
        """
        Crea configuración leyendo credenciales desde variables de entorno.

        Este es el método recomendado para producción - NUNCA hardcodear
        API keys en el código.

        Args:
            provider_type: Tipo de proveedor
            model_name: Nombre del modelo
            api_key_env_var: Nombre de la variable de entorno con la API key
            endpoint_env_var: Nombre de la variable de entorno con el endpoint (opcional)
            **kwargs: Otros parámetros para LLMProviderConfig

        Returns:
            LLMProviderConfig con credenciales desde variables de entorno

        Ejemplo:
            >>> # En producción, el cliente configura:
            >>> # export LLM_API_KEY="sk-..."
            >>> # export LLM_ENDPOINT="https://custom-endpoint.com"
            >>>
            >>> config = LLMProviderConfig.from_env(
            ...     provider_type=LLMProviderType.OPENAI,
            ...     model_name="gpt-4",
            ...     api_key_env_var="LLM_API_KEY"
            ... )
        """
        import os

        # Leer API key desde variable de entorno
        api_key = os.getenv(api_key_env_var)
        if api_key is None and provider_type != LLMProviderType.OLLAMA:
            raise ValueError(
                f"Variable de entorno '{api_key_env_var}' no encontrada. "
                f"El cliente debe configurar: export {api_key_env_var}=<tu-api-key>"
            )

        # Leer endpoint desde variable de entorno (opcional)
        api_endpoint = None
        if endpoint_env_var:
            api_endpoint = os.getenv(endpoint_env_var)

        return cls(
            provider_type=provider_type,
            model_name=model_name,
            api_key=api_key,
            api_endpoint=api_endpoint,
            **kwargs
        )

    @classmethod
    def for_ollama(
        cls,
        model_name: str = "dolphin3",
        endpoint: str = "http://localhost:11434",
        **kwargs
    ) -> "LLMProviderConfig":
        """
        Crea configuración para Ollama local.

        Ollama no requiere API key ya que corre localmente.

        Args:
            model_name: Modelo Ollama a usar
            endpoint: URL del servidor Ollama
            **kwargs: Otros parámetros

        Returns:
            LLMProviderConfig configurado para Ollama

        Ejemplo:
            >>> config = LLMProviderConfig.for_ollama("dolphin3")
        """
        return cls(
            provider_type=LLMProviderType.OLLAMA,
            model_name=model_name,
            api_endpoint=endpoint,
            api_key=None,  # Ollama no necesita API key
            **kwargs
        )

    @classmethod
    def for_openai(
        cls,
        model_name: str = "gpt-4",
        api_key_env_var: str = "OPENAI_API_KEY",
        **kwargs
    ) -> "LLMProviderConfig":
        """
        Crea configuración para OpenAI.

        Args:
            model_name: Modelo OpenAI a usar
            api_key_env_var: Variable de entorno con la API key
            **kwargs: Otros parámetros

        Returns:
            LLMProviderConfig configurado para OpenAI

        Ejemplo:
            >>> # El cliente debe configurar: export OPENAI_API_KEY="sk-..."
            >>> config = LLMProviderConfig.for_openai("gpt-4")
        """
        return cls.from_env(
            provider_type=LLMProviderType.OPENAI,
            model_name=model_name,
            api_key_env_var=api_key_env_var,
            **kwargs
        )

    def with_model(self, model_name: str) -> "LLMProviderConfig":
        """
        Crea una nueva configuración con un modelo diferente.

        Args:
            model_name: Nuevo nombre de modelo

        Returns:
            Nueva instancia de LLMProviderConfig

        Ejemplo:
            >>> base_config = LLMProviderConfig.for_ollama("dolphin3")
            >>> llama_config = base_config.with_model("llama3.2")
        """
        from copy import copy
        new_config = copy(self)
        new_config.model_name = model_name
        return new_config

    def to_dict(self) -> dict:
        """
        Convierte a diccionario (útil para logging/debug).

        IMPORTANTE: Excluye api_key por seguridad.

        Returns:
            Dict con configuración (sin credenciales sensibles)
        """
        return {
            "provider_type": self.provider_type.value,
            "model_name": self.model_name,
            "api_endpoint": self.api_endpoint,
            "api_key": "***" if self.api_key else None,  # Ocultar API key
            "default_temperature": self.default_temperature,
            "default_max_tokens": self.default_max_tokens,
            "default_top_p": self.default_top_p,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "verify_ssl": self.verify_ssl,
            "description": self.description,
        }
