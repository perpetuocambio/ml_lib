"""LLM provider configuration."""

from __future__ import annotations

from dataclasses import dataclass

from infrastructure.config.base.base_config import BaseInfrastructureConfig
from infrastructure.config.base.config_loader import ConfigLoader
from infrastructure.config.base.config_validator import ConfigValidator
from infrastructure.config.providers.llm_config_data import LLMConfigData
from infrastructure.providers.llm.entities.llm_provider_type import LLMProviderType


@dataclass(frozen=True)
class LLMProviderConfig(BaseInfrastructureConfig):
    """Configuration for LLM providers.

    Centralizes all LLM provider configuration with proper validation
    and environment loading capabilities.
    """

    provider_type: LLMProviderType
    model_name: str
    api_endpoint: str
    api_key: str
    timeout_seconds: int = 30
    max_retries: int = 3
    max_tokens: int = 4096
    temperature: float = 0.7

    @classmethod
    def from_environment(cls) -> LLMProviderConfig:
        """Load LLM configuration from environment variables.

        Environment variables:
            LLM_PROVIDER_TYPE: Provider type (openai, anthropic, ollama, etc.)
            LLM_MODEL_NAME: Model name
            LLM_API_ENDPOINT: API endpoint URL
            LLM_API_KEY: API key for authentication
            LLM_TIMEOUT_SECONDS: Request timeout (default: 30)
            LLM_MAX_RETRIES: Maximum retry attempts (default: 3)
            LLM_MAX_TOKENS: Maximum tokens per request (default: 4096)
            LLM_TEMPERATURE: Sampling temperature (default: 0.7)

        Returns:
            Configured LLM provider instance.
        """
        provider_type_str = ConfigLoader.get_env_var("LLM_PROVIDER_TYPE")
        provider_type = LLMProviderType(provider_type_str.lower())

        return cls(
            provider_type=provider_type,
            model_name=ConfigLoader.get_env_var("LLM_MODEL_NAME"),
            api_endpoint=ConfigLoader.get_env_var("LLM_API_ENDPOINT"),
            api_key=ConfigLoader.get_env_var("LLM_API_KEY"),
            timeout_seconds=ConfigLoader.get_env_int(
                "LLM_TIMEOUT_SECONDS", default=30, required=False
            ),
            max_retries=ConfigLoader.get_env_int(
                "LLM_MAX_RETRIES", default=3, required=False
            ),
            max_tokens=ConfigLoader.get_env_int(
                "LLM_MAX_TOKENS", default=4096, required=False
            ),
            temperature=float(
                ConfigLoader.get_env_var(
                    "LLM_TEMPERATURE", default="0.7", required=False
                )
            ),
        )

    @classmethod
    def from_config_data(cls, data: LLMConfigData) -> LLMProviderConfig:
        """Load configuration from dictionary.

        Args:
            data: Configuration data.

        Returns:
            Configured LLM provider instance.
        """
        provider_type = LLMProviderType(data.provider_type)

        return cls(
            provider_type=provider_type,
            model_name=data.model_name,
            api_endpoint=data.api_endpoint,
            api_key=data.api_key,
            timeout_seconds=data.timeout_seconds,
            max_retries=data.max_retries,
            max_tokens=data.max_tokens,
            temperature=data.temperature,
        )

    def validate(self) -> list[str]:
        """Validate LLM configuration.

        Returns:
            List of validation errors.
        """
        errors = []

        # Validate provider type
        errors.extend(
            ConfigValidator.validate_enum_value(
                self.provider_type.value, LLMProviderType, "Provider type"
            )
        )

        # Validate model name
        errors.extend(ConfigValidator.validate_model_name(self.model_name))

        # Validate API endpoint
        errors.extend(ConfigValidator.validate_url(self.api_endpoint, "API endpoint"))

        # Validate API key (unless local provider)
        if not self.is_local_provider():
            errors.extend(ConfigValidator.validate_api_key(self.api_key))

        # Validate timeout
        errors.extend(ConfigValidator.validate_timeout(self.timeout_seconds))

        # Validate max_retries
        if self.max_retries < 0 or self.max_retries > 10:
            errors.append("Max retries must be between 0 and 10")

        # Validate max_tokens
        if self.max_tokens <= 0 or self.max_tokens > 100000:
            errors.append("Max tokens must be between 1 and 100000")

        # Validate temperature
        if self.temperature < 0.0 or self.temperature > 2.0:
            errors.append("Temperature must be between 0.0 and 2.0")

        return errors

    def to_config_data(self) -> LLMConfigData:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation.
        """
        return LLMConfigData(
            provider_type=self.provider_type.value,
            model_name=self.model_name,
            api_endpoint=self.api_endpoint,
            api_key=self.api_key,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

    def is_local_provider(self) -> bool:
        """Check if this is a local provider.

        Returns:
            True if provider runs locally.
        """
        return self.provider_type in (
            LLMProviderType.OLLAMA,
            LLMProviderType.LOCAL_MODEL,
        )

    def get_safe_dict(self) -> LLMConfigData:
        """Get dictionary representation with sensitive data masked.

        Returns:
            Safe dictionary with API key masked.
        """
        return self.to_config_data().mask_sensitive_data()

    @classmethod
    def create_ollama_config(
        cls, model_name: str = "llama2", host: str = "localhost", port: int = 11434
    ) -> LLMProviderConfig:
        """Create configuration for Ollama local provider.

        Args:
            model_name: Ollama model name.
            host: Ollama host.
            port: Ollama port.

        Returns:
            Configured Ollama instance.
        """
        return cls(
            provider_type=LLMProviderType.OLLAMA,
            model_name=model_name,
            api_endpoint=f"http://{host}:{port}",
            api_key="",  # Ollama doesn't require API key
            timeout_seconds=60,  # Local models may be slower
            max_retries=2,
            max_tokens=4096,
            temperature=0.7,
        )

    @classmethod
    def create_openai_config(
        cls, api_key: str, model_name: str = "gpt-4"
    ) -> LLMProviderConfig:
        """Create configuration for OpenAI provider.

        Args:
            api_key: OpenAI API key.
            model_name: OpenAI model name.

        Returns:
            Configured OpenAI instance.
        """
        return cls(
            provider_type=LLMProviderType.OPENAI,
            model_name=model_name,
            api_endpoint="https://api.openai.com/v1",
            api_key=api_key,
            timeout_seconds=30,
            max_retries=3,
            max_tokens=4096,
            temperature=0.7,
        )
