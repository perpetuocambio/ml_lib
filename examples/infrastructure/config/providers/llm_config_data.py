"""LLM configuration data types."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfigData:
    """Type-safe container for LLM configuration data - replaces dict with typed classes."""

    provider_type: str
    model_name: str
    api_endpoint: str
    api_key: str
    timeout_seconds: int
    max_retries: int
    max_tokens: int
    temperature: float

    def mask_sensitive_data(self) -> "LLMConfigData":
        """Return copy with sensitive data masked."""
        return LLMConfigData(
            provider_type=self.provider_type,
            model_name=self.model_name,
            api_endpoint=self.api_endpoint,
            api_key="***",
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
