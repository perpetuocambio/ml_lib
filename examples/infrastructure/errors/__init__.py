"""Exposes the custom exceptions for the infrastructure layer."""

from infrastructure.errors.configuration_error import ConfigurationError
from infrastructure.errors.database_error import DatabaseError
from infrastructure.errors.http_client_error import HttpClientError
from infrastructure.errors.infrastructure_error import InfrastructureError
from infrastructure.errors.llm_provider_error import LLMProviderError
from infrastructure.errors.processing_error import ProcessingError

__all__ = [
    "ConfigurationError",
    "DatabaseError",
    "HttpClientError",
    "InfrastructureError",
    "LLMProviderError",
    "ProcessingError",
]
