"""Configuration template container."""

from dataclasses import dataclass

from infrastructure.config.algorithms.extraction_config import ExtractionConfig
from infrastructure.config.algorithms.scraping_config import ScrapingConfig
from infrastructure.config.providers.llm_provider_config import LLMProviderConfig


@dataclass(frozen=True)
class ConfigurationTemplate:
    """Type-safe configuration template - replaces dict with typed classes."""

    llm_config: LLMProviderConfig
    scraping_config: ScrapingConfig
    extraction_config: ExtractionConfig
