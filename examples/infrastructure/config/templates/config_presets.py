"""Configuration presets and templates for common scenarios."""

from infrastructure.config.algorithms.extraction_config import ExtractionConfig
from infrastructure.config.algorithms.scraping_config import ScrapingConfig
from infrastructure.config.providers.llm_provider_config import LLMProviderConfig
from infrastructure.config.templates.preset_catalog import PresetCatalog
from infrastructure.serialization.protocol_serializer import ProtocolSerializer


class ConfigPresets:
    """Predefined configuration presets for common use cases."""

    @staticmethod
    def get_llm_preset(preset_name: str) -> LLMProviderConfig:
        """Get LLM configuration preset.

        Args:
            preset_name: Name of the preset.

        Returns:
            LLM configuration object.

        Raises:
            ValueError: If preset name is not found.
        """
        presets = ProtocolSerializer.serialize_llm_presets(
            {
                "development_ollama": {
                    "provider_type": "ollama",
                    "model_name": "llama2",
                    "api_endpoint": "http://localhost:11434",
                    "api_key": "",
                    "timeout_seconds": 60,
                    "max_retries": 2,
                    "max_tokens": 4096,
                    "temperature": 0.7,
                },
                "production_openai": {
                    "provider_type": "openai",
                    "model_name": "gpt-4",
                    "api_endpoint": "https://api.openai.com/v1",
                    "api_key": "${OPENAI_API_KEY}",  # Will be replaced from environment
                    "timeout_seconds": 30,
                    "max_retries": 3,
                    "max_tokens": 4096,
                    "temperature": 0.7,
                },
                "production_anthropic": {
                    "provider_type": "anthropic",
                    "model_name": "claude-3-sonnet-20240229",
                    "api_endpoint": "https://api.anthropic.com",
                    "api_key": "${ANTHROPIC_API_KEY}",  # Will be replaced from environment
                    "timeout_seconds": 30,
                    "max_retries": 3,
                    "max_tokens": 4096,
                    "temperature": 0.7,
                },
                "fast_inference": {
                    "provider_type": "openai",
                    "model_name": "gpt-3.5-turbo",
                    "api_endpoint": "https://api.openai.com/v1",
                    "api_key": "${OPENAI_API_KEY}",
                    "timeout_seconds": 15,
                    "max_retries": 2,
                    "max_tokens": 2048,
                    "temperature": 0.3,
                },
            }
        )

        if preset_name not in presets:
            available = ", ".join(presets.keys())
            raise ValueError(
                f"Unknown LLM preset: {preset_name}. Available: {available}"
            )

        return LLMProviderConfig.from_dict(presets[preset_name])

    @staticmethod
    def get_scraping_preset(preset_name: str) -> ScrapingConfig:
        """Get scraping configuration preset.

        Args:
            preset_name: Name of the preset.

        Returns:
            Dictionary with scraping configuration.

        Raises:
            ValueError: If preset name is not found.
        """
        presets = ProtocolSerializer.serialize_scraping_presets(
            {
                "respectful": {
                    "max_content_length": 50000,
                    "max_pages": 10,
                    "depth": "SINGLE_PAGE",
                    "content_type": "TEXT_WITH_LINKS",
                    "follow_external_links": False,
                    "respect_robots_txt": True,
                    "delay_between_requests": 3.0,
                    "max_retries": 2,
                    "timeout_seconds": 30,
                    "include_metadata": True,
                    "user_agent": "PyIntelCivil/1.0 (Respectful Mode)",
                },
                "fast": {
                    "max_content_length": 10000,
                    "max_pages": 5,
                    "depth": "SINGLE_PAGE",
                    "content_type": "TEXT_ONLY",
                    "follow_external_links": False,
                    "respect_robots_txt": True,
                    "delay_between_requests": 0.5,
                    "max_retries": 1,
                    "timeout_seconds": 15,
                    "include_metadata": False,
                    "user_agent": "PyIntelCivil/1.0 (Fast Mode)",
                },
                "comprehensive": {
                    "max_content_length": 200000,
                    "max_pages": 50,
                    "depth": "WITH_LINKS",
                    "content_type": "FULL_CONTENT",
                    "follow_external_links": False,
                    "respect_robots_txt": True,
                    "delay_between_requests": 2.0,
                    "max_retries": 3,
                    "timeout_seconds": 60,
                    "include_metadata": True,
                    "user_agent": "PyIntelCivil/1.0 (Comprehensive Mode)",
                },
                "research": {
                    "max_content_length": 100000,
                    "max_pages": 25,
                    "depth": "WITH_LINKS",
                    "content_type": "TEXT_WITH_LINKS",
                    "follow_external_links": False,
                    "respect_robots_txt": True,
                    "delay_between_requests": 1.5,
                    "max_retries": 3,
                    "timeout_seconds": 45,
                    "include_metadata": True,
                    "user_agent": "PyIntelCivil/1.0 (Research Mode)",
                },
            }
        )

        if preset_name not in presets:
            available = ", ".join(presets.keys())
            raise ValueError(
                f"Unknown scraping preset: {preset_name}. Available: {available}"
            )

        return ScrapingConfig.from_dict(presets[preset_name])

    @staticmethod
    def get_extraction_preset(preset_name: str) -> ExtractionConfig:
        """Get extraction configuration preset.

        Args:
            preset_name: Name of the preset.

        Returns:
            Dictionary with extraction configuration.

        Raises:
            ValueError: If preset name is not found.
        """
        presets = ProtocolSerializer.serialize_llm_presets(
            {
                "fast": {
                    "strategy": "markitdown",
                    "preserve_formatting": False,
                    "extract_images": False,
                    "extract_tables": False,
                    "extract_metadata": False,
                    "output_format": "markdown",
                    "max_file_size_mb": 50,
                    "timeout_seconds": 60,
                    "priority": "high",
                },
                "comprehensive": {
                    "strategy": "auto",
                    "preserve_formatting": True,
                    "extract_images": True,
                    "extract_tables": True,
                    "extract_metadata": True,
                    "output_format": "markdown",
                    "max_file_size_mb": 200,
                    "timeout_seconds": 600,
                    "priority": "normal",
                },
                "text_only": {
                    "strategy": "markitdown",
                    "preserve_formatting": True,
                    "extract_images": False,
                    "extract_tables": True,
                    "extract_metadata": True,
                    "output_format": "markdown",
                    "max_file_size_mb": 100,
                    "timeout_seconds": 300,
                    "priority": "normal",
                },
                "academic": {
                    "strategy": "docling",
                    "preserve_formatting": True,
                    "extract_images": True,
                    "extract_tables": True,
                    "extract_metadata": True,
                    "output_format": "markdown",
                    "max_file_size_mb": 500,
                    "timeout_seconds": 900,
                    "priority": "low",
                },
            }
        )

        if preset_name not in presets:
            available = ", ".join(presets.keys())
            raise ValueError(
                f"Unknown extraction preset: {preset_name}. Available: {available}"
            )

        return ExtractionConfig.from_dict(presets[preset_name])

    @staticmethod
    def get_all_presets() -> PresetCatalog:
        """Get all available presets by category.

        Returns:
            Type-safe preset catalog with all available presets.
        """
        return PresetCatalog.create_default()

    @staticmethod
    def create_config_from_preset(
        config_type: str, preset_name: str
    ) -> LLMProviderConfig | ScrapingConfig | ExtractionConfig:
        """Create configuration instance from preset.

        Args:
            config_type: Type of configuration (llm, scraping, extraction).
            preset_name: Name of the preset.

        Returns:
            Configuration instance.

        Raises:
            ValueError: If config type or preset name is invalid.
        """
        if config_type == "llm":
            return ConfigPresets.get_llm_preset(preset_name)
        elif config_type == "scraping":
            return ConfigPresets.get_scraping_preset(preset_name)
        elif config_type == "extraction":
            return ConfigPresets.get_extraction_preset(preset_name)
        else:
            raise ValueError(f"Unknown config type: {config_type}")
