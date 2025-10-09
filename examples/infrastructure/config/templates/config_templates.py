"""Configuration templates for generating new configurations."""

from infrastructure.config.templates.config_presets import ConfigPresets
from infrastructure.config.types.configuration_template import ConfigurationTemplate


class ConfigTemplates:
    """Configuration templates for generating new configurations."""

    @staticmethod
    def generate_development_template() -> ConfigurationTemplate:
        """Generate complete development environment template.

        Returns:
            Configuration template for development environment.
        """
        return ConfigurationTemplate(
            llm_config=ConfigPresets.get_llm_preset("development_ollama"),
            scraping_config=ConfigPresets.get_scraping_preset("fast"),
            extraction_config=ConfigPresets.get_extraction_preset("fast"),
        )

    @staticmethod
    def generate_production_template() -> ConfigurationTemplate:
        """Generate complete production environment template.

        Returns:
            Configuration template for production environment.
        """
        return ConfigurationTemplate(
            llm_config=ConfigPresets.get_llm_preset("production_openai"),
            scraping_config=ConfigPresets.get_scraping_preset("respectful"),
            extraction_config=ConfigPresets.get_extraction_preset("comprehensive"),
        )

    @staticmethod
    def generate_research_template() -> ConfigurationTemplate:
        """Generate template optimized for research workflows.

        Returns:
            Configuration template for research workflows.
        """
        return ConfigurationTemplate(
            llm_config=ConfigPresets.get_llm_preset("production_anthropic"),
            scraping_config=ConfigPresets.get_scraping_preset("research"),
            extraction_config=ConfigPresets.get_extraction_preset("academic"),
        )

    @staticmethod
    def generate_custom_template(
        llm_preset: str,
        scraping_preset: str,
        extraction_preset: str,
    ) -> ConfigurationTemplate:
        """Generate custom template from presets.

        Args:
            llm_preset: LLM preset name.
            scraping_preset: Scraping preset name.
            extraction_preset: Extraction preset name.

        Returns:
            Custom configuration template.
        """
        return ConfigurationTemplate(
            llm_config=ConfigPresets.get_llm_preset(llm_preset),
            scraping_config=ConfigPresets.get_scraping_preset(scraping_preset),
            extraction_config=ConfigPresets.get_extraction_preset(extraction_preset),
        )
