"""Extraction algorithm configuration."""

from __future__ import annotations

from dataclasses import dataclass

from infrastructure.config.base.base_config import BaseInfrastructureConfig
from infrastructure.config.base.config_loader import ConfigLoader
from infrastructure.config.base.config_validator import ConfigValidator
from infrastructure.data.extractors.enums.extraction_strategy import (
    ExtractionStrategy,
)
from infrastructure.data.extractors.enums.output_format import OutputFormat
from infrastructure.data.extractors.enums.processing_priority import (
    ProcessingPriority,
)
from infrastructure.serialization.protocol_serializer import ProtocolSerializer


@dataclass(frozen=True)
class ExtractionConfig(BaseInfrastructureConfig):
    """Configuration for document extraction algorithms.

    Consolidates all extraction-related configurations including
    MarkItDown, Docling, and general extraction settings.
    """

    # General extraction settings
    strategy: ExtractionStrategy = ExtractionStrategy.AUTO
    preserve_formatting: bool = True
    extract_images: bool = False
    extract_tables: bool = True
    extract_metadata: bool = True
    output_format: OutputFormat = OutputFormat.MARKDOWN
    max_file_size_mb: int = 100
    timeout_seconds: int = 300
    priority: ProcessingPriority = ProcessingPriority.NORMAL

    # MarkItDown specific settings
    markitdown_enable_plugins: bool = False
    markitdown_docintel_endpoint: str | None = None
    markitdown_llm_model: str | None = None
    markitdown_llm_prompt: str | None = None
    markitdown_use_ollama: bool = False
    markitdown_include_front_matter: bool = False

    # Docling specific settings
    docling_enable_ocr: bool = True
    docling_parse_formulas: bool = True
    docling_extract_figures: bool = True

    @classmethod
    def from_environment(cls) -> ExtractionConfig:
        """Load extraction configuration from environment variables.

        Environment variables:
            EXTRACTION_STRATEGY: Extraction strategy (auto, markitdown, docling)
            EXTRACTION_PRESERVE_FORMATTING: Preserve document formatting (default: true)
            EXTRACTION_EXTRACT_IMAGES: Extract images (default: false)
            EXTRACTION_EXTRACT_TABLES: Extract tables (default: true)
            EXTRACTION_EXTRACT_METADATA: Extract metadata (default: true)
            EXTRACTION_OUTPUT_FORMAT: Output format (markdown, html, text)
            EXTRACTION_MAX_FILE_SIZE_MB: Maximum file size in MB (default: 100)
            EXTRACTION_TIMEOUT_SECONDS: Processing timeout (default: 300)
            EXTRACTION_PRIORITY: Processing priority (low, normal, high)

            # MarkItDown specific
            MARKITDOWN_ENABLE_PLUGINS: Enable plugins (default: false)
            MARKITDOWN_DOCINTEL_ENDPOINT: Azure Document Intelligence endpoint
            MARKITDOWN_LLM_MODEL: LLM model for image descriptions
            MARKITDOWN_LLM_PROMPT: Custom prompt for images
            MARKITDOWN_USE_OLLAMA: Use Ollama integration (default: false)
            MARKITDOWN_INCLUDE_FRONT_MATTER: Include front matter (default: false)

            # Docling specific
            DOCLING_ENABLE_OCR: Enable OCR (default: true)
            DOCLING_PARSE_FORMULAS: Parse mathematical formulas (default: true)
            DOCLING_EXTRACT_FIGURES: Extract figures (default: true)

        Returns:
            Configured extraction instance.
        """
        strategy_str = ConfigLoader.get_env_var(
            "EXTRACTION_STRATEGY", default="auto", required=False
        )
        strategy = ExtractionStrategy(strategy_str.lower())

        output_format_str = ConfigLoader.get_env_var(
            "EXTRACTION_OUTPUT_FORMAT", default="markdown", required=False
        )
        output_format = OutputFormat(output_format_str.lower())

        priority_str = ConfigLoader.get_env_var(
            "EXTRACTION_PRIORITY", default="normal", required=False
        )
        priority = ProcessingPriority(priority_str.lower())

        return cls(
            strategy=strategy,
            preserve_formatting=ConfigLoader.get_env_bool(
                "EXTRACTION_PRESERVE_FORMATTING", default=True
            ),
            extract_images=ConfigLoader.get_env_bool(
                "EXTRACTION_EXTRACT_IMAGES", default=False
            ),
            extract_tables=ConfigLoader.get_env_bool(
                "EXTRACTION_EXTRACT_TABLES", default=True
            ),
            extract_metadata=ConfigLoader.get_env_bool(
                "EXTRACTION_EXTRACT_METADATA", default=True
            ),
            output_format=output_format,
            max_file_size_mb=ConfigLoader.get_env_int(
                "EXTRACTION_MAX_FILE_SIZE_MB", default=100, required=False
            ),
            timeout_seconds=ConfigLoader.get_env_int(
                "EXTRACTION_TIMEOUT_SECONDS", default=300, required=False
            ),
            priority=priority,
            # MarkItDown settings
            markitdown_enable_plugins=ConfigLoader.get_env_bool(
                "MARKITDOWN_ENABLE_PLUGINS", default=False
            ),
            markitdown_docintel_endpoint=ConfigLoader.get_env_var(
                "MARKITDOWN_DOCINTEL_ENDPOINT", default=None, required=False
            ),
            markitdown_llm_model=ConfigLoader.get_env_var(
                "MARKITDOWN_LLM_MODEL", default=None, required=False
            ),
            markitdown_llm_prompt=ConfigLoader.get_env_var(
                "MARKITDOWN_LLM_PROMPT", default=None, required=False
            ),
            markitdown_use_ollama=ConfigLoader.get_env_bool(
                "MARKITDOWN_USE_OLLAMA", default=False
            ),
            markitdown_include_front_matter=ConfigLoader.get_env_bool(
                "MARKITDOWN_INCLUDE_FRONT_MATTER", default=False
            ),
            # Docling settings
            docling_enable_ocr=ConfigLoader.get_env_bool(
                "DOCLING_ENABLE_OCR", default=True
            ),
            docling_parse_formulas=ConfigLoader.get_env_bool(
                "DOCLING_PARSE_FORMULAS", default=True
            ),
            docling_extract_figures=ConfigLoader.get_env_bool(
                "DOCLING_EXTRACT_FIGURES", default=True
            ),
        )

    @classmethod
    def from_protocol_data(
        cls,
        data: str | int | float | bool | list[str],
        protocol_serializer: ProtocolSerializer,
    ) -> ExtractionConfig:
        """Load configuration from protocol data using ProtocolSerializer.

        Args:
            data: Protocol data.
            protocol_serializer: Serializer for protocol boundary conversions.

        Returns:
            Configured extraction instance.
        """
        # Use ProtocolSerializer for all dict conversions
        return protocol_serializer.deserialize_config_data(data, cls)

    def validate(self) -> list[str]:
        """Validate extraction configuration.

        Returns:
            List of validation errors.
        """
        errors = []

        # Validate enums
        errors.extend(
            ConfigValidator.validate_enum_value(
                self.strategy.value, ExtractionStrategy, "Strategy"
            )
        )
        errors.extend(
            ConfigValidator.validate_enum_value(
                self.output_format.value, OutputFormat, "Output format"
            )
        )
        errors.extend(
            ConfigValidator.validate_enum_value(
                self.priority.value, ProcessingPriority, "Priority"
            )
        )

        # Validate file size
        if self.max_file_size_mb <= 0 or self.max_file_size_mb > 1000:
            errors.append("Max file size must be between 1 and 1000 MB")

        # Validate timeout
        errors.extend(ConfigValidator.validate_timeout(self.timeout_seconds))

        # Validate MarkItDown endpoint if provided
        if self.markitdown_docintel_endpoint:
            errors.extend(
                ConfigValidator.validate_url(
                    self.markitdown_docintel_endpoint, "MarkItDown DocIntel endpoint"
                )
            )

        # Validate LLM model name if provided
        if self.markitdown_llm_model:
            errors.extend(
                ConfigValidator.validate_model_name(
                    self.markitdown_llm_model, "MarkItDown LLM model"
                )
            )

        return errors

    def to_protocol_data(
        self, protocol_serializer: ProtocolSerializer
    ) -> str | int | float | bool | list[str]:
        """Convert to protocol data using ProtocolSerializer - NO direct dict usage.

        Returns:
            Protocol data representation.
        """
        # Use ProtocolSerializer for all dict conversions
        return protocol_serializer.serialize_config_data(self)

    @classmethod
    def create_fast_mode(cls) -> ExtractionConfig:
        """Create configuration optimized for speed.

        Returns:
            Fast extraction configuration.
        """
        return cls(
            strategy=ExtractionStrategy.MARKITDOWN,
            extract_images=False,
            extract_tables=False,
            extract_metadata=False,
            timeout_seconds=60,
            priority=ProcessingPriority.HIGH,
            markitdown_enable_plugins=False,
            docling_enable_ocr=False,
        )

    @classmethod
    def create_comprehensive_mode(cls) -> ExtractionConfig:
        """Create configuration for comprehensive extraction.

        Returns:
            Comprehensive extraction configuration.
        """
        return cls(
            strategy=ExtractionStrategy.AUTO,
            extract_images=True,
            extract_tables=True,
            extract_metadata=True,
            timeout_seconds=600,
            priority=ProcessingPriority.NORMAL,
            markitdown_enable_plugins=False,  # Keep security
            docling_enable_ocr=True,
            docling_parse_formulas=True,
            docling_extract_figures=True,
        )
