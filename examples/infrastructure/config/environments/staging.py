"""Staging environment configuration."""

import os
from dataclasses import dataclass

from infrastructure.config.base.base_config import BaseInfrastructureConfig
from infrastructure.config.types.staging_config_data import StagingConfigData


@dataclass(frozen=True)
class StagingConfig(BaseInfrastructureConfig):
    """Staging-specific configuration settings."""

    # Staging flags
    debug_mode: bool = False
    log_level: str = "INFO"
    hot_reload: bool = False

    # Performance settings for staging
    cache_ttl_seconds: int = 1800  # 30 minutes for staging
    validation_strict: bool = True  # Strict validation in staging
    health_checks_enabled: bool = True

    # Staging database settings
    database_host: str = "staging-db.pyintelcivil.internal"
    database_port: int = 5432
    database_name: str = "pyintelcivil_staging"
    database_ssl_required: bool = True

    # Staging LLM settings
    llm_timeout_seconds: int = 45
    llm_max_retries: int = 3

    # Staging extraction settings
    extraction_timeout_seconds: int = 300
    extraction_max_file_size_mb: int = 100

    # Staging scraping settings
    scraping_delay_between_requests: float = 1.0
    scraping_max_pages: int = 25
    scraping_respect_robots_txt: bool = True

    @classmethod
    def from_environment(cls) -> "StagingConfig":
        """Load staging configuration from environment variables."""
        return cls(
            debug_mode=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            hot_reload=os.getenv("HOT_RELOAD", "false").lower() == "true",
            cache_ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "1800")),
            validation_strict=os.getenv("VALIDATION_STRICT", "true").lower() == "true",
            health_checks_enabled=os.getenv("HEALTH_CHECKS_ENABLED", "true").lower()
            == "true",
            database_host=os.getenv(
                "DATABASE_HOST", "staging-db.pyintelcivil.internal"
            ),
            database_port=int(os.getenv("DATABASE_PORT", "5432")),
            database_name=os.getenv("DATABASE_NAME", "pyintelcivil_staging"),
            database_ssl_required=os.getenv("DATABASE_SSL_REQUIRED", "true").lower()
            == "true",
            llm_timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", "45")),
            llm_max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
            extraction_timeout_seconds=int(
                os.getenv("EXTRACTION_TIMEOUT_SECONDS", "300")
            ),
            extraction_max_file_size_mb=int(
                os.getenv("EXTRACTION_MAX_FILE_SIZE_MB", "100")
            ),
            scraping_delay_between_requests=float(
                os.getenv("SCRAPING_DELAY_BETWEEN_REQUESTS", "1.0")
            ),
            scraping_max_pages=int(os.getenv("SCRAPING_MAX_PAGES", "25")),
            scraping_respect_robots_txt=os.getenv(
                "SCRAPING_RESPECT_ROBOTS_TXT", "true"
            ).lower()
            == "true",
        )

    @classmethod
    def from_config_data(cls, data: StagingConfigData) -> "StagingConfig":
        """Load configuration from typed data."""
        return cls(
            debug_mode=data.debug_mode,
            log_level=data.log_level,
            hot_reload=data.hot_reload,
            cache_ttl_seconds=data.cache_ttl_seconds,
            validation_strict=data.validation_strict,
            health_checks_enabled=data.health_checks_enabled,
            database_host=data.database_host,
            database_port=data.database_port,
            database_name=data.database_name,
            database_ssl_required=data.database_ssl_required,
            llm_timeout_seconds=data.llm_timeout_seconds,
            llm_max_retries=data.llm_max_retries,
            extraction_timeout_seconds=data.extraction_timeout_seconds,
            extraction_max_file_size_mb=data.extraction_max_file_size_mb,
            scraping_delay_between_requests=data.scraping_delay_between_requests,
            scraping_max_pages=data.scraping_max_pages,
            scraping_respect_robots_txt=data.scraping_respect_robots_txt,
        )

    def validate(self) -> list[str]:
        """Validate staging configuration."""
        errors = []

        if self.cache_ttl_seconds < 300:
            errors.append("Cache TTL should be at least 300 seconds in staging")

        if not self.database_ssl_required:
            errors.append("SSL must be required for database connections in staging")

        if (
            self.extraction_max_file_size_mb <= 0
            or self.extraction_max_file_size_mb > 200
        ):
            errors.append(
                "Extraction max file size should be between 1 and 200 MB for staging"
            )

        if self.scraping_max_pages <= 0 or self.scraping_max_pages > 50:
            errors.append("Scraping max pages should be between 1 and 50 for staging")

        if self.scraping_delay_between_requests < 0.5:
            errors.append("Scraping delay should be at least 0.5 seconds in staging")

        return errors

    def to_config_data(self) -> StagingConfigData:
        """Convert configuration to typed data."""
        return StagingConfigData(
            debug_mode=self.debug_mode,
            log_level=self.log_level,
            hot_reload=self.hot_reload,
            cache_ttl_seconds=self.cache_ttl_seconds,
            validation_strict=self.validation_strict,
            health_checks_enabled=self.health_checks_enabled,
            database_host=self.database_host,
            database_port=self.database_port,
            database_name=self.database_name,
            database_ssl_required=self.database_ssl_required,
            llm_timeout_seconds=self.llm_timeout_seconds,
            llm_max_retries=self.llm_max_retries,
            extraction_timeout_seconds=self.extraction_timeout_seconds,
            extraction_max_file_size_mb=self.extraction_max_file_size_mb,
            scraping_delay_between_requests=self.scraping_delay_between_requests,
            scraping_max_pages=self.scraping_max_pages,
            scraping_respect_robots_txt=self.scraping_respect_robots_txt,
        )
