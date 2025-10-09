"""Development environment configuration."""

import os
from dataclasses import dataclass

from infrastructure.config.base.base_config import BaseInfrastructureConfig
from infrastructure.config.types.development_config_data import DevelopmentConfigData


@dataclass(frozen=True)
class DevelopmentConfig(BaseInfrastructureConfig):
    """Development-specific configuration settings."""

    # Development flags
    debug_mode: bool = True
    log_level: str = "DEBUG"
    hot_reload: bool = True

    # Performance settings for development
    cache_ttl_seconds: int = 300  # 5 minutes for faster development
    validation_strict: bool = False  # Allow invalid configs in development
    health_checks_enabled: bool = True

    # Development database settings
    database_host: str = "localhost"
    database_port: int = 5432
    database_name: str = "pyintelcivil_dev"
    database_ssl_required: bool = False

    # Development LLM settings
    llm_timeout_seconds: int = 60
    llm_max_retries: int = 2

    # Development extraction settings
    extraction_timeout_seconds: int = 60
    extraction_max_file_size_mb: int = 50

    # Development scraping settings
    scraping_delay_between_requests: float = 0.5
    scraping_max_pages: int = 5
    scraping_respect_robots_txt: bool = False  # More permissive for testing

    @classmethod
    def from_environment(cls) -> "DevelopmentConfig":
        """Load development configuration from environment variables."""
        return cls(
            debug_mode=os.getenv("DEBUG", "true").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "DEBUG"),
            hot_reload=os.getenv("HOT_RELOAD", "true").lower() == "true",
            cache_ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "300")),
            validation_strict=os.getenv("VALIDATION_STRICT", "false").lower() == "true",
            health_checks_enabled=os.getenv("HEALTH_CHECKS_ENABLED", "true").lower()
            == "true",
            database_host=os.getenv("DATABASE_HOST", "localhost"),
            database_port=int(os.getenv("DATABASE_PORT", "5432")),
            database_name=os.getenv("DATABASE_NAME", "pyintelcivil_dev"),
            database_ssl_required=os.getenv("DATABASE_SSL_REQUIRED", "false").lower()
            == "true",
            llm_timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
            llm_max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
            extraction_timeout_seconds=int(
                os.getenv("EXTRACTION_TIMEOUT_SECONDS", "60")
            ),
            extraction_max_file_size_mb=int(
                os.getenv("EXTRACTION_MAX_FILE_SIZE_MB", "50")
            ),
            scraping_delay_between_requests=float(
                os.getenv("SCRAPING_DELAY_BETWEEN_REQUESTS", "0.5")
            ),
            scraping_max_pages=int(os.getenv("SCRAPING_MAX_PAGES", "5")),
            scraping_respect_robots_txt=os.getenv(
                "SCRAPING_RESPECT_ROBOTS_TXT", "false"
            ).lower()
            == "true",
        )

    @classmethod
    def from_config_data(cls, data: DevelopmentConfigData) -> "DevelopmentConfig":
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
        """Validate development configuration."""
        errors = []

        if self.cache_ttl_seconds < 60:
            errors.append("Cache TTL should be at least 60 seconds")

        if (
            self.extraction_max_file_size_mb <= 0
            or self.extraction_max_file_size_mb > 500
        ):
            errors.append(
                "Extraction max file size should be between 1 and 500 MB for development"
            )

        if self.scraping_max_pages <= 0 or self.scraping_max_pages > 100:
            errors.append(
                "Scraping max pages should be between 1 and 100 for development"
            )

        return errors

    def to_config_data(self) -> DevelopmentConfigData:
        """Convert configuration to typed data."""
        return DevelopmentConfigData(
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
