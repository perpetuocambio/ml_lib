"""Production environment configuration."""

import os
from dataclasses import dataclass

from infrastructure.config.base.base_config import BaseInfrastructureConfig
from infrastructure.config.types.production_config_data import ProductionConfigData


@dataclass(frozen=True)
class ProductionConfig(BaseInfrastructureConfig):
    """Production-specific configuration settings."""

    # Production flags
    debug_mode: bool = False
    log_level: str = "WARNING"
    hot_reload: bool = False

    # Performance settings for production
    cache_ttl_seconds: int = 3600  # 1 hour for production
    validation_strict: bool = True  # Always strict in production
    health_checks_enabled: bool = True

    # Production database settings
    database_host: str = "prod-db.pyintelcivil.com"
    database_port: int = 5432
    database_name: str = "pyintelcivil_prod"
    database_ssl_required: bool = True

    # Production LLM settings
    llm_timeout_seconds: int = 30
    llm_max_retries: int = 3

    # Production extraction settings
    extraction_timeout_seconds: int = 600  # 10 minutes max
    extraction_max_file_size_mb: int = 200

    # Production scraping settings
    scraping_delay_between_requests: float = 2.0  # Respectful to external services
    scraping_max_pages: int = 50
    scraping_respect_robots_txt: bool = True

    @classmethod
    def from_environment(cls) -> "ProductionConfig":
        """Load production configuration from environment variables."""
        return cls(
            debug_mode=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "WARNING"),
            hot_reload=os.getenv("HOT_RELOAD", "false").lower() == "true",
            cache_ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "3600")),
            validation_strict=os.getenv("VALIDATION_STRICT", "true").lower() == "true",
            health_checks_enabled=os.getenv("HEALTH_CHECKS_ENABLED", "true").lower()
            == "true",
            database_host=os.getenv("DATABASE_HOST", "prod-db.pyintelcivil.com"),
            database_port=int(os.getenv("DATABASE_PORT", "5432")),
            database_name=os.getenv("DATABASE_NAME", "pyintelcivil_prod"),
            database_ssl_required=os.getenv("DATABASE_SSL_REQUIRED", "true").lower()
            == "true",
            llm_timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", "30")),
            llm_max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
            extraction_timeout_seconds=int(
                os.getenv("EXTRACTION_TIMEOUT_SECONDS", "600")
            ),
            extraction_max_file_size_mb=int(
                os.getenv("EXTRACTION_MAX_FILE_SIZE_MB", "200")
            ),
            scraping_delay_between_requests=float(
                os.getenv("SCRAPING_DELAY_BETWEEN_REQUESTS", "2.0")
            ),
            scraping_max_pages=int(os.getenv("SCRAPING_MAX_PAGES", "50")),
            scraping_respect_robots_txt=os.getenv(
                "SCRAPING_RESPECT_ROBOTS_TXT", "true"
            ).lower()
            == "true",
        )

    @classmethod
    def from_config_data(cls, data: ProductionConfigData) -> "ProductionConfig":
        """Load configuration from typed data."""
        return cls(
            debug_mode=data.debug_mode,
            log_level=data.log_level,
            enable_profiling=data.enable_profiling,
            performance_monitoring=data.performance_monitoring,
            auto_scaling=data.auto_scaling,
            cache_ttl=data.cache_ttl,
            max_connections=data.max_connections,
            connection_timeout=data.connection_timeout,
            request_timeout=data.request_timeout,
            enable_metrics=data.enable_metrics,
            metrics_endpoint=data.metrics_endpoint,
            health_check_interval=data.health_check_interval,
            backup_enabled=data.backup_enabled,
        )

    def validate(self) -> list[str]:
        """Validate production configuration."""
        errors = []

        if self.debug_mode:
            errors.append("Debug mode must be disabled in production")

        if self.hot_reload:
            errors.append("Hot reload must be disabled in production")

        if self.cache_ttl_seconds < 1800:
            errors.append("Cache TTL should be at least 1800 seconds in production")

        if not self.validation_strict:
            errors.append("Validation must be strict in production")

        if not self.database_ssl_required:
            errors.append("SSL must be required for database connections in production")

        if (
            self.extraction_max_file_size_mb <= 0
            or self.extraction_max_file_size_mb > 500
        ):
            errors.append(
                "Extraction max file size should be between 1 and 500 MB for production"
            )

        if self.scraping_max_pages <= 0 or self.scraping_max_pages > 100:
            errors.append(
                "Scraping max pages should be between 1 and 100 for production"
            )

        if self.scraping_delay_between_requests < 1.0:
            errors.append("Scraping delay should be at least 1.0 seconds in production")

        if not self.scraping_respect_robots_txt:
            errors.append("robots.txt must be respected in production")

        return errors

    def to_config_data(self) -> ProductionConfigData:
        """Convert configuration to typed data."""
        return ProductionConfigData(
            debug_mode=self.debug_mode,
            log_level=self.log_level,
            hot_reload=self.hot_reload,
            cache_ttl_seconds=self.cache_ttl_seconds,
            validation_strict=self.validation_strict,
            health_checks_enabled=self.health_checks_enabled,
            enable_profiling=self.enable_profiling,
            performance_monitoring=self.performance_monitoring,
            auto_scaling=self.auto_scaling,
            max_connections=self.max_connections,
            connection_timeout=self.connection_timeout,
            request_timeout=self.request_timeout,
            enable_metrics=self.enable_metrics,
            metrics_endpoint=self.metrics_endpoint,
            health_check_interval=self.health_check_interval,
            backup_enabled=self.backup_enabled,
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
