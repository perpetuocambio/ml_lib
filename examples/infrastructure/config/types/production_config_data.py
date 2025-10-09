"""Production configuration data types."""

from dataclasses import dataclass

from infrastructure.config.types.base_config_data import BaseConfigData


@dataclass(frozen=True)
class ProductionConfigData(BaseConfigData):
    """Type-safe container for production configuration data - replaces dict with typed classes."""

    debug_mode: bool
    log_level: str
    hot_reload: bool
    cache_ttl_seconds: int
    validation_strict: bool
    health_checks_enabled: bool
    enable_profiling: bool
    performance_monitoring: bool
    auto_scaling: bool
    max_connections: int
    connection_timeout: int
    request_timeout: int
    enable_metrics: bool
    metrics_endpoint: str
    health_check_interval: int
    backup_enabled: bool
    database_host: str
    database_port: int
    database_name: str
    database_ssl_required: bool
    llm_timeout_seconds: int
    llm_max_retries: int
    extraction_timeout_seconds: int
    extraction_max_file_size_mb: int
    scraping_delay_between_requests: float
    scraping_max_pages: int
    scraping_respect_robots_txt: bool
