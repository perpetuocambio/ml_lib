"""Web scraping algorithm configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from infrastructure.config.base.base_config import BaseInfrastructureConfig
from infrastructure.config.base.config_loader import ConfigLoader
from infrastructure.config.base.config_validator import ConfigValidator
from infrastructure.config.types.http_headers_container import HttpHeadersContainer
from infrastructure.serialization.protocol_serializer import ProtocolSerializer
from infrastructure.web_scraping.dtos.content_type import ContentType
from infrastructure.web_scraping.dtos.scraping_depth import ScrapingDepth


@dataclass(frozen=True)
class ScrapingConfig(BaseInfrastructureConfig):
    """Configuration for web scraping algorithms.

    Centralizes all web scraping configuration with proper validation
    and environment loading capabilities.
    """

    max_content_length: int = 50000
    max_pages: int = 10
    depth: ScrapingDepth = ScrapingDepth.SINGLE_PAGE
    content_type: ContentType = ContentType.TEXT_WITH_LINKS
    follow_external_links: bool = False
    respect_robots_txt: bool = True
    delay_between_requests: float = 1.0
    max_retries: int = 3
    timeout_seconds: int = 30
    include_metadata: bool = True
    user_agent: str = "PyIntelCivil/1.0"
    custom_headers: HttpHeadersContainer = field(
        default_factory=HttpHeadersContainer.empty
    )

    @classmethod
    def from_environment(cls) -> ScrapingConfig:
        """Load scraping configuration from environment variables.

        Environment variables:
            SCRAPING_MAX_CONTENT_LENGTH: Maximum content length (default: 50000)
            SCRAPING_MAX_PAGES: Maximum pages to scrape (default: 10)
            SCRAPING_DEPTH: Scraping depth (single_page, with_links, deep)
            SCRAPING_CONTENT_TYPE: Content type (text_only, text_with_links, full_content)
            SCRAPING_FOLLOW_EXTERNAL_LINKS: Follow external links (default: false)
            SCRAPING_RESPECT_ROBOTS_TXT: Respect robots.txt (default: true)
            SCRAPING_DELAY_BETWEEN_REQUESTS: Delay between requests in seconds (default: 1.0)
            SCRAPING_MAX_RETRIES: Maximum retry attempts (default: 3)
            SCRAPING_TIMEOUT_SECONDS: Request timeout (default: 30)
            SCRAPING_INCLUDE_METADATA: Include metadata (default: true)
            SCRAPING_USER_AGENT: User agent string

        Returns:
            Configured scraping instance.
        """
        depth_str = ConfigLoader.get_env_var(
            "SCRAPING_DEPTH", default="single_page", required=False
        )
        depth = ScrapingDepth(depth_str.upper())

        content_type_str = ConfigLoader.get_env_var(
            "SCRAPING_CONTENT_TYPE", default="text_with_links", required=False
        )
        content_type = ContentType(content_type_str.upper())

        return cls(
            max_content_length=ConfigLoader.get_env_int(
                "SCRAPING_MAX_CONTENT_LENGTH", default=50000, required=False
            ),
            max_pages=ConfigLoader.get_env_int(
                "SCRAPING_MAX_PAGES", default=10, required=False
            ),
            depth=depth,
            content_type=content_type,
            follow_external_links=ConfigLoader.get_env_bool(
                "SCRAPING_FOLLOW_EXTERNAL_LINKS", default=False
            ),
            respect_robots_txt=ConfigLoader.get_env_bool(
                "SCRAPING_RESPECT_ROBOTS_TXT", default=True
            ),
            delay_between_requests=float(
                ConfigLoader.get_env_var(
                    "SCRAPING_DELAY_BETWEEN_REQUESTS", default="1.0", required=False
                )
            ),
            max_retries=ConfigLoader.get_env_int(
                "SCRAPING_MAX_RETRIES", default=3, required=False
            ),
            timeout_seconds=ConfigLoader.get_env_int(
                "SCRAPING_TIMEOUT_SECONDS", default=30, required=False
            ),
            include_metadata=ConfigLoader.get_env_bool(
                "SCRAPING_INCLUDE_METADATA", default=True
            ),
            user_agent=ConfigLoader.get_env_var(
                "SCRAPING_USER_AGENT", default="PyIntelCivil/1.0", required=False
            ),
            custom_headers=HttpHeadersContainer.empty(),  # Empty headers from environment
        )

    @classmethod
    def from_protocol_data(
        cls,
        data: str | int | float | bool | list[str],
        protocol_serializer: ProtocolSerializer,
    ) -> ScrapingConfig:
        """Load configuration from protocol data using ProtocolSerializer.

        Args:
            data: Protocol data.
            protocol_serializer: Serializer for protocol boundary conversions.

        Returns:
            Configured scraping instance.
        """
        # Use ProtocolSerializer for all dict conversions
        return protocol_serializer.deserialize_config_data(data, cls)

    def validate(self) -> list[str]:
        """Validate scraping configuration.

        Returns:
            List of validation errors.
        """
        errors = []

        # Validate content length
        if (
            self.max_content_length <= 0 or self.max_content_length > 10000000
        ):  # 10MB max
            errors.append(
                "Max content length must be between 1 and 10,000,000 characters"
            )

        # Validate max pages
        if self.max_pages <= 0 or self.max_pages > 1000:
            errors.append("Max pages must be between 1 and 1000")

        # Validate enums
        errors.extend(
            ConfigValidator.validate_enum_value(
                self.depth.value, ScrapingDepth, "Depth"
            )
        )
        errors.extend(
            ConfigValidator.validate_enum_value(
                self.content_type.value, ContentType, "Content type"
            )
        )

        # Validate delay
        if self.delay_between_requests < 0 or self.delay_between_requests > 60:
            errors.append("Delay between requests must be between 0 and 60 seconds")

        # Validate retries
        if self.max_retries < 0 or self.max_retries > 10:
            errors.append("Max retries must be between 0 and 10")

        # Validate timeout
        errors.extend(ConfigValidator.validate_timeout(self.timeout_seconds))

        # Validate user agent
        if not self.user_agent or len(self.user_agent) > 200:
            errors.append("User agent must be between 1 and 200 characters")

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

    def is_aggressive_scraping(self) -> bool:
        """Check if configuration is set for aggressive scraping.

        Returns:
            True if configuration suggests aggressive scraping.
        """
        return (
            self.delay_between_requests < 0.5
            or self.max_pages > 100
            or not self.respect_robots_txt
            or self.follow_external_links
        )

    def get_estimated_time_seconds(self) -> float:
        """Estimate total scraping time.

        Returns:
            Estimated time in seconds.
        """
        base_time_per_page = self.timeout_seconds + self.delay_between_requests
        retry_overhead = (
            base_time_per_page * 0.1 * self.max_retries
        )  # 10% overhead per retry
        return (base_time_per_page + retry_overhead) * self.max_pages

    @classmethod
    def create_fast_mode(cls) -> ScrapingConfig:
        """Create configuration optimized for speed.

        Returns:
            Fast scraping configuration.
        """
        return cls(
            max_content_length=10000,
            max_pages=5,
            depth=ScrapingDepth.SINGLE_PAGE,
            content_type=ContentType.TEXT_ONLY,
            follow_external_links=False,
            respect_robots_txt=True,
            delay_between_requests=0.5,
            max_retries=1,
            timeout_seconds=15,
            include_metadata=False,
            user_agent="PyIntelCivil/1.0 (Fast Mode)",
        )

    @classmethod
    def create_comprehensive_mode(cls) -> ScrapingConfig:
        """Create configuration for comprehensive scraping.

        Returns:
            Comprehensive scraping configuration.
        """
        return cls(
            max_content_length=200000,
            max_pages=50,
            depth=ScrapingDepth.WITH_LINKS,
            content_type=ContentType.FULL_CONTENT,
            follow_external_links=False,  # Keep safe
            respect_robots_txt=True,
            delay_between_requests=2.0,  # Be respectful
            max_retries=3,
            timeout_seconds=60,
            include_metadata=True,
            user_agent="PyIntelCivil/1.0 (Comprehensive Mode)",
        )

    @classmethod
    def create_respectful_mode(cls) -> ScrapingConfig:
        """Create configuration that is very respectful to servers.

        Returns:
            Respectful scraping configuration.
        """
        return cls(
            max_content_length=50000,
            max_pages=10,
            depth=ScrapingDepth.SINGLE_PAGE,
            content_type=ContentType.TEXT_WITH_LINKS,
            follow_external_links=False,
            respect_robots_txt=True,
            delay_between_requests=3.0,  # Extra respectful delay
            max_retries=2,
            timeout_seconds=30,
            include_metadata=True,
            user_agent="PyIntelCivil/1.0 (Respectful Mode)",
        )
