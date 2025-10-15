"""Image naming configuration - separate module to avoid circular imports."""

from dataclasses import dataclass


@dataclass
class ImageNamingConfig:
    """Configuration for image naming conventions."""

    include_timestamp: bool = True
    """Include ISO 8601 timestamp in filename."""

    include_guid: bool = True
    """Include GUID in filename."""

    include_prompt_excerpt: bool = False
    """Include sanitized prompt excerpt (first N chars)."""

    prompt_excerpt_length: int = 30
    """Length of prompt excerpt if included."""

    timestamp_format: str = "%Y%m%d_%H%M%S"
    """Timestamp format (default: YYYYMMDD_HHMMSS)."""

    separator: str = "_"
    """Separator between filename components."""

    extension: str = "png"
    """File extension."""

    @classmethod
    def standard(cls) -> "ImageNamingConfig":
        """Standard naming: timestamp + GUID."""
        return cls(
            include_timestamp=True,
            include_guid=True,
            include_prompt_excerpt=False,
        )

    @classmethod
    def descriptive(cls) -> "ImageNamingConfig":
        """Descriptive naming: timestamp + prompt excerpt + GUID."""
        return cls(
            include_timestamp=True,
            include_guid=True,
            include_prompt_excerpt=True,
            prompt_excerpt_length=30,
        )

    @classmethod
    def guid_only(cls) -> "ImageNamingConfig":
        """GUID-only naming (most anonymous)."""
        return cls(
            include_timestamp=False,
            include_guid=True,
            include_prompt_excerpt=False,
        )
