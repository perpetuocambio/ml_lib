"""Base configuration data types."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BaseConfigData:
    """Base class for all configuration data types - replaces dict with typed classes."""

    pass
