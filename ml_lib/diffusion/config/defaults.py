"""Default configuration singleton for the diffusion module."""

from pathlib import Path
from .base import DiffusionConfig

# Global default configuration instance
_DEFAULT_CONFIG: DiffusionConfig | None = None


def get_default_config() -> DiffusionConfig:
    """Get the default configuration singleton.

    Returns:
        The default DiffusionConfig instance.

    Example:
        >>> config = get_default_config()
        >>> print(config.lora.max_loras)
        3
    """
    global _DEFAULT_CONFIG
    if _DEFAULT_CONFIG is None:
        _DEFAULT_CONFIG = DiffusionConfig(
            cache_dir=Path.home() / ".cache" / "ml_lib" / "diffusion"
        )
    return _DEFAULT_CONFIG


def set_default_config(config: DiffusionConfig) -> None:
    """Set a custom default configuration.

    This allows overriding the default config for testing or custom setups.

    Args:
        config: The new default configuration.

    Example:
        >>> from ml_lib.diffusion.config import DiffusionConfig, set_default_config
        >>> custom_config = DiffusionConfig(safety_level=SafetyLevel.RELAXED)
        >>> set_default_config(custom_config)
    """
    global _DEFAULT_CONFIG
    _DEFAULT_CONFIG = config


def reset_default_config() -> None:
    """Reset the default configuration to initial state.

    Useful for testing or when you want to reload the default config.

    Example:
        >>> reset_default_config()
        >>> config = get_default_config()  # Gets a fresh default config
    """
    global _DEFAULT_CONFIG
    _DEFAULT_CONFIG = None


__all__ = [
    "get_default_config",
    "set_default_config",
    "reset_default_config",
]
