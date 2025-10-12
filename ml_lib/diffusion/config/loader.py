"""Configuration loader for the diffusion module."""

import json
import yaml
from pathlib import Path
from typing import Any

from .base import DiffusionConfig
from .types import OptimizationLevel, SafetyLevel


class ConfigLoader:
    """Loads configuration from files or dictionaries.

    Supports loading from:
    - JSON files
    - YAML files
    - Python dictionaries

    Example:
        >>> loader = ConfigLoader()
        >>> config = loader.from_file("config.yaml")
        >>> config = loader.from_dict({"safety_level": "relaxed"})
    """

    @staticmethod
    def from_file(path: str | Path) -> DiffusionConfig:
        """Load configuration from a file.

        Args:
            path: Path to configuration file (JSON or YAML).

        Returns:
            DiffusionConfig instance.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file format is not supported.

        Example:
            >>> config = ConfigLoader.from_file("settings.yaml")
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        # Read file
        content = path.read_text()

        # Parse based on extension
        if path.suffix in [".yaml", ".yml"]:
            try:
                import yaml
                data = yaml.safe_load(content)
            except ImportError:
                raise ImportError("PyYAML is required to load YAML files. Install with: pip install pyyaml")
        elif path.suffix == ".json":
            data = json.loads(content)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        return ConfigLoader.from_dict(data)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> DiffusionConfig:
        """Load configuration from a dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            DiffusionConfig instance.

        Example:
            >>> config = ConfigLoader.from_dict({
            ...     "safety_level": "strict",
            ...     "optimization_level": "quality"
            ... })
        """
        # Convert string enums to enum instances
        if "safety_level" in data and isinstance(data["safety_level"], str):
            data["safety_level"] = SafetyLevel(data["safety_level"])

        if "optimization_level" in data and isinstance(data["optimization_level"], str):
            data["optimization_level"] = OptimizationLevel(data["optimization_level"])

        # Convert cache_dir string to Path
        if "cache_dir" in data and isinstance(data["cache_dir"], str):
            data["cache_dir"] = Path(data["cache_dir"])

        # Create config with provided data
        # Note: This is a simplified version. A full implementation would
        # recursively handle nested configurations.
        return DiffusionConfig(**data)

    @staticmethod
    def to_dict(config: DiffusionConfig) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Args:
            config: DiffusionConfig instance.

        Returns:
            Dictionary representation.

        Example:
            >>> config = get_default_config()
            >>> data = ConfigLoader.to_dict(config)
            >>> print(data["safety_level"])
            'strict'
        """
        from dataclasses import asdict

        data = asdict(config)

        # Convert enums to strings
        if "safety_level" in data:
            data["safety_level"] = data["safety_level"].value

        if "optimization_level" in data:
            data["optimization_level"] = data["optimization_level"].value

        # Convert Path to string
        if "cache_dir" in data and data["cache_dir"] is not None:
            data["cache_dir"] = str(data["cache_dir"])

        return data

    @staticmethod
    def to_file(config: DiffusionConfig, path: str | Path, format: str = "yaml") -> None:
        """Save configuration to file.

        Args:
            config: DiffusionConfig instance.
            path: Path to save file.
            format: File format ('yaml' or 'json').

        Example:
            >>> config = get_default_config()
            >>> ConfigLoader.to_file(config, "my_config.yaml")
        """
        path = Path(path)
        data = ConfigLoader.to_dict(config)

        if format == "yaml":
            try:
                import yaml
                content = yaml.dump(data, default_flow_style=False, sort_keys=False)
            except ImportError:
                raise ImportError("PyYAML is required to save YAML files. Install with: pip install pyyaml")
        elif format == "json":
            content = json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

        path.write_text(content)


__all__ = ["ConfigLoader"]
