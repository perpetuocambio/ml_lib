"""
ComfyUI Path Integration - OOP-based, no hardcoded env vars.

Allows seamless integration with existing ComfyUI installations,
enabling access to all 3,678+ LoRAs and other models without duplication.
"""

import logging
from pathlib import Path
from typing import Optional

from ml_lib.diffusion.models import ModelType
from ml_lib.diffusion.config.path_config import ModelPathConfig
from ml_lib.diffusion.services import ModelRegistry

logger = logging.getLogger(__name__)


def detect_comfyui_installation(
    search_paths: Optional[list[Path | str]] = None,
) -> Optional[Path]:
    """
    Detect ComfyUI installation directory.

    Args:
        search_paths: Custom paths to search (None = use defaults)

    Returns:
        Path to ComfyUI root if found, None otherwise

    Example:
        >>> # Auto-detect from common locations
        >>> path = detect_comfyui_installation()

        >>> # Search specific paths
        >>> path = detect_comfyui_installation([
        ...     "/opt/comfyui",
        ...     "/home/user/apps/comfyui"
        ... ])
    """
    if search_paths is None:
        # Default search paths (no env vars)
        search_paths = [
            Path("/src/ComfyUI"),  # Docker/container common
            Path.home() / "ComfyUI",  # User home
            Path("./ComfyUI"),  # Current directory
            Path("../ComfyUI"),  # Parent directory
        ]
    else:
        search_paths = [Path(p) for p in search_paths]

    for candidate in search_paths:
        if candidate.exists() and (candidate / "models").exists():
            logger.info(f"ComfyUI detected at: {candidate}")
            return candidate

    logger.debug(f"ComfyUI not found in {len(search_paths)} locations")
    return None


class ComfyUIPathResolver:
    """
    OOP-based resolver for ComfyUI model paths.

    Uses ModelPathConfig for clean, configurable path management.

    Example:
        >>> # Option 1: Auto-detect from ComfyUI
        >>> resolver = ComfyUIPathResolver.from_comfyui("/src/ComfyUI")

        >>> # Option 2: From config
        >>> config = ModelPathConfig(lora_paths=["/models/loras"])
        >>> resolver = ComfyUIPathResolver(config)

        >>> # Option 3: Empty resolver, add paths programmatically
        >>> resolver = ComfyUIPathResolver()
        >>> resolver.config.add_model_path(ModelType.LORA, "/custom/loras")

        >>> # Usage
        >>> loras = resolver.scan_models(ModelType.LORA)
        >>> print(f"Found {len(loras)} LoRAs")
    """

    def __init__(self, config: Optional[ModelPathConfig] = None):
        """
        Initialize path resolver with configuration.

        Args:
            config: Model path configuration (None = empty config)
        """
        self.config = config or ModelPathConfig()
        logger.info(
            f"ComfyUIPathResolver initialized with "
            f"{len(self.config.get_all_configured_types())} model types"
        )

    @classmethod
    def from_comfyui(
        cls,
        comfyui_root: Path | str,
        custom_subdirs: Optional[dict[ModelType, str]] = None,
    ) -> "ComfyUIPathResolver":
        """
        Create resolver from ComfyUI installation.

        Args:
            comfyui_root: Path to ComfyUI root directory
            custom_subdirs: Custom subdirectory mapping

        Returns:
            ComfyUIPathResolver with discovered paths

        Example:
            >>> resolver = ComfyUIPathResolver.from_comfyui("/src/ComfyUI")
        """
        config = ModelPathConfig.from_root(comfyui_root, custom_subdirs)
        return cls(config)

    @classmethod
    def from_auto_detect(
        cls,
        search_paths: Optional[list[Path | str]] = None,
    ) -> Optional["ComfyUIPathResolver"]:
        """
        Auto-detect and create resolver.

        Args:
            search_paths: Paths to search for ComfyUI

        Returns:
            ComfyUIPathResolver if found, None otherwise

        Example:
            >>> resolver = ComfyUIPathResolver.from_auto_detect()
            >>> if resolver:
            ...     loras = resolver.scan_models(ModelType.LORA)
        """
        comfyui_root = detect_comfyui_installation(search_paths)
        if comfyui_root:
            return cls.from_comfyui(comfyui_root)
        return None

    def get_model_paths(self, model_type: ModelType) -> list[Path]:
        """
        Get all paths for a specific model type.

        Args:
            model_type: Type of model

        Returns:
            List of paths (may be empty)
        """
        return self.config.get_paths(model_type)

    def scan_models(
        self,
        model_type: ModelType,
        extensions: Optional[list[str]] = None,
        recursive: bool = True,
    ) -> list[Path]:
        """
        Scan for models of specific type across all configured paths.

        Args:
            model_type: Type of model to scan
            extensions: File extensions to include (None = all common)
            recursive: Scan subdirectories

        Returns:
            List of model file paths

        Example:
            >>> loras = resolver.scan_models(ModelType.LORA)
            >>> print(f"Found {len(loras)} LoRAs")
        """
        base_paths = self.get_model_paths(model_type)
        if not base_paths:
            logger.debug(f"No paths configured for {model_type}")
            return []

        # Default extensions for different model types
        if extensions is None:
            extensions = self._get_default_extensions(model_type)

        models = []
        pattern = "**/*" if recursive else "*"

        for base_path in base_paths:
            if not base_path.exists():
                continue

            for ext in extensions:
                models.extend(base_path.glob(f"{pattern}.{ext}"))

        models = sorted(set(models))  # Remove duplicates, sort
        logger.info(f"Found {len(models)} {model_type.value} models")
        return models

    def _get_default_extensions(self, model_type: ModelType) -> list[str]:
        """Get default file extensions for model type."""
        extension_map = {
            ModelType.BASE_MODEL: ["safetensors", "ckpt", "pt"],
            ModelType.LORA: ["safetensors", "pt"],
            ModelType.CONTROLNET: ["safetensors", "pt", "pth"],
            ModelType.VAE: ["safetensors", "pt", "ckpt"],
            ModelType.EMBEDDING: ["pt", "bin", "safetensors"],
            ModelType.CLIP: ["safetensors", "pt"],
            ModelType.CLIP_VISION: ["safetensors", "pt"],
            ModelType.TEXT_ENCODER: ["safetensors", "pt"],
            ModelType.UNET: ["safetensors", "pt", "gguf"],
        }
        return extension_map.get(model_type, ["safetensors", "pt", "ckpt"])

    def get_comfyui_metadata(self, model_path: Path) -> Optional[dict]:
        """
        Try to load ComfyUI metadata for a model.

        ComfyUI stores metadata in .json files next to models.

        Args:
            model_path: Path to model file

        Returns:
            Metadata dict if found, None otherwise
        """
        metadata_path = model_path.with_suffix(".metadata.json")
        if not metadata_path.exists():
            # Try without .safetensors/.pt extension
            stem = model_path.stem
            metadata_path = model_path.parent / f"{stem}.metadata.json"

        if metadata_path.exists():
            try:
                import json

                with open(metadata_path) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata from {metadata_path}: {e}")

        return None

    def get_stats(self) -> dict[str, int]:
        """
        Get statistics about available models.

        Returns:
            Dictionary with counts per model type

        Example:
            >>> stats = resolver.get_stats()
            >>> print(stats)
            {'checkpoint': 45, 'lora': 3678, 'controlnet': 12, ...}
        """
        stats = {}
        for model_type in ModelType:
            if model_type in self.paths:
                models = self.scan_models(model_type)
                stats[model_type.value] = len(models)
        return stats

    def create_registry_from_comfyui(
        self, model_types: Optional[list[ModelType]] = None
    ) -> "ModelRegistry":
        """
        Create a ModelRegistry populated from ComfyUI models.

        Args:
            model_types: Types to scan (None = all available)

        Returns:
            ModelRegistry with all ComfyUI models registered

        Example:
            >>> resolver = ComfyUIPathResolver()
            >>> registry = resolver.create_registry_from_comfyui()
            >>> print(f"Registered {len(registry.get_all_models())} models")
        """
        registry = ModelRegistry()

        if model_types is None:
            model_types = list(self.paths.keys())

        total_registered = 0

        for model_type in model_types:
            models = self.scan_models(model_type)
            logger.info(f"Registering {len(models)} {model_type.value} models...")

            for model_path in models:
                try:
                    # Load metadata if available
                    metadata = self.get_comfyui_metadata(model_path)

                    # Extract basic info from filename
                    model_id = model_path.stem  # Without extension

                    # Register model
                    registry.register_model(
                        model_type=model_type,
                        model_id=model_id,
                        path=str(model_path),
                        metadata=metadata or {},
                    )
                    total_registered += 1

                except Exception as e:
                    logger.warning(f"Failed to register {model_path}: {e}")

        logger.info(f"✅ Registered {total_registered} models from ComfyUI")
        return registry

    def get_stats(self) -> dict[str, int]:
        """
        Get statistics about available models.

        Returns:
            Dictionary with counts per model type

        Example:
            >>> stats = resolver.get_stats()
            >>> print(f"LoRAs: {stats['lora']:,}")
        """
        stats = {}
        for model_type in self.config.get_all_configured_types():
            models = self.scan_models(model_type)
            stats[model_type.value] = len(models)
        return stats


# Convenience functions for quick setup
def create_comfyui_registry(
    comfyui_root: Optional[str | Path] = None,
    search_paths: Optional[list[Path | str]] = None,
) -> "ModelRegistry":
    """
    Convenience function to create a registry from ComfyUI installation.

    Args:
        comfyui_root: Path to ComfyUI (None = auto-detect)
        search_paths: Paths to search if auto-detecting

    Returns:
        ModelRegistry with all ComfyUI models

    Example:
        >>> # Auto-detect
        >>> registry = create_comfyui_registry()

        >>> # Explicit path
        >>> registry = create_comfyui_registry("/src/ComfyUI")

        >>> # Custom search
        >>> registry = create_comfyui_registry(
        ...     search_paths=["/opt/comfyui", "/home/user/comfyui"]
        ... )
    """
    if comfyui_root is not None:
        resolver = ComfyUIPathResolver.from_comfyui(comfyui_root)
    else:
        resolver = ComfyUIPathResolver.from_auto_detect(search_paths)
        if resolver is None:
            raise ValueError(
                "Could not auto-detect ComfyUI. "
                "Provide comfyui_root explicitly or add to search_paths."
            )

    return resolver.create_registry_from_comfyui()
