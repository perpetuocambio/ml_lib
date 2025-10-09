"""Configuration loader for intelligent prompting system."""

import logging
import yaml
from pathlib import Path
from typing import Any, Dict, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PrompterConfig:
    """Complete configuration for intelligent prompting system."""

    concept_categories: Dict[str, List[str]]
    blocked_tags: List[str]
    priority_tags: List[str]
    anatomy_tags: List[str]
    scoring_weights: Dict[str, float]
    lora_limits: Dict[str, Any]
    age_profiles: Dict[str, Any]
    group_profiles: Dict[str, Any]
    activity_profiles: Dict[str, Any]
    detail_presets: Dict[str, Any]
    default_ranges: Dict[str, Any]
    vram_presets: Dict[str, Any]
    prompt_structure: Dict[str, Any]
    model_strategies: Dict[str, Any]
    negative_prompts: Dict[str, Any]


class ConfigLoader:
    """Loads configuration from YAML files."""

    def __init__(self, config_dir: Path | None = None):
        """
        Initialize config loader.

        Args:
            config_dir: Directory containing YAML configs. Defaults to project root config/
        """
        if config_dir is None:
            # Default to config/intelligent_prompting/ in project root
            # Navigate from ml_lib/diffusion/intelligent/prompting/ -> project root
            module_dir = Path(__file__).parent  # prompting/
            project_root = module_dir.parent.parent.parent.parent  # ml_lib root
            self.config_dir = project_root / "config" / "intelligent_prompting"
        else:
            self.config_dir = Path(config_dir)

        if not self.config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {self.config_dir}")

        logger.info(f"ConfigLoader initialized from {self.config_dir}")

    def load_all(self) -> PrompterConfig:
        """
        Load all configuration files.

        Returns:
            Complete configuration
        """
        logger.info("Loading all configuration files...")

        # Load concept categories
        concept_categories = self._load_yaml("concept_categories.yaml")

        # Load LoRA filters
        lora_filters = self._load_yaml("lora_filters.yaml")

        # Load generation profiles
        generation_profiles = self._load_yaml("generation_profiles.yaml")

        # Load prompting strategies
        prompting_strategies = self._load_yaml("prompting_strategies.yaml")

        # Combine into single config
        config = PrompterConfig(
            concept_categories=concept_categories,
            blocked_tags=lora_filters.get("blocked_tags", []),
            priority_tags=lora_filters.get("priority_tags", []),
            anatomy_tags=lora_filters.get("anatomy_tags", []),
            scoring_weights=lora_filters.get("scoring_weights", {}),
            lora_limits={
                "max_loras": lora_filters.get("max_loras", 3),
                "min_confidence": lora_filters.get("min_confidence", 0.5),
                "max_total_weight": lora_filters.get("max_total_weight", 3.0),
            },
            age_profiles=generation_profiles.get("age_profiles", {}),
            group_profiles=generation_profiles.get("group_profiles", {}),
            activity_profiles=generation_profiles.get("activity_profiles", {}),
            detail_presets=generation_profiles.get("detail_presets", {}),
            default_ranges=generation_profiles.get("default_ranges", {}),
            vram_presets=generation_profiles.get("vram_presets", {}),
            prompt_structure=prompting_strategies.get("prompt_structure", {}),
            model_strategies=prompting_strategies.get("model_strategies", {}),
            negative_prompts=prompting_strategies.get("negative_prompts", {}),
        )

        logger.info("Configuration loaded successfully")
        return config

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """
        Load a YAML file.

        Args:
            filename: Name of YAML file in config_dir

        Returns:
            Parsed YAML content
        """
        filepath = self.config_dir / filename

        if not filepath.exists():
            logger.warning(f"Config file not found: {filepath}, returning empty dict")
            return {}

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f)

            logger.debug(f"Loaded {filename}")
            return content if content is not None else {}

        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")
            return {}

    def get_concept_categories(self) -> Dict[str, List[str]]:
        """Load only concept categories."""
        return self._load_yaml("concept_categories.yaml")

    def get_lora_filters(self) -> Dict[str, Any]:
        """Load only LoRA filters."""
        return self._load_yaml("lora_filters.yaml")

    def get_generation_profiles(self) -> Dict[str, Any]:
        """Load only generation profiles."""
        return self._load_yaml("generation_profiles.yaml")

    def get_prompting_strategies(self) -> Dict[str, Any]:
        """Load only prompting strategies."""
        return self._load_yaml("prompting_strategies.yaml")


# Singleton instance for convenience
_default_loader: ConfigLoader | None = None


def get_default_config() -> PrompterConfig:
    """
    Get default configuration (singleton pattern).

    Returns:
        Loaded configuration
    """
    global _default_loader

    if _default_loader is None:
        _default_loader = ConfigLoader()

    return _default_loader.load_all()


def reload_config(config_dir: Path | None = None) -> PrompterConfig:
    """
    Force reload configuration from disk.

    Args:
        config_dir: Optional custom config directory

    Returns:
        Reloaded configuration
    """
    global _default_loader

    _default_loader = ConfigLoader(config_dir)
    return _default_loader.load_all()
