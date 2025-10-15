"""LoRA models and entities consolidation.

This module consolidates all LoRA-related classes from the diffusion system:
- LoRAInfo: Runtime information about LoRAs used in generation
- LoRASerializable: Serialization structure for metadata
- LoRAPreferences: User preferences for LoRA selection
- LoRARecommendation: AI recommendations for LoRA selection
- LoRARecommenderProtocol: Interface for LoRA recommender implementations
- LoRAWeights: Low-level LoRA weight structures
- LoRAConfig: Configuration for individual LoRAs
- LoRAInterface: Base interface for LoRA adapters
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable, Any
import numpy as np

from ml_lib.diffusion.registry.registry import ModelMetadata


# ============================================================================
# RUNTIME INFORMATION
# ============================================================================


@dataclass
class LoRAInfo:
    """Information about a LoRA used in generation."""

    name: str
    """LoRA name/ID."""

    alpha: float
    """LoRA weight/alpha value."""

    source: str = "huggingface"
    """Source: 'huggingface', 'civitai', 'local'."""


@dataclass
class LoRASerializable:
    """Serialized LoRA information."""

    name: str
    alpha: float
    source: str


# ============================================================================
# USER PREFERENCES
# ============================================================================


@dataclass
class LoRAPreferences:
    """LoRA selection preferences."""

    max_loras: int = 3
    """Maximum number of LoRAs to apply."""

    min_confidence: float = 0.6
    """Minimum confidence score for LoRA recommendation."""

    allow_style_mixing: bool = True
    """Allow mixing different artistic styles."""

    blocked_tags: list[str] = field(default_factory=list)
    """Tags to block from LoRA selection."""

    def __post_init__(self):
        """Validate preferences."""
        if self.max_loras < 0:
            raise ValueError("max_loras must be non-negative")
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")


# ============================================================================
# AI RECOMMENDATIONS
# ============================================================================


@dataclass
class LoRARecommendation:
    """Recommendation for a LoRA."""

    lora_name: str
    lora_metadata: ModelMetadata
    confidence_score: float
    suggested_alpha: float
    matching_concepts: list[str] = field(default_factory=list)
    reasoning: str = ""

    def __post_init__(self):
        """Validate recommendation."""
        assert 0.0 <= self.confidence_score <= 1.0, "Confidence must be between 0 and 1"
        assert 0.0 < self.suggested_alpha <= 2.0, "Alpha should be between 0 and 2"

    def is_compatible_with(self, other: "LoRARecommendation") -> bool:
        """
        Check compatibility with another LoRA.

        Args:
            other: Another LoRA recommendation

        Returns:
            True if compatible
        """
        # Check for style conflicts
        style_keywords = ["anime", "photorealistic", "cartoon", "3d", "realistic"]

        self_styles = [kw for kw in style_keywords if kw in self.lora_name.lower()]
        other_styles = [kw for kw in style_keywords if kw in other.lora_name.lower()]

        # If both have conflicting styles, not compatible
        if self_styles and other_styles:
            if set(self_styles).isdisjoint(set(other_styles)):
                return False

        # Check base model compatibility
        if self.lora_metadata.base_model != other.lora_metadata.base_model:
            return False

        return True


# ============================================================================
# PROTOCOLS
# ============================================================================


@runtime_checkable
class LoRARecommenderProtocol(Protocol):
    """Protocol for LoRA recommender implementations."""

    def recommend(
        self,
        prompt_analysis,
        base_model: str,
        max_loras: int = 5,
        min_confidence: float = 0.7,
    ) -> list:
        """
        Recommend LoRAs based on prompt analysis.

        Args:
            prompt_analysis: Analysis of the prompt
            base_model: Base model being used
            max_loras: Maximum LoRAs to recommend
            min_confidence: Minimum confidence threshold

        Returns:
            List of LoRA recommendations
        """
        ...


# ============================================================================
# LOW-LEVEL STRUCTURES
# ============================================================================


@dataclass
class LoRAWeights:
    """Estructura de pesos LoRA."""

    adapter_name: str
    rank: int
    alpha: float
    target_modules: list[str]
    lora_up: dict[str, np.ndarray]  # Matrices de up-projection
    lora_down: dict[str, np.ndarray]  # Matrices de down-projection
    metadata: dict[str, Any] = field(default_factory=dict)

    def compute_delta_weights(self, scaling: float = 1.0) -> dict[str, np.ndarray]:
        """Calcula los pesos delta: (alpha/rank) * up @ down * scaling."""
        delta_weights = {}
        scale_factor = (self.alpha / self.rank) * scaling

        for module_name in self.target_modules:
            if module_name in self.lora_up and module_name in self.lora_down:
                delta = scale_factor * (
                    self.lora_up[module_name] @ self.lora_down[module_name]
                )
                delta_weights[module_name] = delta

        return delta_weights

    def merge_with(
        self, other: "LoRAWeights", weight_self: float = 0.5, weight_other: float = 0.5
    ) -> "LoRAWeights":
        """Fusiona con otro LoRA usando pesos específicos."""
        merged_up = {}
        merged_down = {}

        all_modules = set(self.target_modules) | set(other.target_modules)

        for module in all_modules:
            if module in self.lora_up and module in other.lora_up:
                merged_up[module] = (
                    weight_self * self.lora_up[module]
                    + weight_other * other.lora_up[module]
                )
                merged_down[module] = (
                    weight_self * self.lora_down[module]
                    + weight_other * other.lora_down[module]
                )

        return LoRAWeights(
            adapter_name=f"{self.adapter_name}+{other.adapter_name}",
            rank=max(self.rank, other.rank),
            alpha=(self.alpha + other.alpha) / 2,
            target_modules=list(all_modules),
            lora_up=merged_up,
            lora_down=merged_down,
            metadata={
                "merged_from": [self.adapter_name, other.adapter_name],
                "merge_weights": [weight_self, weight_other],
            },
        )


@dataclass
class LoRAConfig:
    """Configuración individual de LoRA."""

    adapter_name: str
    lora_path: Path
    alpha: float = 1.0
    target_modules: list[str] | None = None  # None = auto-detect
    merge_on_load: bool = False

    # Advanced options
    rank: int | None = None  # Auto-detect if None
    apply_to_text_encoder: bool = False
    apply_to_unet: bool = True


# ============================================================================
# INTERFACES
# ============================================================================


class LoRAInterface(ABC):
    """Interface base para adaptadores LoRA."""

    @abstractmethod
    def load_weights(self, path: Path) -> dict[str, np.ndarray]:
        """Carga los pesos LoRA desde archivo."""
        pass

    @abstractmethod
    def get_target_modules(self) -> list[str]:
        """Retorna los módulos objetivo para inyección."""
        pass

    @abstractmethod
    def compute_scaled_weights(self, alpha: float) -> dict[str, np.ndarray]:
        """Calcula pesos escalados por alpha."""
        pass

    @abstractmethod
    def validate_compatibility(self, base_model: str) -> bool:
        """Valida compatibilidad con modelo base."""
        pass
