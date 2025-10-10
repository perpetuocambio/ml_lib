"""Pipeline configuration entities."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class OperationMode(Enum):
    """Operation mode for the pipeline."""

    AUTO = "auto"
    """Fully automatic - all decisions made by AI."""

    ASSISTED = "assisted"
    """AI suggests, user confirms."""

    MANUAL = "manual"
    """User has full control."""


class Priority(Enum):
    """Generation priority."""

    SPEED = "speed"
    """Optimize for fastest generation."""

    QUALITY = "quality"
    """Optimize for best quality."""

    BALANCED = "balanced"
    """Balance speed and quality."""


@dataclass
class GenerationConstraints:
    """Constraints for generation."""

    max_time_seconds: Optional[float] = None
    """Maximum generation time in seconds."""

    max_vram_gb: Optional[float] = None
    """Maximum VRAM usage in GB."""

    priority: Priority = Priority.BALANCED
    """Generation priority."""

    def __post_init__(self):
        """Validate constraints."""
        if self.max_time_seconds is not None and self.max_time_seconds <= 0:
            raise ValueError("max_time_seconds must be positive")
        if self.max_vram_gb is not None and self.max_vram_gb <= 0:
            raise ValueError("max_vram_gb must be positive")


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


@dataclass
class MemorySettings:
    """Memory management settings."""

    max_vram_gb: float = 8.0
    """Maximum VRAM to use in GB."""

    offload_strategy: str = "balanced"
    """Offload strategy: 'none', 'balanced', 'aggressive'."""

    enable_quantization: bool = False
    """Enable automatic quantization."""

    quantization_dtype: str = "fp16"
    """Quantization data type: 'fp16', 'int8'."""

    def __post_init__(self):
        """Validate settings."""
        if self.max_vram_gb <= 0:
            raise ValueError("max_vram_gb must be positive")
        if self.offload_strategy not in ("none", "balanced", "aggressive"):
            raise ValueError("offload_strategy must be 'none', 'balanced', or 'aggressive'")
        if self.quantization_dtype not in ("fp16", "int8", "int4"):
            raise ValueError("quantization_dtype must be 'fp16', 'int8', or 'int4'")


@dataclass
class OllamaConfig:
    """Ollama LLM configuration."""

    base_url: str = "http://localhost:11434"
    """Ollama API base URL."""

    model: str = "llama2"
    """Model to use for analysis."""

    timeout: float = 30.0
    """Request timeout in seconds."""

    def __post_init__(self):
        """Validate config."""
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")


@dataclass
class PipelineConfig:
    """Configuration for intelligent generation pipeline."""

    base_model: str = "stabilityai/sdxl-base-1.0"
    """Base model ID (HuggingFace or CivitAI)."""

    mode: OperationMode = OperationMode.AUTO
    """Operation mode."""

    constraints: GenerationConstraints = field(default_factory=GenerationConstraints)
    """Generation constraints."""

    lora_preferences: LoRAPreferences = field(default_factory=LoRAPreferences)
    """LoRA selection preferences."""

    memory_settings: MemorySettings = field(default_factory=MemorySettings)
    """Memory management settings."""

    ollama_config: Optional[OllamaConfig] = None
    """Ollama configuration for LLM analysis (None = use rule-based fallback)."""

    enable_learning: bool = True
    """Enable learning from user feedback."""

    cache_dir: Optional[str] = None
    """Custom cache directory (None = default)."""

    def __post_init__(self):
        """Validate config."""
        if not self.base_model:
            raise ValueError("base_model cannot be empty")
