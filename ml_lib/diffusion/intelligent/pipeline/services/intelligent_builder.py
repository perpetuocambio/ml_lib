"""
Intelligent Pipeline Builder - Zero-configuration image generation.

User provides: prompt + simple options
System handles: ALL technical complexity

Example:
    >>> from ml_lib.diffusion.intelligent.pipeline import IntelligentPipelineBuilder
    >>>
    >>> # Simplest usage - auto-detect everything
    >>> builder = IntelligentPipelineBuilder.from_comfyui_auto()
    >>> image = builder.generate("a beautiful sunset over mountains")
    >>>
    >>> # With style hint
    >>> image = builder.generate(
    ...     "a girl with pink hair",
    ...     style="anime",
    ...     quality="high"
    ... )
    >>>
    >>> # With Ollama intelligence
    >>> builder = IntelligentPipelineBuilder.from_comfyui_auto(enable_ollama=True)
    >>> image = builder.generate("cyberpunk city at night")
    >>> # Automatically selects best base model + LoRAs based on prompt
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from ml_lib.diffusion.config import detect_comfyui_installation, ModelPathConfig
from ml_lib.diffusion.intelligent.hub_integration.entities import BaseModel
from ml_lib.diffusion.intelligent.memory.services import (
    MemoryOptimizer,
    OptimizationLevel,
)
from ml_lib.diffusion.intelligent.pipeline.services.model_orchestrator import (
    ModelOrchestrator,
    DiffusionArchitecture,
)
from ml_lib.system.resource_monitor import ResourceMonitor

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """
    Simple user-facing generation configuration.

    User provides these simple options, system figures out technical details.
    """

    # Core
    prompt: str
    negative_prompt: str = (
        "worst quality, low quality, blurry, distorted, bad anatomy"
    )

    # Style hints (optional)
    style: Optional[str] = None  # "realistic", "anime", "artistic", etc.
    quality: str = "balanced"  # "fast", "balanced", "high", "ultra"

    # Output
    width: int = 1024
    height: int = 1024
    num_images: int = 1
    seed: Optional[int] = None

    # Technical overrides (optional, for advanced users)
    steps: Optional[int] = None
    cfg_scale: Optional[float] = None
    sampler: Optional[str] = None


@dataclass
class SelectedModels:
    """Models selected by orchestrator."""

    # Required
    base_model_path: Path
    base_model_architecture: BaseModel

    # Optional components
    vae_path: Optional[Path] = None
    lora_paths: list[Path] = None
    lora_weights: list[float] = None
    controlnet_path: Optional[Path] = None

    # Generation parameters (optimized)
    steps: int = 30
    cfg_scale: float = 7.0
    sampler: str = "DPM++ 2M"
    scheduler: str = "karras"
    clip_skip: int = 2

    # Memory optimization
    optimization_level: OptimizationLevel = OptimizationLevel.BALANCED

    def __post_init__(self):
        """Initialize lists."""
        if self.lora_paths is None:
            self.lora_paths = []
        if self.lora_weights is None:
            self.lora_weights = []


class IntelligentPipelineBuilder:
    """
    Zero-configuration pipeline builder.

    Abstracts ALL technical complexity:
    - Model selection
    - Architecture detection
    - Component compatibility
    - Memory optimization
    - Parameter tuning

    User just provides prompt.
    """

    def __init__(
        self,
        model_config: ModelPathConfig,
        enable_ollama: bool = False,
        ollama_model: str = "llama3.2",
        device: Optional[str] = None,
    ):
        """
        Initialize builder.

        Args:
            model_config: Model path configuration
            enable_ollama: Enable intelligent model selection via Ollama
            ollama_model: Ollama model for semantic analysis
            device: Device to use (None = auto-detect)
        """
        self.model_config = model_config
        self.enable_ollama = enable_ollama
        self.ollama_model = ollama_model

        # Initialize resource monitor
        self.resource_monitor = ResourceMonitor()

        # Detect device
        self.device = device or self._detect_device()

        # Initialize orchestrator
        model_paths = self._collect_model_paths()
        self.orchestrator = ModelOrchestrator(
            model_paths=model_paths,
            enable_ollama=enable_ollama,
            ollama_model=ollama_model,
            resource_monitor=self.resource_monitor,
        )

        # Memory optimizer (configured per-generation)
        self.memory_optimizer: Optional[MemoryOptimizer] = None

        logger.info(
            f"IntelligentPipelineBuilder initialized: "
            f"device={self.device}, ollama={enable_ollama}"
        )

    @classmethod
    def from_comfyui_auto(
        cls,
        enable_ollama: bool = False,
        search_paths: Optional[list[Path | str]] = None,
    ) -> "IntelligentPipelineBuilder":
        """
        Create builder with ComfyUI auto-detection.

        Args:
            enable_ollama: Enable intelligent model selection
            search_paths: Custom search paths for ComfyUI

        Returns:
            Configured builder ready to generate

        Example:
            >>> builder = IntelligentPipelineBuilder.from_comfyui_auto()
            >>> image = builder.generate("a sunset")
        """
        comfyui_root = detect_comfyui_installation(search_paths)
        if not comfyui_root:
            raise ValueError(
                "ComfyUI not found. Use from_paths() to specify model paths manually."
            )

        config = ModelPathConfig.from_root(comfyui_root)
        return cls(model_config=config, enable_ollama=enable_ollama)

    @classmethod
    def from_paths(
        cls,
        model_paths: dict[str, list[Path | str]],
        enable_ollama: bool = False,
    ) -> "IntelligentPipelineBuilder":
        """
        Create builder from explicit model paths.

        Args:
            model_paths: Dict mapping model types to paths
                Example: {"lora": ["/path/loras"], "checkpoint": ["/path/checkpoints"]}
            enable_ollama: Enable intelligent model selection

        Returns:
            Configured builder

        Example:
            >>> builder = IntelligentPipelineBuilder.from_paths({
            ...     "checkpoint": ["/models/checkpoints"],
            ...     "lora": ["/models/loras"],
            ...     "vae": ["/models/vae"]
            ... })
        """
        config = ModelPathConfig(
            checkpoint_paths=model_paths.get("checkpoint", []),
            lora_paths=model_paths.get("lora", []),
            vae_paths=model_paths.get("vae", []),
            controlnet_paths=model_paths.get("controlnet", []),
            embedding_paths=model_paths.get("embedding", []),
            clip_paths=model_paths.get("clip", []),
            clip_vision_paths=model_paths.get("clip_vision", []),
        )
        return cls(model_config=config, enable_ollama=enable_ollama)

    def _detect_device(self) -> str:
        """Auto-detect best device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _collect_model_paths(self) -> list[Path]:
        """Collect all configured model paths for orchestrator."""
        paths = set()

        # Add all configured paths
        paths.update(self.model_config.checkpoint_paths)
        paths.update(self.model_config.lora_paths)
        paths.update(self.model_config.vae_paths)
        paths.update(self.model_config.controlnet_paths)
        paths.update(self.model_config.embedding_paths)
        paths.update(self.model_config.clip_paths)
        paths.update(self.model_config.clip_vision_paths)

        return list(paths)

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        style: Optional[str] = None,
        quality: str = "balanced",
        width: int = 1024,
        height: int = 1024,
        num_images: int = 1,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Image.Image | list[Image.Image]:
        """
        Generate image(s) from prompt.

        This is the main user-facing API. User provides simple options,
        system handles all technical complexity.

        Args:
            prompt: Text description of desired image
            negative_prompt: What to avoid (optional)
            style: Style hint - "realistic", "anime", "artistic", etc.
            quality: Generation quality - "fast", "balanced", "high", "ultra"
            width: Image width in pixels
            height: Image height in pixels
            num_images: Number of images to generate
            seed: Random seed for reproducibility (None = random)
            **kwargs: Advanced overrides (steps, cfg_scale, sampler, etc.)

        Returns:
            Single image if num_images=1, list otherwise

        Example:
            >>> # Simple
            >>> image = builder.generate("a cat")
            >>>
            >>> # With style
            >>> image = builder.generate("a girl", style="anime", quality="high")
            >>>
            >>> # Advanced override
            >>> image = builder.generate("a landscape", steps=50, cfg_scale=9.0)
        """
        # Create config
        config = GenerationConfig(
            prompt=prompt,
            negative_prompt=negative_prompt
            or "worst quality, low quality, blurry",
            style=style,
            quality=quality,
            width=width,
            height=height,
            num_images=num_images,
            seed=seed,
            steps=kwargs.get("steps"),
            cfg_scale=kwargs.get("cfg_scale"),
            sampler=kwargs.get("sampler"),
        )

        logger.info(f"Generating: {prompt[:50]}...")

        # Step 1: Select optimal models
        selected = self._select_models(config)
        logger.info(
            f"Selected: {selected.base_model_architecture.value} "
            f"with {len(selected.lora_paths)} LoRAs"
        )

        # Step 2: Load and configure pipeline
        pipeline = self._load_pipeline(selected)

        # Step 3: Apply memory optimization
        self._optimize_memory(pipeline, selected.optimization_level)

        # Step 4: Generate
        images = self._generate_images(pipeline, config, selected)

        # Step 5: Cleanup
        self._cleanup(pipeline)

        return images[0] if num_images == 1 else images

    def _select_models(self, config: GenerationConfig) -> SelectedModels:
        """
        Select optimal models for generation.

        This is where the magic happens - analyzes prompt, available models,
        resources, and selects best configuration.
        """
        # TODO: Implement intelligent selection
        # For now, return a placeholder showing the structure

        # Get available resources
        resources = self.resource_monitor.get_current_stats()

        # Determine optimization level based on quality + resources
        opt_level = self._determine_optimization_level(
            quality=config.quality, available_memory_gb=resources.available_gpu_memory_gb()
        )

        logger.info(
            f"Selected optimization: {opt_level.value} "
            f"(GPU: {resources.available_gpu_memory_gb():.1f}GB available)"
        )

        # TODO: Actual model selection logic
        # This is a placeholder structure
        return SelectedModels(
            base_model_path=Path("/placeholder"),
            base_model_architecture=BaseModel.SDXL,
            optimization_level=opt_level,
        )

    def _determine_optimization_level(
        self, quality: str, available_memory_gb: float
    ) -> OptimizationLevel:
        """Determine optimization level based on quality and available memory."""
        # Map quality to memory requirements
        quality_memory_map = {
            "fast": 4.0,  # 4GB+ = NONE, else BALANCED
            "balanced": 8.0,  # 8GB+ = NONE, else BALANCED
            "high": 12.0,  # 12GB+ = NONE, 8GB+ = BALANCED, else AGGRESSIVE
            "ultra": 16.0,  # 16GB+ = NONE, 12GB+ = BALANCED, else AGGRESSIVE
        }

        required_memory = quality_memory_map.get(quality, 8.0)

        if available_memory_gb >= required_memory:
            return OptimizationLevel.NONE
        elif available_memory_gb >= required_memory * 0.66:
            return OptimizationLevel.BALANCED
        elif available_memory_gb >= required_memory * 0.5:
            return OptimizationLevel.AGGRESSIVE
        else:
            return OptimizationLevel.ULTRA

    def _load_pipeline(self, selected: SelectedModels):
        """Load diffusion pipeline with selected models."""
        # TODO: Implement actual pipeline loading
        logger.info(f"Loading {selected.base_model_architecture.value} pipeline...")
        return None  # Placeholder

    def _optimize_memory(self, pipeline, optimization_level: OptimizationLevel):
        """Apply memory optimization to pipeline."""
        from ml_lib.diffusion.intelligent.memory.services import (
            MemoryOptimizerConfig,
        )

        config = MemoryOptimizerConfig.from_level(optimization_level)
        self.memory_optimizer = MemoryOptimizer(config)

        if pipeline:
            self.memory_optimizer.optimize_pipeline(pipeline)

    def _generate_images(
        self, pipeline, config: GenerationConfig, selected: SelectedModels
    ) -> list[Image.Image]:
        """Generate images using pipeline."""
        # TODO: Implement actual generation
        logger.info(
            f"Generating {config.num_images} image(s) at {config.width}x{config.height}..."
        )
        return []  # Placeholder

    def _cleanup(self, pipeline):
        """Clean up resources."""
        if self.memory_optimizer:
            self.memory_optimizer.cleanup()

        # Offload pipeline
        if pipeline:
            try:
                if hasattr(pipeline, "to"):
                    pipeline.to("cpu")
            except:
                pass

        # Force garbage collection
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_stats(self) -> dict:
        """Get builder statistics."""
        return {
            "device": self.device,
            "ollama_enabled": self.enable_ollama,
            "orchestrator": self.orchestrator.get_stats(),
            "resources": {
                "gpu_memory_gb": self.resource_monitor.get_current_stats()
                .available_gpu_memory_gb(),
                "ram_gb": self.resource_monitor.get_current_stats().ram.available_gb,
            },
        }
