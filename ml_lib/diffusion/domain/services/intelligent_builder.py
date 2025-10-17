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

import gc
import logging
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from diffusers import (
    AutoencoderKL,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    AutoPipelineForText2Image,
)
from PIL import Image

from ml_lib.diffusion.config import detect_comfyui_installation, ModelPathConfig
from ml_lib.diffusion.models import BaseModel, ModelType
from .memory_optimizer import (
    MemoryOptimizer,
    MemoryOptimizationConfig,
    OptimizationLevel,
)
from ml_lib.diffusion.domain.services.memory_optimizer import MemoryMonitor
from ml_lib.diffusion.domain.services.model_orchestrator import (
    ModelOrchestrator,
    DiffusionArchitecture,
)
from ml_lib.diffusion.domain.services.model_registry import ModelRegistry
from ml_lib.diffusion.domain.services.ollama_selector import (
    OllamaModelSelector,
    ModelMatcher,
)
from ml_lib.diffusion.domain.services.prompt_analyzer import PromptAnalyzer
from ml_lib.diffusion.domain.services.prompt_compactor import PromptCompactor
from ml_lib.diffusion.models.value_objects import ProcessedPrompt
from ml_lib.diffusion.models.content_tags import PromptCompactionResult
from ml_lib.diffusion.infrastructure.storage.user_preferences_db import UserPreferencesDB, UserPreferences
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
    memory_mode: str = "auto" # "auto", "low", "balanced", "aggressive"

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
        ollama_model: str = "dolphin3",
        ollama_url: Optional[str] = None,
        device: Optional[str] = None,
        enable_auto_download: bool = False,
        user_preferences_db: Optional[UserPreferencesDB] = None,
        user_id: Optional[str] = None,
    ):
        """
        Initialize builder.

        Args:
            model_config: Model path configuration
            enable_ollama: Enable intelligent model selection via Ollama
            ollama_model: Ollama model for semantic analysis
            device: Device to use (None = auto-detect)
            enable_auto_download: Enable automatic model download from HF/CivitAI
            user_preferences_db: User preferences database (optional)
            user_id: User ID for preferences (required if user_preferences_db is provided)
        """
        self.model_config = model_config
        self.enable_ollama = enable_ollama
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.enable_auto_download = enable_auto_download

        # User preferences
        self.user_prefs_db = user_preferences_db
        self.user_id = user_id
        if user_preferences_db and not user_id:
            raise ValueError("user_id is required when user_preferences_db is provided")

        # Initialize resource monitor
        self.resource_monitor = ResourceMonitor()

        # Detect device
        self.device = device or self._detect_device()

        # Initialize model registry (for auto-download)
        if enable_auto_download:
            self.registry = ModelRegistry()
            logger.info("ModelRegistry initialized for auto-download")
        else:
            self.registry = None

        # Initialize orchestrator with SQLite
        self.orchestrator = ModelOrchestrator(
            enable_ollama=enable_ollama,
            ollama_model=ollama_model,
            resource_monitor=self.resource_monitor,
        )

        # Initialize prompt analyzer for prompt optimization
        # Note: optimize_for_model() doesn't use LLM, so we disable it for speed
        self.prompt_analyzer = PromptAnalyzer(
            ollama_url=ollama_url or "http://localhost:11434",
            model_name=ollama_model,
            use_llm=False,  # Prompt optimization uses rules, not LLM
        )

        # Initialize prompt compactor for intelligent compaction
        self.prompt_compactor = PromptCompactor()

        # Memory optimizer (configured per-generation)
        self.memory_optimizer: Optional[MemoryOptimizer] = None

        logger.info(
            f"IntelligentPipelineBuilder initialized: "
            f"device={self.device}, ollama={enable_ollama}, "
            f"auto_download={enable_auto_download}"
        )

    @classmethod
    def from_comfyui_auto(
        cls,
        enable_ollama: bool = True,  # Activado por defecto para selección inteligente
        ollama_model: Optional[str] = None,
        ollama_url: Optional[str] = None,
        search_paths: Optional[list[Path | str]] = None,
        device: Optional[str] = None,
        enable_auto_download: bool = False,
    ) -> "IntelligentPipelineBuilder":
        """
        Create builder with ComfyUI auto-detection.

        Args:
            enable_ollama: Enable intelligent model selection
            search_paths: Custom search paths for ComfyUI
            enable_auto_download: Enable automatic model download

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
        return cls(
            model_config=config,
            enable_ollama=enable_ollama,
            ollama_model=ollama_model or "dolphin3",
            ollama_url=ollama_url,
            device=device,
            enable_auto_download=enable_auto_download,
        )

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


    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        style: Optional[str] = None,
        quality: str = "balanced",
        memory_mode: str = "auto",
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
            memory_mode=memory_mode,
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
        logger.info("Starting intelligent model selection...")

        # Get available resources
        resources = self.resource_monitor.get_current_stats()

        # Step 1: Analyze prompt with Ollama (if enabled)
        prompt_analysis = None
        if self.enable_ollama:
            try:
                selector = OllamaModelSelector(
                    ollama_model=self.ollama_model,
                    ollama_url=self.ollama_url or "http://localhost:11434"
                )
                prompt_analysis = selector.analyze_prompt(config.prompt)
                if prompt_analysis:
                    logger.info(
                        f"Prompt analysis: style={prompt_analysis.style}, "
                        f"suggested_model={prompt_analysis.suggested_base_model}"
                    )

                # Stop Ollama server to free GPU memory for generation (force=True)
                if selector.stop_server(force=True):
                    logger.info("Stopped Ollama server to free GPU memory for generation")
                    # Wait a bit for process to fully terminate
                    import time
                    time.sleep(2)
                    # Force CUDA cleanup
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                else:
                    logger.warning("Failed to stop Ollama server, generation may use more memory")

            except Exception as e:
                logger.warning(f"Ollama analysis failed, using fallback: {e}")

        # Step 2: Get available models from orchestrator
        base_models = self.orchestrator.metadata_index.get(ModelType.BASE_MODEL, [])
        loras = self.orchestrator.metadata_index.get(ModelType.LORA, [])
        vaes = self.orchestrator.metadata_index.get(ModelType.VAE, [])

        logger.info(
            f"Available: {len(base_models)} base models, "
            f"{len(loras)} LoRAs, {len(vaes)} VAEs"
        )

        # Step 3: Select base model
        selected_base = None
        base_architecture = BaseModel.SDXL  # Default

        if prompt_analysis and base_models:
            # Use ModelMatcher with Ollama analysis
            matcher = ModelMatcher()
            selected_base = matcher.match_base_model(prompt_analysis, base_models)

        if not selected_base and base_models:
            # Fallback: Use style hint or highest rated
            if config.style:
                for model in sorted(base_models, key=lambda m: m.popularity_score, reverse=True):
                    if config.style.lower() in model.model_name.lower():
                        selected_base = model
                        break

            if not selected_base:
                # Just use most popular
                selected_base = max(base_models, key=lambda m: m.popularity_score)

        if selected_base:
            base_model_path = selected_base.file_path
            base_architecture = selected_base.get_base_model_enum()
            logger.info(f"Selected base model: {selected_base.model_name} ({base_architecture.value})")
        else:
            # No models with metadata found locally
            logger.warning("No base models found locally")

            # Try auto-download if enabled
            if self.enable_auto_download and self.registry:
                logger.info("Attempting auto-download from HuggingFace/CivitAI...")

                # Determine what to search for based on prompt analysis or style
                search_query = "stable-diffusion-xl-base"  # Default to SDXL

                if prompt_analysis:
                    # Use suggested architecture from prompt analysis
                    arch_map = {
                        "SD15": "stable-diffusion-v1-5",
                        "SDXL": "stable-diffusion-xl-base",
                        "FLUX": "flux-dev",
                        "SD3": "stable-diffusion-3",
                    }
                    search_query = arch_map.get(
                        prompt_analysis.suggested_base_model, search_query
                    )
                elif config.style:
                    # Use style hint
                    if "anime" in config.style.lower():
                        search_query = "anime sdxl"
                    elif "realistic" in config.style.lower():
                        search_query = "realistic-vision"

                try:
                    downloaded_model = self.registry.find_or_download(
                        query=search_query,
                        model_type=ModelType.BASE_MODEL,
                        base_model=base_architecture,
                        auto_download=True,
                    )

                    if downloaded_model and downloaded_model.local_path:
                        base_model_path = downloaded_model.local_path
                        logger.info(
                            f"✅ Auto-downloaded: {downloaded_model.name} "
                            f"({downloaded_model.size_gb:.1f}GB)"
                        )
                    else:
                        logger.error("Auto-download failed, cannot proceed")
                        base_model_path = Path("/placeholder/model.safetensors")
                except Exception as e:
                    logger.error(f"Auto-download error: {e}")
                    base_model_path = Path("/placeholder/model.safetensors")
            else:
                # No auto-download, use placeholder
                base_model_path = Path("/placeholder/model.safetensors")

        # Step 4: Select LoRAs
        selected_loras = []
        lora_weights = []

        # Always try to select LoRAs if available, even without Ollama analysis
        if loras and selected_base:
            try:
                # If no prompt_analysis, create a fallback analysis
                if not prompt_analysis:
                    logger.info("Creating fallback analysis for LoRA selection")
                    selector = OllamaModelSelector(
                        ollama_model=self.ollama_model,
                        ollama_url=self.ollama_url or "http://localhost:11434"
                    )
                    prompt_analysis = selector._fallback_analysis(config.prompt)
                    # Don't need Ollama anymore, stop it immediately
                    selector.stop_server(force=True)

                matcher = ModelMatcher()
                lora_matches = matcher.match_loras(
                    analysis=prompt_analysis,
                    available_loras=loras,
                    base_model_architecture=base_architecture.value,
                    max_loras=3,
                )

                for lora, weight in lora_matches:
                    selected_loras.append(lora.file_path)
                    lora_weights.append(weight)

                logger.info(f"Selected {len(selected_loras)} LoRAs")
            except Exception as e:
                logger.warning(f"LoRA selection failed: {e}")

        # Step 5: Select VAE (if available)
        vae_path = None
        if vaes and selected_base:
            # Find compatible VAE
            arch_info = DiffusionArchitecture.get_architecture(base_architecture)

            for vae in sorted(vaes, key=lambda v: v.popularity_score, reverse=True):
                # Check compatibility
                vae_name_lower = vae.file_name.lower()
                if any(pattern in vae_name_lower for pattern in arch_info.compatible_vae_patterns):
                    vae_path = vae.file_path
                    logger.info(f"Selected VAE: {vae.file_name}")
                    break

        # Step 6: Determine optimization level
        opt_level = self._determine_optimization_level(
            quality=config.quality, memory_mode=config.memory_mode, available_memory_gb=resources.available_gpu_memory_gb()
        )

        logger.info(
            f"Optimization: {opt_level.value} "
            f"(GPU: {resources.available_gpu_memory_gb():.1f}GB available)"
        )

        # Step 7: Determine generation parameters
        # Use prompt analysis recommendations if available, else use architecture defaults
        if prompt_analysis:
            steps = config.steps or prompt_analysis.suggested_steps
            cfg_scale = config.cfg_scale or prompt_analysis.suggested_cfg
            sampler = config.sampler or "DPM++ 2M"
        else:
            arch_info = DiffusionArchitecture.get_architecture(base_architecture)
            steps = config.steps or arch_info.default_steps
            cfg_scale = config.cfg_scale or arch_info.default_cfg
            sampler = config.sampler or arch_info.default_sampler

        # Step 8: Apply user preferences (filter blocked models, apply defaults)
        if self.user_prefs_db and self.user_id:
            base_model_path, selected_loras, lora_weights, steps, cfg_scale, sampler = self._apply_user_preferences(
                base_model_path=base_model_path,
                selected_loras=selected_loras,
                lora_weights=lora_weights,
                steps=steps,
                cfg_scale=cfg_scale,
                sampler=sampler,
                config=config,
            )

        # Return complete selection
        return SelectedModels(
            base_model_path=base_model_path,
            base_model_architecture=base_architecture,
            vae_path=vae_path,
            lora_paths=selected_loras,
            lora_weights=lora_weights,
            steps=steps,
            cfg_scale=cfg_scale,
            sampler=sampler,
            scheduler="karras",
            clip_skip=2,
            optimization_level=opt_level,
        )

    def _apply_user_preferences(
        self,
        base_model_path: Path,
        selected_loras: list[Path],
        lora_weights: list[float],
        steps: int,
        cfg_scale: float,
        sampler: str,
        config: GenerationConfig,
    ) -> tuple[Path, list[Path], list[float], int, float, str]:
        """
        Apply user preferences to model selection.

        Filters out blocked models/LoRAs and applies user defaults.

        Returns:
            Tuple of (base_model_path, selected_loras, lora_weights, steps, cfg_scale, sampler)
        """
        if not self.user_prefs_db or not self.user_id:
            return base_model_path, selected_loras, lora_weights, steps, cfg_scale, sampler

        # Get user preferences
        prefs = self.user_prefs_db.get_or_create_preferences(self.user_id)

        # Check if base model is blocked
        base_model_name = base_model_path.stem
        if base_model_name in prefs.blocked_models:
            logger.warning(f"Base model '{base_model_name}' is blocked by user preferences, finding alternative...")

            # Try to find alternative from favorites
            if prefs.favorite_base_models:
                # Get all available base models
                base_models = self.orchestrator.metadata_index.get(ModelType.BASE_MODEL, [])

                # Find first favorite that's available
                for fav_name in prefs.favorite_base_models:
                    for model in base_models:
                        if fav_name.lower() in model.model_name.lower() or fav_name.lower() in model.file_name.lower():
                            base_model_path = model.file_path
                            logger.info(f"Using favorite model instead: {model.model_name}")
                            break
                    else:
                        continue
                    break

        # Filter blocked LoRAs
        filtered_loras = []
        filtered_weights = []
        for lora_path, weight in zip(selected_loras, lora_weights):
            lora_name = lora_path.stem
            if lora_name not in prefs.blocked_loras:
                filtered_loras.append(lora_path)
                filtered_weights.append(weight)
            else:
                logger.info(f"Filtered out blocked LoRA: {lora_name}")

        selected_loras = filtered_loras
        lora_weights = filtered_weights

        # Apply user's default parameters if not explicitly overridden
        if not config.steps:
            steps = prefs.default_steps
            logger.debug(f"Using user's default steps: {steps}")

        if not config.cfg_scale:
            cfg_scale = prefs.default_cfg
            logger.debug(f"Using user's default CFG: {cfg_scale}")

        if not config.sampler:
            sampler = prefs.default_sampler
            logger.debug(f"Using user's default sampler: {sampler}")

        return base_model_path, selected_loras, lora_weights, steps, cfg_scale, sampler

    def _determine_optimization_level(
        self, quality: str, memory_mode: str, available_memory_gb: float
    ) -> OptimizationLevel:
        """Determine optimization level based on quality and available memory."""
        # Direct mapping from memory_mode if specified
        if memory_mode == "low":
            return OptimizationLevel.BALANCED
        if memory_mode == "balanced":
            return OptimizationLevel.AGGRESSIVE
        if memory_mode == "aggressive":
            return OptimizationLevel.ULTRA

        # Fallback to auto logic based on quality and available VRAM
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
        logger.info(f"Loading {selected.base_model_architecture.value} pipeline...")

        try:
            # Determine which pipeline class to use based on architecture
            architecture = selected.base_model_architecture

            # Check if model file exists
            if not selected.base_model_path.exists():
                logger.error(f"Model file not found: {selected.base_model_path}")
                raise FileNotFoundError(f"Model not found: {selected.base_model_path}")

            # Load base pipeline
            if architecture in [BaseModel.SD15, BaseModel.SD20, BaseModel.SD21]:
                # Stable Diffusion 1.x/2.x
                pipeline = StableDiffusionPipeline.from_single_file(
                    str(selected.base_model_path),
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    use_safetensors=True,
                )
                logger.info("Loaded SD 1.x/2.x pipeline")

            elif architecture in [BaseModel.SDXL, BaseModel.PONY]:
                # Stable Diffusion XL
                pipeline = StableDiffusionXLPipeline.from_single_file(
                    str(selected.base_model_path),
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    use_safetensors=True,
                )
                logger.info("Loaded SDXL pipeline")

            elif architecture == BaseModel.FLUX:
                # Flux (try auto pipeline)
                try:
                    pipeline = AutoPipelineForText2Image.from_single_file(
                        str(selected.base_model_path),
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        use_safetensors=True,
                    )
                    logger.info("Loaded Flux pipeline")
                except Exception as e:
                    logger.warning(f"Flux pipeline failed, trying SDXL fallback: {e}")
                    pipeline = StableDiffusionXLPipeline.from_single_file(
                        str(selected.base_model_path),
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        use_safetensors=True,
                    )

            else:
                # Unknown architecture, try auto
                logger.warning(f"Unknown architecture {architecture}, using auto pipeline")
                pipeline = AutoPipelineForText2Image.from_single_file(
                    str(selected.base_model_path),
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    use_safetensors=True,
                )

            # Move to device
            pipeline = pipeline.to(self.device)

            # Load custom VAE if selected
            if selected.vae_path and selected.vae_path.exists():
                try:
                    vae = AutoencoderKL.from_single_file(
                        str(selected.vae_path),
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    )
                    pipeline.vae = vae.to(self.device)
                    logger.info(f"Loaded custom VAE: {selected.vae_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to load VAE, using default: {e}")

            # Load LoRAs if selected
            if selected.lora_paths:
                for lora_path, weight in zip(selected.lora_paths, selected.lora_weights):
                    if lora_path.exists():
                        try:
                            pipeline.load_lora_weights(
                                str(lora_path.parent),
                                weight_name=lora_path.name,
                                adapter_name=lora_path.stem,
                            )
                            # Set weight
                            if hasattr(pipeline, 'set_adapters'):
                                pipeline.set_adapters([lora_path.stem], adapter_weights=[weight])

                            logger.info(f"Loaded LoRA: {lora_path.name} (weight: {weight:.2f})")
                        except Exception as e:
                            logger.warning(f"Failed to load LoRA {lora_path.name}: {e}")

            # Configure scheduler based on sampler
            self._configure_scheduler(pipeline, selected.sampler, selected.scheduler)

            # Enable VAE tiling for memory efficiency (especially for large images)
            if hasattr(pipeline, 'vae') and hasattr(pipeline.vae, 'enable_tiling'):
                pipeline.vae.enable_tiling()
                logger.info("Enabled VAE tiling for memory efficiency")

            # Set safety checker (disable for speed, re-enable in production)
            if hasattr(pipeline, 'safety_checker'):
                pipeline.safety_checker = None

            logger.info("✅ Pipeline loaded successfully")
            return pipeline

        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            raise

    def _configure_scheduler(self, pipeline, sampler: str, scheduler_type: str):
        """Configure pipeline scheduler based on sampler and scheduler type."""
        from diffusers import (
            DPMSolverMultistepScheduler,
            EulerAncestralDiscreteScheduler,
            EulerDiscreteScheduler,
            KDPM2DiscreteScheduler,
            KDPM2AncestralDiscreteScheduler,
            UniPCMultistepScheduler,
        )

        # Map sampler names to scheduler classes
        sampler_map = {
            "dpm++ 2m": DPMSolverMultistepScheduler,
            "dpm++ 2m karras": DPMSolverMultistepScheduler,
            "dpm++ 2m sde": DPMSolverMultistepScheduler,
            "dpm++ 2m sde karras": DPMSolverMultistepScheduler,
            "euler a": EulerAncestralDiscreteScheduler,
            "euler": EulerDiscreteScheduler,
            "dpm2": KDPM2DiscreteScheduler,
            "dpm2 a": KDPM2AncestralDiscreteScheduler,
            "dpm2 karras": KDPM2DiscreteScheduler,
            "dpm2 a karras": KDPM2AncestralDiscreteScheduler,
            "unipc": UniPCMultistepScheduler,
        }

        sampler_lower = sampler.lower()
        scheduler_class = sampler_map.get(sampler_lower, DPMSolverMultistepScheduler)

        try:
            # Create scheduler config from existing scheduler
            config = pipeline.scheduler.config

            # Determine if we need Karras sigmas
            use_karras = "karras" in sampler_lower or "karras" in scheduler_type.lower()

            # Create new scheduler with appropriate settings
            if scheduler_class == DPMSolverMultistepScheduler:
                pipeline.scheduler = scheduler_class.from_config(
                    config,
                    algorithm_type="dpmsolver++",
                    use_karras_sigmas=use_karras,
                    solver_order=2,
                )
            elif scheduler_class in [EulerAncestralDiscreteScheduler, EulerDiscreteScheduler]:
                # Euler samplers
                pipeline.scheduler = scheduler_class.from_config(config)
            elif scheduler_class in [KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler]:
                # DPM2 samplers
                pipeline.scheduler = scheduler_class.from_config(config, use_karras_sigmas=use_karras)
            else:
                # Default
                pipeline.scheduler = scheduler_class.from_config(config)

            logger.info(f"Configured scheduler: {sampler} (Karras: {use_karras})")

        except Exception as e:
            logger.warning(f"Failed to configure scheduler {sampler}: {e}, using default")

    def _optimize_memory(self, pipeline, optimization_level: OptimizationLevel):
        """Apply memory optimization to pipeline."""
        config = MemoryOptimizationConfig.from_level(optimization_level)
        self.memory_optimizer = MemoryOptimizer(config)

        if pipeline:
            self.memory_optimizer.optimize_pipeline(pipeline)

    def _generate_images(
        self, pipeline, config: GenerationConfig, selected: SelectedModels
    ) -> list[Image.Image]:
        """Generate images using pipeline."""
        logger.info(
            f"Generating {config.num_images} image(s) at {config.width}x{config.height}..."
        )

        if pipeline is None:
            logger.error("Pipeline is None, cannot generate")
            return []

        try:
            # Step 1: Optimize prompts for the selected model architecture
            optimized_positive, optimized_negative = self.prompt_analyzer.optimize_for_model(
                prompt=config.prompt,
                negative_prompt=config.negative_prompt,
                base_model_architecture=selected.base_model_architecture.value,
                quality=config.quality,
            )

            logger.info(f"Optimized positive prompt: {optimized_positive[:100]}...")
            logger.debug(f"Optimized negative prompt: {optimized_negative[:100]}...")

            # Step 2: Compact prompt if needed (to fit CLIP's 77-token limit)
            compaction_result = self.prompt_compactor.compact(
                optimized_positive,
                preserve_nsfw=True,
                min_quality_tags=2,
            )

            # Step 3: Create ProcessedPrompt for user feedback
            processed_prompt = ProcessedPrompt(
                original=config.prompt,
                final=compaction_result.compacted_prompt,
                original_token_count=compaction_result.original_token_count,
                final_token_count=compaction_result.compacted_token_count,
                was_modified=compaction_result.was_compacted,
                modifications=[],
                removed_tokens=compaction_result.removed_tokens,
                warnings=compaction_result.warnings,
                architecture=selected.base_model_architecture.value,
                quality_level=config.quality,
            )

            # Add modification descriptions
            if compaction_result.was_compacted:
                processed_prompt.add_modification(
                    f"Added quality tags for {selected.base_model_architecture.value}"
                )
                processed_prompt.add_modification(
                    f"Compacted from {compaction_result.original_token_count} to {compaction_result.compacted_token_count} tokens"
                )

            # Step 4: Log user feedback message
            if processed_prompt.was_modified:
                feedback_msg = processed_prompt.get_user_message(verbose=False)
                logger.warning(f"\n{feedback_msg}")

                # Additional critical warning if high-priority content was lost
                if processed_prompt.has_critical_loss:
                    logger.error(
                        "CRITICAL: Essential content was removed during processing! "
                        "Image may not match your expectations."
                    )

            # Use the final processed prompt for generation
            final_positive_prompt = processed_prompt.final

            # Start memory monitoring
            with MemoryMonitor(self.memory_optimizer) as monitor:
                start_time = time.time()

                # Prepare generation kwargs with processed prompts
                generation_kwargs = {
                    "prompt": final_positive_prompt,  # Use compacted prompt
                    "negative_prompt": optimized_negative,
                    "num_inference_steps": selected.steps,
                    "guidance_scale": selected.cfg_scale,
                    "width": config.width,
                    "height": config.height,
                    "num_images_per_prompt": config.num_images,
                }

                # Add seed if specified
                if config.seed is not None:
                    generator = torch.Generator(device=self.device).manual_seed(config.seed)
                    generation_kwargs["generator"] = generator
                    logger.info(f"Using seed: {config.seed}")

                # Add CLIP skip if pipeline supports it
                if hasattr(pipeline, "text_encoder") and selected.clip_skip > 1:
                    generation_kwargs["clip_skip"] = selected.clip_skip

                # Generate
                logger.info(
                    f"Generation params: steps={selected.steps}, "
                    f"cfg={selected.cfg_scale:.1f}, "
                    f"sampler={selected.sampler}"
                )

                result = pipeline(**generation_kwargs)

                # Extract images
                if hasattr(result, "images"):
                    images = result.images
                else:
                    # Some pipelines return different format
                    images = [result] if isinstance(result, Image.Image) else []

                generation_time = time.time() - start_time
                peak_vram = monitor.get_peak_memory()

                logger.info(
                    f"✅ Generated {len(images)} image(s) in {generation_time:.1f}s "
                    f"(peak VRAM: {peak_vram:.2f}GB)"
                )

                # Step 5: Record generation in user preferences database
                if self.user_prefs_db and self.user_id and images:
                    self._record_successful_generation(
                        config=config,
                        selected=selected,
                    )

                return images

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            traceback.print_exc()
            return []

    def _record_successful_generation(
        self,
        config: GenerationConfig,
        selected: SelectedModels,
    ):
        """Record successful generation in user preferences database for learning."""
        if not self.user_prefs_db or not self.user_id:
            return

        try:
            # Extract model names from paths
            base_model_name = selected.base_model_path.stem
            lora_names = [lora_path.stem for lora_path in selected.lora_paths]

            # Record generation
            self.user_prefs_db.record_generation(
                user_id=self.user_id,
                prompt=config.prompt,
                base_model=base_model_name,
                loras=lora_names,
                quality=config.quality,
                steps=selected.steps,
                cfg=selected.cfg_scale,
                sampler=selected.sampler,
                width=config.width,
                height=config.height,
                rating=None,  # User can rate later via API
            )

            logger.debug(f"Recorded generation for user {self.user_id}")

        except Exception as e:
            logger.warning(f"Failed to record generation: {e}")

    def _cleanup(self, pipeline):
        """Clean up resources."""
        if self.memory_optimizer:
            self.memory_optimizer.cleanup_after_generation()

        # Offload pipeline
        if pipeline:
            try:
                if hasattr(pipeline, "to"):
                    pipeline.to("cpu")
            except Exception as e:
                logger.debug(f"Could not offload pipeline to CPU: {e}")

        # Force garbage collection
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
