"""Intelligent generation pipeline - main orchestrator."""

import logging
import random
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Protocol

import torch
from PIL import Image
from diffusers import DiffusionPipeline

from ..models.pipeline import (
    PipelineConfig,
    GenerationResult,
    GenerationMetadata,
    LoRAInfo,
    GenerationExplanation,
    Recommendations,
    OperationMode,
)
from ml_lib.diffusion.models.value_objects import ParameterModifications
from ml_lib.diffusion.services.model_registry import ModelRegistry
from ml_lib.diffusion.services.prompt_analyzer import PromptAnalyzer
from ml_lib.diffusion.services.parameter_optimizer import ParameterOptimizer
from ml_lib.diffusion.services.learning_engine import LearningEngine
from ml_lib.diffusion.services.lora_recommender import LoRARecommender
from ml_lib.diffusion.services.model_offloader import ModelOffloader
from ml_lib.diffusion.handlers.memory_manager import MemoryManager
from ml_lib.diffusion.services.memory_optimizer import (
    MemoryOptimizer,
    MemoryOptimizationConfig,
    OptimizationLevel,
    MemoryMonitor,
)
from ml_lib.diffusion.services.learning_engine import GenerationFeedback
from ml_lib.llm.providers import OllamaProvider
from ml_lib.llm.clients import LLMClient

# Import from other intelligent modules
# Note: These would need to be properly implemented/available
logger = logging.getLogger(__name__)


class PromptAnalysisProtocol(Protocol):
    """Protocol for prompt analysis results."""

    pass


class ParameterOptimizationProtocol(Protocol):
    """Protocol for optimized parameters."""

    num_steps: int
    guidance_scale: float
    width: int
    height: int
    sampler_name: str


class LoRARecommendationProtocol(Protocol):
    """Protocol for LoRA recommendations."""

    lora_name: str
    suggested_alpha: float
    confidence_score: float
    reasoning: str


class IntelligentGenerationPipeline:
    """
    Intelligent pipeline for image generation.

    Integrates:
    - Prompt analysis (US 14.2)
    - LoRA recommendation (US 14.2)
    - Parameter optimization (US 14.2)
    - Memory management (US 14.3)
    - Model hub integration (US 14.1)
    - Learning engine (US 14.2)

    Example:
        >>> pipeline = IntelligentGenerationPipeline()
        >>> result = pipeline.generate("anime girl with magical powers")
        >>> result.image.save("output.png")
        >>> print(result.explanation.summary)
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        model_registry: Optional[ModelRegistry] = None,
    ):
        """
        Initialize intelligent generation pipeline.

        Args:
            config: Pipeline configuration (None = use defaults)
            model_registry: Model registry instance (None = create new)
        """
        self.config = config or PipelineConfig()

        # Initialize subsystems
        self._init_subsystems(model_registry)

        # Pipeline state
        self.diffusion_pipeline: Optional[DiffusionPipeline] = None
        self.current_base_model: Optional[str] = None

        logger.info(
            f"IntelligentGenerationPipeline initialized (mode: {self.config.mode.value})"
        )

    def _init_subsystems(self, model_registry: Optional[ModelRegistry]):
        """Initialize all subsystem services."""
        # Model Registry (US 14.1)
        self.registry = model_registry or ModelRegistry()

        # Prompt Analysis (US 14.2)
        ollama_client = (
            self._init_ollama() if self.config.ollama_config else None
        )
        self.prompt_analyzer = PromptAnalyzer(ollama_client=ollama_client)

        # LoRA Recommendation (US 14.2)
        self.lora_recommender = LoRARecommender(registry=self.registry)

        # Parameter Optimization (US 14.2)
        self.param_optimizer = ParameterOptimizer()

        # Memory Management (US 14.3) - OUR MARKET VALUE DIFFERENTIATOR
        self.memory_manager = MemoryManager()
        self.model_offloader = ModelOffloader(
            strategy=self.config.memory_settings.offload_strategy,
            max_vram_gb=self.config.memory_settings.max_vram_gb,
            memory_manager=self.memory_manager,
        )

        # Memory Optimizer (EXTREME OPTIMIZATION - MARKET VALUE)
        opt_level = self._get_optimization_level()
        opt_config = MemoryOptimizationConfig.from_level(opt_level)
        self.memory_optimizer = MemoryOptimizer(opt_config)

        logger.info(f"Memory optimizer enabled: {opt_level.value} mode")

        # Learning Engine (US 14.2)
        if self.config.enable_learning:
            db_path = None
            if self.config.cache_dir:
                db_path = Path(self.config.cache_dir) / "learning.db"
            self.learning_engine = LearningEngine(db_path=db_path)
        else:
            self.learning_engine = None

        logger.info("All subsystems initialized successfully")

    def _get_optimization_level(self) -> OptimizationLevel:
        """
        Determine memory optimization level from config.

        Returns:
            OptimizationLevel enum
        """
        strategy = self.config.memory_settings.offload_strategy.value
        vram = self.memory_manager.resources.available_vram_gb if self.memory_manager else 12.0

        # Map strategy to optimization level
        if strategy == "none":
            return OptimizationLevel.NONE
        elif strategy == "balanced":
            return OptimizationLevel.BALANCED
        elif strategy == "sequential" or strategy == "aggressive":
            return OptimizationLevel.AGGRESSIVE
        elif vram < 6:
            # Auto upgrade to ULTRA for low VRAM
            return OptimizationLevel.ULTRA
        else:
            return OptimizationLevel.BALANCED

    def _init_ollama(self) -> Optional[LLMClient]:
        """Initialize Ollama client for semantic analysis."""
        if self.config.ollama_config is None:
            return None

        provider = OllamaProvider(
            base_url=self.config.ollama_config.base_url,
            model=self.config.ollama_config.model,
            timeout=self.config.ollama_config.timeout,
        )

        return LLMClient(provider=provider)

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        seed: Optional[int] = None,
        **overrides: str,
    ) -> GenerationResult:
        """
        Generate image with intelligent pipeline.

        Workflow:
        1. Analyze prompt (semantic understanding)
        2. Recommend LoRAs (based on analysis)
        3. Optimize parameters (steps, CFG, resolution, etc.)
        4. Configure memory management
        5. Load/prepare models
        6. Generate image
        7. Build metadata and explanation

        Args:
            prompt: Text prompt for generation
            negative_prompt: Negative prompt (optional)
            seed: Random seed (None = random)
            **overrides: Override any optimized parameters

        Returns:
            GenerationResult with image, metadata, and explanation

        Example:
            >>> result = pipeline.generate(
            ...     "anime girl, magical powers",
            ...     negative_prompt="low quality",
            ...     seed=42
            ... )
            >>> result.image.save("output.png")
        """
        generation_id = str(uuid.uuid4())
        start_time = time.time()

        logger.info(f"Starting generation {generation_id[:8]}... for: {prompt[:50]}")

        # Phase 1: Analysis
        logger.debug("Phase 1: Analyzing prompt...")
        analysis = self.prompt_analyzer.analyze(prompt)

        # Phase 2: LoRA Recommendation
        logger.debug("Phase 2: Recommending LoRAs...")
        lora_recs = self.lora_recommender.recommend(
            prompt_analysis=analysis,
            base_model=self.config.base_model,
            max_loras=self.config.lora_preferences.max_loras,
            min_confidence=self.config.lora_preferences.min_confidence,
        )

        # Apply learning engine adjustments if available
        if self.learning_engine:
            lora_recs = self._apply_learning_adjustments(lora_recs)

        # Phase 3: Parameter Optimization
        logger.debug("Phase 3: Optimizing parameters...")
        params = self.param_optimizer.optimize(
            prompt_analysis=analysis, constraints=self.config.constraints
        )

        # Apply user overrides
        for key, value in overrides.items():
            if hasattr(params, key):
                setattr(params, key, value)
                logger.debug(f"Override applied: {key} = {value}")

        # Phase 4: Prepare Pipeline
        logger.debug("Phase 4: Preparing diffusion pipeline...")
        self._ensure_pipeline_loaded(
            base_model=self.config.base_model,
            loras=lora_recs,
        )

        # Phase 5: Generate
        logger.debug("Phase 5: Generating image...")
        seed = seed if seed is not None else random.randint(0, 2**32 - 1)

        # Note: In production, this would actually call the diffusion pipeline
        # For now, we create a placeholder to show the structure
        image = self._generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            params=params,
            seed=seed,
        )

        # Phase 6: Build Metadata and Explanation
        generation_time = time.time() - start_time
        peak_vram = (
            self.memory_manager.get_peak_vram_usage()
            if self.memory_manager
            else 0.0
        )

        metadata = GenerationMetadata(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            steps=params.num_steps,
            cfg_scale=params.guidance_scale,
            width=params.width,
            height=params.height,
            sampler=params.sampler_name,
            loras_used=[
                LoRAInfo(name=r.lora_name, alpha=r.suggested_alpha)
                for r in lora_recs
            ],
            generation_time_seconds=generation_time,
            peak_vram_gb=peak_vram,
            base_model_id=self.config.base_model,
            pipeline_type="intelligent",
        )

        explanation = self._build_explanation(
            analysis=analysis,
            lora_recs=lora_recs,
            params=params,
            generation_time=generation_time,
        )

        logger.info(
            f"Generation {generation_id[:8]} completed in {generation_time:.2f}s"
        )

        return GenerationResult(
            id=generation_id,
            image=image,
            metadata=metadata,
            explanation=explanation,
        )

    def analyze_and_recommend(self, prompt: str) -> Recommendations:
        """
        Analyze prompt and get recommendations without generating.

        Useful for ASSISTED mode where user wants to review/modify before generating.

        Args:
            prompt: Text prompt to analyze

        Returns:
            Recommendations with analysis, suggested LoRAs, and parameters

        Example:
            >>> recs = pipeline.analyze_and_recommend("anime girl")
            >>> print(recs.get_summary())
            >>> # User reviews and modifies
            >>> recs.suggested_params.num_steps = 50
            >>> # Then generate with modifications
            >>> result = pipeline.generate_from_recommendations(prompt, recs)
        """
        logger.info(f"Analyzing and recommending for: {prompt[:50]}")

        # Analysis
        analysis = self.prompt_analyzer.analyze(prompt)

        # LoRA recommendation
        lora_recs = self.lora_recommender.recommend(
            prompt_analysis=analysis,
            base_model=self.config.base_model,
            max_loras=self.config.lora_preferences.max_loras,
        )

        if self.learning_engine:
            lora_recs = self._apply_learning_adjustments(lora_recs)

        # Parameter optimization
        params = self.param_optimizer.optimize(
            prompt_analysis=analysis, constraints=self.config.constraints
        )

        # Build explanation
        explanation = self._build_explanation(analysis, lora_recs, params)

        return Recommendations(
            prompt_analysis=analysis,
            suggested_loras=lora_recs,
            suggested_params=params,
            explanation=explanation.summary,
        )

    def generate_from_recommendations(
        self,
        prompt: str,
        recommendations: Recommendations,
        negative_prompt: str = "",
        seed: Optional[int] = None,
    ) -> GenerationResult:
        """
        Generate image using specific recommendations.

        Allows user to modify recommendations before generation (ASSISTED mode).

        Args:
            prompt: Text prompt
            recommendations: Recommendations to use
            negative_prompt: Negative prompt (optional)
            seed: Random seed (None = random)

        Returns:
            GenerationResult
        """
        generation_id = str(uuid.uuid4())
        start_time = time.time()

        logger.info(
            f"Generating from recommendations {generation_id[:8]}... for: {prompt[:50]}"
        )

        # Use provided recommendations directly
        lora_recs = recommendations.suggested_loras
        params = recommendations.suggested_params

        # Prepare pipeline
        self._ensure_pipeline_loaded(
            base_model=self.config.base_model,
            loras=lora_recs,
        )

        # Generate
        seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        image = self._generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            params=params,
            seed=seed,
        )

        # Build metadata and explanation
        generation_time = time.time() - start_time
        peak_vram = (
            self.memory_manager.get_peak_vram_usage()
            if self.memory_manager
            else 0.0
        )

        metadata = GenerationMetadata(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            steps=params.num_steps,
            cfg_scale=params.guidance_scale,
            width=params.width,
            height=params.height,
            sampler=params.sampler_name,
            loras_used=[
                LoRAInfo(name=r.lora_name, alpha=r.suggested_alpha)
                for r in lora_recs
            ],
            generation_time_seconds=generation_time,
            peak_vram_gb=peak_vram,
            base_model_id=self.config.base_model,
            pipeline_type="intelligent",
        )

        explanation = self._build_explanation(
            analysis=recommendations.prompt_analysis,
            lora_recs=lora_recs,
            params=params,
            generation_time=generation_time,
        )

        return GenerationResult(
            id=generation_id,
            image=image,
            metadata=metadata,
            explanation=explanation,
        )

    def provide_feedback(
        self,
        generation_id: str,
        rating: int,
        comments: str = "",
        modified_params: Optional[ParameterModifications] = None,
    ):
        """
        Provide feedback for a generation to improve future recommendations.

        Args:
            generation_id: ID of the generation
            rating: Rating from 1-5
            comments: Optional comments
            modified_params: Parameters user modified (if any)
        """
        if not self.learning_engine:
            logger.warning("Learning engine not enabled - feedback ignored")
            return

        # Note: In production, we'd retrieve the original generation details
        # For now, this is a simplified interface

        feedback = GenerationFeedback(
            feedback_id=generation_id,
            timestamp=datetime.now().isoformat(),
            original_prompt="",  # Would be retrieved from generation log
            recommended_loras=[],  # Would be retrieved
            recommended_params={},  # Would be retrieved
            rating=rating,
            user_modified_params=modified_params.to_dict() if modified_params else None,
            notes=comments,
        )

        self.learning_engine.record_feedback(feedback)
        logger.info(f"Feedback recorded for {generation_id[:8]} (rating: {rating})")

    def _ensure_pipeline_loaded(
        self,
        base_model: str,
        loras: list[LoRARecommendationProtocol],
    ):
        """
        Ensure diffusion pipeline is loaded with correct model and LoRAs.

        Args:
            base_model: Base model ID
            loras: LoRA recommendations
        """
        # Load base model if different
        if self.current_base_model != base_model:
            logger.info(f"Loading base model: {base_model}")
            self._load_base_model(base_model)

        # Apply LoRAs
        if loras:
            logger.info(f"Applying {len(loras)} LoRAs")
            self._apply_loras(loras)

    def _load_base_model(self, model_id: str):
        """Load base diffusion model with EXTREME memory optimization - OUR MARKET VALUE."""
        logger.info(f"Loading {model_id} with extreme memory optimization...")
        dtype = torch.float16 if self.config.memory_settings.enable_quantization else torch.float32

        self.diffusion_pipeline = DiffusionPipeline.from_pretrained(
            model_id, torch_dtype=dtype, use_safetensors=True,
            variant="fp16" if dtype == torch.float16 else None
        )

        # APPLY ALL MEMORY OPTIMIZATIONS - OUR MARKET DIFFERENTIATOR
        if self.memory_optimizer:
            logger.info("Applying EXTREME memory optimizations (market differentiator)...")
            self.memory_optimizer.optimize_pipeline(self.diffusion_pipeline)
            # Cleanup immediately after load
            self.memory_optimizer.cleanup_after_model_load()
        else:
            # Fallback to basic optimizations if optimizer not available
            logger.warning("MemoryOptimizer not available, using basic optimizations")
            if self.config.memory_settings.offload_strategy.value == "balanced":
                self.diffusion_pipeline.enable_model_cpu_offload()
            elif self.config.memory_settings.offload_strategy.value == "aggressive":
                self.diffusion_pipeline.enable_sequential_cpu_offload()
            else:
                self.diffusion_pipeline = self.diffusion_pipeline.to("cuda")

        self.current_base_model = model_id
        logger.info(f"✅ Model loaded and optimized: {model_id}")

    def _apply_loras(self, lora_recs: list[LoRARecommendationProtocol]):
        """
        Apply LoRAs to the pipeline.

        Args:
            lora_recs: LoRA recommendations
        """
        if not self.diffusion_pipeline:
            logger.warning("No pipeline loaded - cannot apply LoRAs")
            return

        for rec in lora_recs:
            logger.debug(
                f"Applying LoRA: {rec.lora_name} (alpha={rec.suggested_alpha:.2f})"
            )
            try:
                # Real implementation using diffusers
                # Try to load LoRA weights if available
                lora_path = self._resolve_lora_path(rec.lora_name)
                if lora_path:
                    self.diffusion_pipeline.load_lora_weights(
                        lora_path,
                        adapter_name=rec.lora_name
                    )
                    # Set LoRA scale
                    self.diffusion_pipeline.set_adapters(
                        [rec.lora_name],
                        adapter_weights=[rec.suggested_alpha]
                    )
                    logger.info(f"✅ LoRA loaded: {rec.lora_name}")
                else:
                    logger.warning(f"LoRA path not found for: {rec.lora_name}")
            except Exception as e:
                logger.error(f"Failed to load LoRA {rec.lora_name}: {e}")
                # Continue with other LoRAs

    def _resolve_lora_path(self, lora_name: str) -> Optional[str]:
        """
        Resolve LoRA name to file path.

        Args:
            lora_name: LoRA name to resolve

        Returns:
            Path to LoRA file or None if not found
        """
        # Try to get from registry
        if self.registry:
            lora_info = self.registry.get_lora_by_name(lora_name)
            if lora_info and hasattr(lora_info, 'path'):
                return str(lora_info.path)

        # Could also check common locations
        # For now, return None if not in registry
        return None

    def _generate_image(
        self,
        prompt: str,
        negative_prompt: str,
        params: ParameterOptimizationProtocol,
        seed: int,
    ) -> Image.Image:
        """
        Generate image using diffusion pipeline with IMMEDIATE memory cleanup.

        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt
            params: Optimized parameters
            seed: Random seed

        Returns:
            PIL Image
        """
        if self.diffusion_pipeline is None:
            # Fallback to placeholder if no real pipeline
            logger.warning("No real pipeline loaded - returning placeholder")
            return Image.new("RGB", (params.width, params.height), color="gray")

        try:
            logger.debug(
                f"Generating with: steps={params.num_steps}, cfg={params.guidance_scale}, "
                f"size={params.width}x{params.height}, seed={seed}"
            )

            # Create generator for reproducibility
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
            generator.manual_seed(seed)

            # Generate image with memory monitoring (MARKET VALUE)
            if self.memory_optimizer:
                with MemoryMonitor(self.memory_optimizer) as monitor:
                    result = self.diffusion_pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=params.num_steps,
                        guidance_scale=params.guidance_scale,
                        width=params.width,
                        height=params.height,
                        generator=generator,
                    )
                    image = result.images[0]
                # Memory is automatically freed after exiting context (IMMEDIATE CLEANUP)
                logger.info(f"Peak memory during generation: {monitor.get_peak_memory():.2f}GB")
            else:
                result = self.diffusion_pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=params.num_steps,
                    guidance_scale=params.guidance_scale,
                    width=params.width,
                    height=params.height,
                    generator=generator,
                )
                image = result.images[0]

            return image

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            # Cleanup on error
            if self.memory_optimizer:
                self.memory_optimizer.cleanup_after_generation()
            return Image.new("RGB", (params.width, params.height), color="red")

    def _build_explanation(
        self,
        analysis: PromptAnalysisProtocol,
        lora_recs: list[LoRARecommendationProtocol],
        params: ParameterOptimizationProtocol,
        generation_time: Optional[float] = None,
    ) -> GenerationExplanation:
        """
        Build explanation of decisions made.

        Args:
            analysis: Prompt analysis
            lora_recs: LoRA recommendations
            params: Optimized parameters
            generation_time: Generation time in seconds (optional)

        Returns:
            GenerationExplanation
        """
        # Build summary
        summary_parts = []

        if lora_recs:
            lora_names = ", ".join(
                f"{r.lora_name} (α={r.suggested_alpha:.2f})" for r in lora_recs
            )
            summary_parts.append(f"LoRAs: {lora_names}")

        summary_parts.append(
            f"Params: {params.num_steps} steps, CFG {params.guidance_scale}, "
            f"{params.width}×{params.height}"
        )

        if hasattr(analysis, "complexity_category"):
            summary_parts.append(f"Complexity: {analysis.complexity_category.value}")

        summary = " | ".join(summary_parts)

        # LoRA reasoning
        lora_reasoning = {
            rec.lora_name: rec.reasoning for rec in lora_recs if rec.reasoning
        }

        # Parameter reasoning
        param_reasoning = {
            "steps": f"Set to {params.num_steps} based on prompt complexity",
            "cfg_scale": f"Set to {params.guidance_scale} for optimal guidance",
            "resolution": f"{params.width}×{params.height} based on content type",
        }

        # Performance notes
        performance_notes = []
        if generation_time:
            performance_notes.append(f"Generated in {generation_time:.2f}s")
        if hasattr(params, "estimated_vram_gb") and params.estimated_vram_gb:
            performance_notes.append(
                f"Estimated VRAM: {params.estimated_vram_gb:.2f}GB"
            )

        return GenerationExplanation(
            summary=summary,
            lora_reasoning=lora_reasoning,
            parameter_reasoning=param_reasoning,
            performance_notes=performance_notes,
        )

    def _apply_learning_adjustments(
        self, lora_recs: list[LoRARecommendationProtocol]
    ) -> list[LoRARecommendationProtocol]:
        """
        Apply learning engine adjustments to LoRA recommendations.

        Args:
            lora_recs: Original recommendations

        Returns:
            Adjusted recommendations
        """
        if not self.learning_engine:
            return lora_recs

        for rec in lora_recs:
            adjustment = self.learning_engine.get_lora_adjustment_factor(
                rec.lora_name
            )
            if adjustment != 1.0:
                rec.confidence_score *= adjustment
                logger.debug(
                    f"Learning adjustment for {rec.lora_name}: {adjustment:.2f}x"
                )

        return lora_recs
