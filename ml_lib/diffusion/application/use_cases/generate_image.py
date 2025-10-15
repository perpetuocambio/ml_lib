"""Generate Image Use Case - Application Layer.

This is the Application Layer orchestrator that coordinates domain services
to fulfill the "generate image" use case.

Responsibilities:
- Coordinate domain services
- Handle transaction boundaries
- Convert between DTOs and domain objects
- No business logic (that's in domain)

This replaces the massive IntelligentGenerationPipeline god class.
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from PIL import Image

from ml_lib.diffusion.domain.services.lora_recommendation_service import (
    LoRARecommendationService,
)
from ml_lib.diffusion.domain.interfaces.prompt_analyzer import IPromptAnalyzer
from ml_lib.diffusion.domain.interfaces.resource_monitor import IResourceMonitor
from ml_lib.diffusion.domain.entities.lora import LoRARecommendation


@dataclass
class GenerateImageRequest:
    """Input DTO for image generation."""

    prompt: str
    negative_prompt: str = ""
    base_model: str = "SDXL"
    seed: Optional[int] = None
    num_steps: Optional[int] = None  # None = auto-optimize
    cfg_scale: Optional[float] = None  # None = use default for model
    width: Optional[int] = None  # None = auto-optimize
    height: Optional[int] = None  # None = auto-optimize
    max_loras: int = 3
    min_lora_confidence: float = 0.5


@dataclass
class GenerateImageResult:
    """Output DTO for image generation."""

    image: Image.Image
    prompt_used: str
    negative_prompt_used: str
    seed: int
    loras_applied: list[str]  # Names of LoRAs used
    generation_time_seconds: float
    peak_vram_gb: float
    explanation: str  # Human-readable explanation of decisions


class GenerateImageUseCase:
    """
    Use case for generating an image.

    Clean separation of concerns:
    - Application layer (this): Orchestrates
    - Domain layer (services/entities): Business logic
    - Infrastructure layer: DB, API, filesystem access
    """

    def __init__(
        self,
        lora_service: LoRARecommendationService,
        prompt_analyzer: IPromptAnalyzer,
        resource_monitor: IResourceMonitor,
        # More services will be injected as we migrate
    ):
        """
        Initialize use case with dependencies.

        Args:
            lora_service: LoRA recommendation domain service
            prompt_analyzer: Prompt analysis service
            resource_monitor: System resource monitoring
        """
        self.lora_service = lora_service
        self.prompt_analyzer = prompt_analyzer
        self.resource_monitor = resource_monitor

    def execute(self, request: GenerateImageRequest) -> GenerateImageResult:
        """
        Execute the use case.

        Args:
            request: Generation request DTO

        Returns:
            Generation result DTO

        This method orchestrates the workflow without containing business logic.
        All business logic is in domain services and entities.
        """
        import time
        import random

        start_time = time.time()

        # Step 1: Analyze prompt (delegates to domain)
        analysis = self.prompt_analyzer.analyze(request.prompt)

        # Step 2: Get LoRA recommendations (delegates to domain service)
        lora_recommendations = self.lora_service.recommend(
            prompt=request.prompt,
            base_model=request.base_model,
            max_loras=request.max_loras,
            min_confidence=request.min_lora_confidence,
        )

        # Step 3: Check system resources
        resources = self.resource_monitor.get_current_stats()

        # Step 4: Determine parameters
        # (In full implementation, this would delegate to ParameterOptimizer domain service)
        seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
        steps = request.num_steps or 30  # TODO: Use ParameterOptimizer
        cfg = request.cfg_scale or 7.0  # TODO: Use model-specific defaults
        width = request.width or 1024
        height = request.height or 1024

        # Step 5: Generate image
        # (In full implementation, this would delegate to DiffusionBackend infrastructure)
        # For now, placeholder
        image = self._generate_placeholder(width, height)

        # Step 6: Build explanation
        explanation = self._build_explanation(
            lora_recommendations,
            analysis,
            resources,
        )

        # Step 7: Calculate metrics
        generation_time = time.time() - start_time

        # Return DTO
        return GenerateImageResult(
            image=image,
            prompt_used=request.prompt,
            negative_prompt_used=request.negative_prompt,
            seed=seed,
            loras_applied=[rec.lora.name for rec in lora_recommendations],
            generation_time_seconds=generation_time,
            peak_vram_gb=resources.used_vram_gb,
            explanation=explanation,
        )

    def _generate_placeholder(self, width: int, height: int) -> Image.Image:
        """
        Placeholder for actual image generation.

        In full implementation, this would delegate to a DiffusionBackend
        infrastructure service.
        """
        return Image.new("RGB", (width, height), color="gray")

    def _build_explanation(
        self,
        loras: list[LoRARecommendation],
        analysis: any,  # PromptAnalysis when migrated
        resources: any,  # ResourceStats
    ) -> str:
        """
        Build human-readable explanation.

        Args:
            loras: LoRA recommendations
            analysis: Prompt analysis
            resources: System resources

        Returns:
            Explanation string
        """
        parts = []

        # LoRA selection reasoning
        if loras:
            lora_names = ", ".join(
                f"{r.lora.name} ({r.confidence})" for r in loras
            )
            parts.append(f"LoRAs selected: {lora_names}")

            # Include reasoning
            for rec in loras:
                parts.append(f"  - {rec.lora.name}: {rec.reasoning}")
        else:
            parts.append("No LoRAs recommended for this prompt")

        # Resource usage
        if resources.has_gpu:
            parts.append(
                f"GPU VRAM: {resources.used_vram_gb:.1f}GB / {resources.total_vram_gb:.1f}GB"
            )

        return "\n".join(parts)
