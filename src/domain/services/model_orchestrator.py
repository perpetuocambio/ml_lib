"""
Model Orchestrator - Intelligent model selection and orchestration.

Uses SQLite database for model metadata to automatically:
- Select optimal base model for prompt
- Choose compatible VAE, encoders, LoRAs
- Configure optimal generation parameters
- Manage memory constraints

Completely abstracted from user - they only provide prompt + simple options.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ml_lib.diffusion.domain.value_objects_models import BaseModel, ModelType
from ml_lib.system.resource_monitor import ResourceMonitor
from ml_lib.diffusion.infrastructure.storage.metadata_db import MetadataDatabase

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadataFile:
    """
    Parsed metadata from ComfyUI .metadata.json files.

    These files are created by custom_nodes like civitai_comfy_nodes.
    """

    # Identity
    file_name: str
    model_name: str
    file_path: Path

    # Model info
    base_model: str  # "SDXL 1.0", "SD 1.5", "Flux", etc.
    model_type: str  # "LORA", "Checkpoint", etc.

    # Quality indicators
    download_count: int = 0
    rating: float = 0.0
    thumbs_up: int = 0

    # Training info
    trigger_words: list[str] = field(default_factory=list)
    trained_words: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    # Optimal generation parameters (from example images)
    recommended_steps: Optional[int] = None
    recommended_cfg: Optional[float] = None
    recommended_sampler: Optional[str] = None
    recommended_scheduler: Optional[str] = None
    recommended_clip_skip: Optional[int] = None
    recommended_lora_weight: float = 1.0

    # Technical
    size_bytes: int = 0
    sha256: str = ""

    def __post_init__(self):
        """Convert path to Path object."""
        if isinstance(self.file_path, str):
            self.file_path = Path(self.file_path)

    @property
    def size_gb(self) -> float:
        """Size in GB."""
        return self.size_bytes / (1024**3)

    @property
    def popularity_score(self) -> float:
        """
        Calculate popularity score (0-100).

        Combines downloads, rating, thumbs up.
        """
        # Normalize downloads (log scale, max ~100k)
        import math
        download_score = min(math.log10(max(self.download_count, 1)) / 5, 1.0) * 50

        # Rating (0-5 scale)
        rating_score = (self.rating / 5.0) * 30

        # Thumbs up (log scale, max ~1k)
        thumbs_score = min(math.log10(max(self.thumbs_up, 1)) / 3, 1.0) * 20

        return download_score + rating_score + thumbs_score

    def get_base_model_enum(self) -> BaseModel:
        """Convert string base_model to enum."""
        base_lower = self.base_model.lower()

        if "sdxl" in base_lower or "xl" in base_lower:
            if "pony" in base_lower:
                return BaseModel.PONY
            return BaseModel.SDXL
        elif "sd 3" in base_lower or "sd3" in base_lower:
            return BaseModel.SD3
        elif "flux" in base_lower:
            return BaseModel.FLUX
        elif "sd 2.1" in base_lower or "sd21" in base_lower:
            return BaseModel.SD21
        elif "sd 2" in base_lower or "sd20" in base_lower:
            return BaseModel.SD20
        elif "sd 1.5" in base_lower or "sd15" in base_lower or "sd 1" in base_lower:
            return BaseModel.SD15
        else:
            return BaseModel.UNKNOWN

    @classmethod
    def from_json_file(cls, json_path: Path) -> Optional["ModelMetadataFile"]:
        """
        Load metadata from .metadata.json file.

        Args:
            json_path: Path to .metadata.json file

        Returns:
            Parsed metadata or None if failed
        """
        try:
            with open(json_path) as f:
                data = json.load(f)

            # Extract basic info
            file_name = data.get("file_name", "")
            model_name = data.get("model_name", "")
            file_path = Path(data.get("file_path", ""))
            base_model = data.get("base_model", "unknown")
            size_bytes = data.get("size", 0)
            sha256 = data.get("sha256", "")

            # Extract CivitAI info if available
            civitai = data.get("civitai") or {}

            # Stats
            stats = civitai.get("stats", {})
            download_count = stats.get("downloadCount", 0)
            rating = stats.get("rating", 0.0)
            thumbs_up = stats.get("thumbsUpCount", 0)

            # Trigger words
            trained_words = civitai.get("trainedWords", [])

            # Tags
            tags = data.get("tags", [])

            # Model type
            model_info = civitai.get("model", {})
            model_type = model_info.get("type", "unknown")

            # Extract recommended parameters from first image metadata
            images = civitai.get("images", [])
            recommended_steps = None
            recommended_cfg = None
            recommended_sampler = None
            recommended_scheduler = None
            recommended_clip_skip = None
            recommended_lora_weight = 1.0

            if images:
                meta = images[0].get("meta", {})
                recommended_steps = meta.get("steps")
                recommended_cfg = meta.get("cfgScale")
                recommended_sampler = meta.get("sampler")
                recommended_scheduler = meta.get("Scheduler")
                recommended_clip_skip = meta.get("clipSkip")

                # Extract LoRA weight if available
                lora_weights = meta.get("Lora weights", {})
                if lora_weights:
                    # Get first LoRA weight
                    recommended_lora_weight = float(list(lora_weights.values())[0])

            return cls(
                file_name=file_name,
                model_name=model_name,
                file_path=file_path,
                base_model=base_model,
                model_type=model_type,
                download_count=download_count,
                rating=rating,
                thumbs_up=thumbs_up,
                trigger_words=trained_words,
                trained_words=trained_words,
                tags=tags,
                recommended_steps=recommended_steps,
                recommended_cfg=recommended_cfg,
                recommended_sampler=recommended_sampler,
                recommended_scheduler=recommended_scheduler,
                recommended_clip_skip=recommended_clip_skip,
                recommended_lora_weight=recommended_lora_weight,
                size_bytes=size_bytes,
                sha256=sha256,
            )

        except Exception as e:
            logger.warning(f"Failed to parse {json_path}: {e}")
            return None


@dataclass
class DiffusionArchitecture:
    """
    Information about a diffusion model architecture.

    Defines requirements and compatibility for different architectures.
    """

    name: BaseModel

    # Components required
    requires_vae: bool = True
    requires_text_encoder: bool = True
    requires_text_encoder_2: bool = False  # SDXL, Flux
    requires_unet: bool = True

    # Typical sizes (for memory estimation)
    typical_base_size_gb: float = 2.0
    typical_vae_size_gb: float = 0.1
    typical_encoder_size_gb: float = 0.5

    # Compatible VAE types
    compatible_vae_patterns: list[str] = field(default_factory=list)

    # Default parameters
    default_steps: int = 30
    default_cfg: float = 7.0
    default_sampler: str = "DPM++ 2M"
    default_scheduler: str = "karras"
    default_clip_skip: int = 2

    @classmethod
    def get_architecture(cls, base_model: BaseModel) -> "DiffusionArchitecture":
        """Get architecture info for base model type."""
        architectures = {
            BaseModel.SD15: cls(
                name=BaseModel.SD15,
                requires_vae=True,
                requires_text_encoder=True,
                requires_text_encoder_2=False,
                requires_unet=True,
                typical_base_size_gb=2.0,
                typical_vae_size_gb=0.1,
                typical_encoder_size_gb=0.5,
                compatible_vae_patterns=["sd15", "sd-vae", "kl-f8"],
                default_steps=25,
                default_cfg=7.5,
                default_sampler="DPM++ 2M Karras",
                default_clip_skip=1,
            ),
            BaseModel.SDXL: cls(
                name=BaseModel.SDXL,
                requires_vae=True,
                requires_text_encoder=True,
                requires_text_encoder_2=True,  # SDXL has dual encoders
                requires_unet=True,
                typical_base_size_gb=6.5,
                typical_vae_size_gb=0.2,
                typical_encoder_size_gb=1.2,
                compatible_vae_patterns=["sdxl", "xl-vae"],
                default_steps=30,
                default_cfg=7.0,
                default_sampler="DPM++ 2M SDE Karras",
                default_scheduler="karras",
                default_clip_skip=2,
            ),
            BaseModel.PONY: cls(
                name=BaseModel.PONY,
                requires_vae=True,
                requires_text_encoder=True,
                requires_text_encoder_2=True,
                requires_unet=True,
                typical_base_size_gb=6.5,
                typical_vae_size_gb=0.2,
                typical_encoder_size_gb=1.2,
                compatible_vae_patterns=["sdxl", "pony", "xl-vae"],
                default_steps=30,
                default_cfg=6.0,
                default_sampler="Euler a",
                default_clip_skip=2,
            ),
            BaseModel.SD3: cls(
                name=BaseModel.SD3,
                requires_vae=True,
                requires_text_encoder=True,
                requires_text_encoder_2=True,
                requires_unet=True,
                typical_base_size_gb=8.0,
                typical_vae_size_gb=0.3,
                typical_encoder_size_gb=1.5,
                compatible_vae_patterns=["sd3"],
                default_steps=28,
                default_cfg=5.0,
                default_sampler="DPM++ 2M",
            ),
            BaseModel.FLUX: cls(
                name=BaseModel.FLUX,
                requires_vae=True,
                requires_text_encoder=True,
                requires_text_encoder_2=True,
                requires_unet=True,
                typical_base_size_gb=12.0,
                typical_vae_size_gb=0.3,
                typical_encoder_size_gb=2.0,
                compatible_vae_patterns=["flux"],
                default_steps=20,
                default_cfg=3.5,
                default_sampler="Euler",
            ),
        }

        return architectures.get(base_model, architectures[BaseModel.SDXL])


class ModelOrchestrator:
    """
    Intelligent model orchestrator.

    Automatically selects and configures models based on:
    - Available models with metadata
    - User prompt (semantic analysis via Ollama if enabled)
    - Available system resources
    - Model compatibility

    User provides: prompt + simple options
    Orchestrator handles: ALL technical details

    Example:
        >>> orchestrator = ModelOrchestrator(
        ...     model_paths=["/path/to/comfyui/models"],
        ...     enable_ollama=True
        ... )
        >>>
        >>> config = orchestrator.select_models(
        ...     prompt="a beautiful anime girl with pink hair",
        ...     style="anime",  # Optional hint
        ... )
        >>>
        >>> print(f"Base: {config.base_model_path}")
        >>> print(f"LoRAs: {len(config.loras)} selected")
        >>> print(f"Steps: {config.generation_params['steps']}")
    """

    def __init__(
        self,
        enable_ollama: bool = False,
        ollama_model: str = "llama3.2",
        resource_monitor: Optional[ResourceMonitor] = None,
        db_path: Optional[Path | str] = None,
    ):
        """
        Initialize orchestrator.

        Args:
            enable_ollama: Enable intelligent selection via Ollama
            ollama_model: Ollama model to use for analysis
            resource_monitor: Resource monitor (None = create new)
            db_path: Path to SQLite database (None = use default)
        """
        self.enable_ollama = enable_ollama
        self.ollama_model = ollama_model
        self.resource_monitor = resource_monitor or ResourceMonitor()

        # Initialize database
        self.db = MetadataDatabase(db_path) if db_path else MetadataDatabase()

        # Cache for quick access
        self.metadata_index: dict[ModelType, list[ModelMetadataFile]] = {}

        # Load models from database
        self._load_from_database()

        total_models = sum(len(m) for m in self.metadata_index.values())

        # If database is empty, try to auto-populate from ComfyUI
        if total_models == 0:
            logger.warning("Database empty, attempting auto-population from ComfyUI...")
            self._auto_populate_database()
            # Reload after population
            self._load_from_database()
            total_models = sum(len(m) for m in self.metadata_index.values())

        logger.info(f"ModelOrchestrator initialized: {total_models} models indexed")

    def _load_from_database(self) -> None:
        """Load models from SQLite database into cache."""
        # Get database stats
        stats = self.db.get_stats()

        if stats.get("total", 0) == 0:
            logger.warning("Database is empty - run migration first")
            return

        # Load models by type
        for model_type in ModelType:
            try:
                db_models = self.db.get_all_models_by_type(model_type)

                if db_models:
                    # Convert ModelMetadata to ModelMetadataFile
                    self.metadata_index[model_type] = [
                        self._convert_metadata_to_file(m) for m in db_models
                    ]

                    logger.debug(f"Loaded {len(db_models)} {model_type.value} models from database")

            except Exception as e:
                logger.error(f"Failed to load {model_type.value} models: {e}")

    def _auto_populate_database(self) -> None:
        """
        Auto-populate database from ComfyUI installation.

        This runs automatically if database is empty on first use.
        """
        try:
            from ml_lib.diffusion.infrastructure.config import detect_comfyui_installation
            from ml_lib.diffusion.infrastructure.storage.comfyui_migrator import ComfyUIMetadataMigrator

            # Detect ComfyUI
            comfyui_root = detect_comfyui_installation()
            if not comfyui_root:
                logger.warning("ComfyUI not found, cannot auto-populate database")
                return

            logger.info(f"Auto-populating database from ComfyUI: {comfyui_root}")

            # Run migration
            migrator = ComfyUIMetadataMigrator(self.db)
            results = migrator.migrate_comfyui_installation(comfyui_root)

            # Log results
            total_success = sum(s for s, _ in results.values())
            total_failed = sum(f for _, f in results.values())

            logger.info(f"Auto-population complete: {total_success} models added, {total_failed} failed")

        except Exception as e:
            logger.error(f"Auto-population failed: {e}")

    def _convert_metadata_to_file(self, metadata) -> ModelMetadataFile:
        """Convert ModelMetadata from database to ModelMetadataFile."""
        return ModelMetadataFile(
            file_name=metadata.local_path.name if metadata.local_path else metadata.name,
            model_name=metadata.name,
            file_path=metadata.local_path or Path(""),
            base_model=metadata.base_model.value,
            model_type=metadata.type.value,
            download_count=metadata.download_count,
            rating=metadata.rating,
            thumbs_up=0,  # Not stored in new schema
            trigger_words=metadata.trigger_words,
            trained_words=metadata.trigger_words,
            tags=metadata.tags,
            recommended_steps=None,  # TODO: Extract from metadata
            recommended_cfg=None,
            recommended_sampler=None,
            recommended_scheduler=None,
            recommended_clip_skip=None,
            recommended_lora_weight=metadata.recommended_weight or 1.0,
            size_bytes=metadata.size_bytes,
            sha256=metadata.sha256,
        )

    def get_stats(self) -> dict[str, any]:
        """Get orchestrator statistics."""
        db_stats = self.db.get_stats()
        return {
            "total_models": db_stats.get("total", 0),
            "local_models": db_stats.get("local_models", 0),
            "by_type": db_stats.get("by_type", {}),
            "by_base_model": db_stats.get("by_base_model", {}),
            "ollama_enabled": self.enable_ollama,
        }

    def select_best_model(
        self,
        base_model: BaseModel,
        style_hint: Optional[str] = None,
        min_popularity: float = 0.0,
    ) -> Optional[ModelMetadataFile]:
        """
        Select best base model from available models.

        Args:
            base_model: Target base model architecture
            style_hint: Optional style hint (e.g., "anime", "realistic")
            min_popularity: Minimum popularity score (0-100)

        Returns:
            Best matching model or None if not found
        """
        checkpoints = self.metadata_index.get(ModelType.BASE_MODEL, [])

        # Filter by base model
        candidates = [
            m for m in checkpoints
            if m.get_base_model_enum() == base_model
            and m.popularity_score >= min_popularity
        ]

        if not candidates:
            return None

        # Sort by popularity and pick best
        candidates.sort(key=lambda m: m.popularity_score, reverse=True)

        return candidates[0]

    def select_compatible_loras(
        self,
        base_model: BaseModel,
        prompt: str,
        max_loras: int = 3,
        min_confidence: float = 0.5,
    ) -> list[ModelMetadataFile]:
        """
        Select compatible LoRAs for prompt.

        Args:
            base_model: Base model architecture
            prompt: User prompt
            max_loras: Maximum LoRAs to select
            min_confidence: Minimum confidence score

        Returns:
            List of selected LoRAs
        """
        loras = self.metadata_index.get(ModelType.LORA, [])

        # Filter by base model compatibility
        compatible = [
            lora for lora in loras
            if lora.get_base_model_enum() == base_model
        ]

        # Score based on trigger words and tags
        prompt_lower = prompt.lower()
        scored_loras = []

        for lora in compatible:
            score = 0.0

            # Check trigger words
            for trigger in lora.trigger_words:
                if trigger.lower() in prompt_lower:
                    score += 0.3

            # Check tags
            for tag in lora.tags:
                if tag.lower() in prompt_lower:
                    score += 0.1

            # Add popularity bonus
            score += lora.popularity_score / 1000  # Small boost

            if score >= min_confidence:
                scored_loras.append((lora, score))

        # Sort by score and return top N
        scored_loras.sort(key=lambda x: x[1], reverse=True)
        return [lora for lora, _ in scored_loras[:max_loras]]
