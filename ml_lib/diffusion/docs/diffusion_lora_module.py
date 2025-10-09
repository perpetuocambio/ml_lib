# ============================================================================
# INTERFACES
# ============================================================================

from abc import ABC, abstractmethod
from typing import Protocol, TypeVar, Generic, Any
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

T = TypeVar("T")


# diffusion/interfaces/lora_interface.py
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


# diffusion/interfaces/pipeline_interface.py
class DiffusionPipelineInterface(ABC, Generic[T]):
    """Interface para pipelines de difusión."""

    @abstractmethod
    def load_checkpoint(self, path: Path) -> "DiffusionPipelineInterface":
        """Carga un checkpoint base."""
        pass

    @abstractmethod
    def add_lora(
        self, lora_path: Path, alpha: float = 1.0, adapter_name: str | None = None
    ) -> "DiffusionPipelineInterface":
        """Añade un LoRA al pipeline."""
        pass

    @abstractmethod
    def merge_loras(
        self, adapter_names: list[str], adapter_weights: list[float]
    ) -> "DiffusionPipelineInterface":
        """Fusiona múltiples LoRAs."""
        pass

    @abstractmethod
    def generate(self, prompt: str, negative_prompt: str | None = None, **kwargs) -> T:
        """Genera imagen(es) basado en prompt."""
        pass

    @abstractmethod
    def remove_lora(self, adapter_name: str) -> None:
        """Elimina un LoRA del pipeline."""
        pass


# diffusion/interfaces/scheduler_interface.py
class SchedulerInterface(ABC):
    """Interface para schedulers de difusión."""

    @abstractmethod
    def set_timesteps(self, num_steps: int) -> None:
        """Configura los timesteps del proceso."""
        pass

    @abstractmethod
    def step(
        self, noise_pred: np.ndarray, timestep: int, latents: np.ndarray
    ) -> np.ndarray:
        """Realiza un paso del proceso de denoising."""
        pass

    @abstractmethod
    def add_noise(
        self, original: np.ndarray, noise: np.ndarray, timestep: int
    ) -> np.ndarray:
        """Añade ruido según el timestep."""
        pass


# ============================================================================
# MODELS
# ============================================================================


# diffusion/models/lora_weights.py
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


# diffusion/models/pipeline_config.py
@dataclass
class DiffusionPipelineConfig:
    """Configuración completa de pipeline de difusión."""

    model_type: str  # "sd15", "sdxl", "pony", "sd3", etc.
    checkpoint_path: Path
    vae_path: Path | None = None
    text_encoder_path: Path | None = None
    scheduler_type: str = "ddpm"
    safety_checker: bool = False
    torch_dtype: str = "float16"
    device: str = "cuda"

    # LoRA configuration
    lora_configs: list["LoRAConfig"] = field(default_factory=list)
    default_lora_alpha: float = 1.0

    # Generation defaults
    default_steps: int = 30
    default_guidance_scale: float = 7.5
    default_width: int = 512
    default_height: int = 512


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


# diffusion/models/generation_params.py
@dataclass
class GenerationParams:
    """Parámetros para generación de imágenes."""

    prompt: str
    negative_prompt: str = ""
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512
    num_images: int = 1
    seed: int | None = None

    # Advanced sampling
    eta: float = 0.0  # DDIM eta parameter
    clip_skip: int = 0

    # LoRA control per-generation
    active_loras: dict[str, float] | None = None  # {adapter_name: alpha}


# ============================================================================
# SERVICES
# ============================================================================


# diffusion/services/lora_service.py
class LoRAService:
    """Servicio principal para gestión de LoRAs."""

    def __init__(
        self,
        safetensors_handler: "SafeTensorsHandler",
        weight_injection_handler: "WeightInjectionHandler",
    ):
        self.safetensors_handler = safetensors_handler
        self.weight_injection = weight_injection_handler
        self._loaded_loras: dict[str, LoRAWeights] = {}

    def load_lora(
        self, lora_path: Path, adapter_name: str, config: LoRAConfig
    ) -> LoRAWeights:
        """Carga un LoRA desde archivo."""
        # Cargar tensors del archivo
        tensors = self.safetensors_handler.load(lora_path)

        # Extraer matrices up/down y metadata
        lora_weights = self._parse_lora_tensors(tensors, adapter_name, config)

        self._loaded_loras[adapter_name] = lora_weights
        return lora_weights

    def _parse_lora_tensors(
        self, tensors: dict[str, np.ndarray], adapter_name: str, config: LoRAConfig
    ) -> LoRAWeights:
        """Parsea tensors en estructura LoRAWeights."""
        lora_up = {}
        lora_down = {}
        target_modules = set()

        # Detectar automáticamente target modules y rank
        rank = None
        alpha = config.alpha

        for key, tensor in tensors.items():
            # Patrones comunes: "lora_unet_down_blocks_0_attentions_0_up"
            if "lora" in key.lower():
                module_name = self._extract_module_name(key)
                target_modules.add(module_name)

                if "up" in key or "alpha" in key:
                    continue

                if "down" in key:
                    lora_down[module_name] = tensor
                    if rank is None and tensor.ndim >= 2:
                        rank = tensor.shape[1]  # Rank es la dim compartida
                else:
                    lora_up[module_name] = tensor

        # Buscar alpha en metadata si existe
        if "alpha" in tensors:
            alpha = float(tensors["alpha"])

        return LoRAWeights(
            adapter_name=adapter_name,
            rank=rank or 4,  # Default rank
            alpha=alpha,
            target_modules=list(target_modules),
            lora_up=lora_up,
            lora_down=lora_down,
            metadata={"source": str(config.lora_path)},
        )

    def _extract_module_name(self, tensor_key: str) -> str:
        """Extrae el nombre del módulo de la key del tensor."""
        # Ejemplo: "lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight"
        # -> "down_blocks.0.attentions.0.proj_in"
        parts = tensor_key.split(".")
        # Lógica de parsing específica por convención
        return ".".join(parts[1:-2])  # Simplificado

    def merge_loras(
        self, adapter_names: list[str], adapter_weights: list[float]
    ) -> LoRAWeights:
        """Fusiona múltiples LoRAs con pesos dados."""
        if len(adapter_names) != len(adapter_weights):
            raise ValueError("adapter_names and adapter_weights must match")

        # Normalizar pesos si es necesario
        total_weight = sum(adapter_weights)
        normalized_weights = [w / total_weight for w in adapter_weights]

        # Comenzar con el primer LoRA
        merged = self._loaded_loras[adapter_names[0]]

        # Fusionar secuencialmente
        for name, weight in zip(adapter_names[1:], normalized_weights[1:]):
            lora = self._loaded_loras[name]
            prev_weight = sum(normalized_weights[: adapter_names.index(name) + 1])

            merged = merged.merge_with(
                lora, weight_self=prev_weight, weight_other=weight
            )

        return merged

    def get_lora(self, adapter_name: str) -> LoRAWeights | None:
        """Obtiene un LoRA cargado."""
        return self._loaded_loras.get(adapter_name)

    def list_loaded_loras(self) -> list[str]:
        """Lista nombres de LoRAs cargados."""
        return list(self._loaded_loras.keys())

    def unload_lora(self, adapter_name: str) -> None:
        """Descarga un LoRA de memoria."""
        if adapter_name in self._loaded_loras:
            del self._loaded_loras[adapter_name]


# diffusion/services/pipeline_service.py
class DiffusionPipelineService:
    """Servicio de gestión de pipelines de difusión."""

    def __init__(
        self,
        lora_service: LoRAService,
        checkpoint_service: "CheckpointService",
        unet_service: "UNetService",
        vae_service: "VAEService",
        conditioning_service: "ConditioningService",
        sampling_service: "SamplingService",
    ):
        self.lora_service = lora_service
        self.checkpoint_service = checkpoint_service
        self.unet_service = unet_service
        self.vae_service = vae_service
        self.conditioning_service = conditioning_service
        self.sampling_service = sampling_service

        self._active_adapters: dict[str, float] = {}
        self._config: DiffusionPipelineConfig | None = None

    def initialize(self, config: DiffusionPipelineConfig) -> None:
        """Inicializa el pipeline con configuración."""
        self._config = config

        # Cargar checkpoint base
        checkpoint = self.checkpoint_service.load(config.checkpoint_path)

        # Inicializar componentes
        self.unet_service.load_from_checkpoint(checkpoint)
        self.vae_service.load_from_checkpoint(checkpoint)
        self.conditioning_service.load_from_checkpoint(checkpoint)

        # Cargar LoRAs si están en config
        for lora_config in config.lora_configs:
            self.add_lora(lora_config)

    def add_lora(self, lora_config: LoRAConfig) -> None:
        """Añade un LoRA al pipeline."""
        # Cargar LoRA
        lora_weights = self.lora_service.load_lora(
            lora_config.lora_path, lora_config.adapter_name, lora_config
        )

        # Inyectar en UNet y/o Text Encoder
        if lora_config.apply_to_unet:
            self.unet_service.inject_lora(lora_weights, alpha=lora_config.alpha)

        if lora_config.apply_to_text_encoder:
            self.conditioning_service.inject_lora(lora_weights, alpha=lora_config.alpha)

        # Registrar como activo
        self._active_adapters[lora_config.adapter_name] = lora_config.alpha

    def set_lora_scale(self, adapter_name: str, alpha: float) -> None:
        """Cambia la escala de un LoRA activo."""
        if adapter_name not in self._active_adapters:
            raise ValueError(f"LoRA {adapter_name} not active")

        self._active_adapters[adapter_name] = alpha

        # Reinyectar con nueva escala
        lora_weights = self.lora_service.get_lora(adapter_name)
        if lora_weights:
            self.unet_service.inject_lora(lora_weights, alpha=alpha)

    def generate(self, params: GenerationParams) -> np.ndarray:
        """Genera imagen(es) basado en parámetros."""
        # Override LoRA scales si se especifican
        if params.active_loras:
            for name, alpha in params.active_loras.items():
                self.set_lora_scale(name, alpha)

        # Encoding del prompt
        text_embeddings = self.conditioning_service.encode_prompt(
            params.prompt, params.negative_prompt
        )

        # Inicializar latents
        latents = self._initialize_latents(
            params.width, params.height, params.num_images, params.seed
        )

        # Configurar scheduler
        self.sampling_service.set_timesteps(params.num_inference_steps)

        # Denoising loop
        for timestep in self.sampling_service.timesteps:
            # Predecir ruido
            noise_pred = self.unet_service.predict_noise(
                latents, timestep, text_embeddings, guidance_scale=params.guidance_scale
            )

            # Aplicar step del scheduler
            latents = self.sampling_service.step(noise_pred, timestep, latents)

        # Decodificar latents a imagen
        images = self.vae_service.decode(latents)

        return images

    def _initialize_latents(
        self, width: int, height: int, batch_size: int, seed: int | None
    ) -> np.ndarray:
        """Inicializa latents aleatorios."""
        latent_height = height // 8
        latent_width = width // 8

        if seed is not None:
            np.random.seed(seed)

        return np.random.randn(
            batch_size,
            4,  # Latent channels
            latent_height,
            latent_width,
        ).astype(np.float32)


# ============================================================================
# HANDLERS
# ============================================================================


# diffusion/handlers/lora_merge_handler.py
class LoRAMergeHandler:
    """Handler especializado en fusión de LoRAs."""

    def __init__(self):
        self._merge_strategies = {
            "weighted_sum": self._weighted_sum_merge,
            "concatenate": self._concatenate_merge,
            "svd_merge": self._svd_merge,
        }

    def merge(
        self,
        loras: list[LoRAWeights],
        weights: list[float],
        strategy: str = "weighted_sum",
    ) -> LoRAWeights:
        """Fusiona LoRAs usando estrategia especificada."""
        if strategy not in self._merge_strategies:
            raise ValueError(f"Unknown merge strategy: {strategy}")

        return self._merge_strategies[strategy](loras, weights)

    def _weighted_sum_merge(
        self, loras: list[LoRAWeights], weights: list[float]
    ) -> LoRAWeights:
        """Fusión por suma ponderada de matrices."""
        # Implementación simplificada
        base = loras[0]
        for lora, weight in zip(loras[1:], weights[1:]):
            base = base.merge_with(lora, weight_self=weights[0], weight_other=weight)
        return base

    def _concatenate_merge(
        self, loras: list[LoRAWeights], weights: list[float]
    ) -> LoRAWeights:
        """Fusión por concatenación (aumenta rank)."""
        # Concatenar matrices up/down para aumentar capacidad
        raise NotImplementedError("Concatenate merge not yet implemented")

    def _svd_merge(self, loras: list[LoRAWeights], weights: list[float]) -> LoRAWeights:
        """Fusión usando SVD para mantener rank bajo."""
        raise NotImplementedError("SVD merge not yet implemented")


# diffusion/handlers/weight_injection_handler.py
class WeightInjectionHandler:
    """Handler para inyección de pesos LoRA en capas."""

    def inject_into_layer(
        self,
        layer_weights: np.ndarray,
        lora_up: np.ndarray,
        lora_down: np.ndarray,
        alpha: float,
        rank: int,
    ) -> np.ndarray:
        """Inyecta pesos LoRA en una capa específica."""
        scaling = alpha / rank
        delta_weight = scaling * (lora_up @ lora_down)
        return layer_weights + delta_weight

    def extract_from_layer(
        self, modified_weights: np.ndarray, original_weights: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extrae LoRA de pesos modificados (inverso)."""
        delta = modified_weights - original_weights
        # Aplicar SVD para descomponer en low-rank
        u, s, vh = np.linalg.svd(delta, full_matrices=False)
        # Retornar componentes up/down
        return u[:, :4] @ np.diag(s[:4]), vh[:4, :]


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

"""
# Crear pipeline
from ml_library.diffusion import DiffusionPipelineService, DiffusionPipelineConfig, LoRAConfig

config = DiffusionPipelineConfig(
    model_type="sdxl",
    checkpoint_path=Path("models/sdxl_base.safetensors"),
    lora_configs=[
        LoRAConfig(
            adapter_name="style_lora",
            lora_path=Path("loras/anime_style.safetensors"),
            alpha=0.8
        ),
        LoRAConfig(
            adapter_name="character_lora",
            lora_path=Path("loras/character.safetensors"),
            alpha=1.0
        )
    ]
)

# Inicializar servicios (con DI)
pipeline_service = DiffusionPipelineService(...)
pipeline_service.initialize(config)

# Generar con múltiples LoRAs
from ml_library.diffusion import GenerationParams

params = GenerationParams(
    prompt="1girl, anime style, masterpiece",
    negative_prompt="low quality, blurry",
    num_inference_steps=30,
    guidance_scale=7.5,
    width=1024,
    height=1024,
    active_loras={
        "style_lora": 0.7,      # Reducir intensidad del estilo
        "character_lora": 1.2   # Aumentar influencia del personaje
    }
)

images = pipeline_service.generate(params)

# Fusionar LoRAs permanentemente
merged_lora = pipeline_service.lora_service.merge_loras(
    adapter_names=["style_lora", "character_lora"],
    adapter_weights=[0.6, 0.4]
)

# Añadir LoRA fusionado como nuevo adaptador
pipeline_service.add_lora(LoRAConfig(
    adapter_name="merged_style_character",
    lora_path=None,  # Ya está en memoria
    alpha=1.0
))
"""
