"""Typed structures for metadata serialization."""

from dataclasses import dataclass, asdict


@dataclass
class LoRASerializable:
    """Serialized LoRA information."""

    name: str
    alpha: float
    source: str


@dataclass
class GenerationMetadataSerializable:
    """Typed structure for generation metadata serialization.

    Use dataclasses.asdict() for JSON serialization.
    """

    prompt: str
    negative_prompt: str
    seed: int
    steps: int
    cfg_scale: float
    width: int
    height: int
    sampler: str
    loras_used: list[LoRASerializable]
    generation_time_seconds: float
    peak_vram_gb: float
    base_model_id: str
    pipeline_type: str
