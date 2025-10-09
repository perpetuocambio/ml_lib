"""Ollama model information."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class OllamaModelInfo:
    """Typed representation of Ollama model information."""

    name: str
    size_bytes: int
    digest: str
    modified_at: datetime
    family: str
    format: str
    parameter_size: str
    quantization_level: str

    def get_size_mb(self) -> float:
        """Get model size in megabytes."""
        return self.size_bytes / (1024 * 1024)

    def get_size_gb(self) -> float:
        """Get model size in gigabytes."""
        return self.size_bytes / (1024 * 1024 * 1024)

    def is_vision_model(self) -> bool:
        """Check if this is a vision-capable model."""
        return "vision" in self.name.lower() or "llava" in self.name.lower()

    def get_base_name(self) -> str:
        """Get base model name without tags."""
        return self.name.split(":")[0] if ":" in self.name else self.name
