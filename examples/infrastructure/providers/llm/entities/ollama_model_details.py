"""Ollama model details representation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OllamaModelDetails:
    """Model details from Ollama API response."""

    family: str
    format: str
    parameter_size: str
    quantization_level: str
