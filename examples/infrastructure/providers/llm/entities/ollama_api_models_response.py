"""Raw Ollama API models response."""

from __future__ import annotations

from dataclasses import dataclass

from infrastructure.providers.llm.entities.ollama_raw_model_data import (
    OllamaRawModelData,
)


@dataclass
class OllamaApiModelsResponse:
    """Raw response structure from Ollama /api/tags endpoint."""

    models: list[OllamaRawModelData]
