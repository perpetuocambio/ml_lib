"""Raw model data from Ollama API."""

from __future__ import annotations

from dataclasses import dataclass

from ml_lib.llm.entities.ollama_model_details import (
    OllamaModelDetails,
)


@dataclass
class OllamaRawModelData:
    """Raw model data structure from Ollama API."""

    name: str
    size: int
    digest: str
    modified_at: str
    details: OllamaModelDetails
