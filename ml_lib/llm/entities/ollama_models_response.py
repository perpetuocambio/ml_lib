"""Ollama models list response."""

from __future__ import annotations

from dataclasses import dataclass

from ml_lib.llm.entities.ollama_model_info import OllamaModelInfo


@dataclass
class OllamaModelsResponse:
    """Typed representation of Ollama models list response."""

    models: list[OllamaModelInfo]

    def get_model_names(self) -> list[str]:
        """Extract model names."""
        return [model.name for model in self.models]
