"""Ollama API response."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OllamaResponse:
    """Typed representation of Ollama API response."""

    response: str
    done: bool
    eval_count: int
