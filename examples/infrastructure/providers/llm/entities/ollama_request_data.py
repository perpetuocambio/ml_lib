"""Ollama request data representation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OllamaRequestData:
    """Typed representation of Ollama request data."""

    model: str
    prompt: str
    stream: bool
    temperature: float
    num_ctx: int
    images: list[str] | None = None
