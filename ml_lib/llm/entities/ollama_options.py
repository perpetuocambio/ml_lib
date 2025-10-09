"""Ollama request options."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OllamaOptions:
    """Configuration options for Ollama LLM requests (temperature, context window)."""

    temperature: float
    num_ctx: int
