"""LLM content for multimodal messages."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LLMContent:
    """Multimodal content for LLM messages."""

    content_type: str  # "text" or "image_url"
    text: str | None = None
    image_url: str | None = None
