"""LLM message for structured communication."""

from __future__ import annotations

from dataclasses import dataclass

from infrastructure.data.extractors.entities.llm_content import LLMContent


@dataclass
class LLMMessage:
    """Structured message for LLM compatible with OpenAI format."""

    role: str
    content: str | list[LLMContent]
