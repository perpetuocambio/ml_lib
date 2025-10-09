"""Configuration entity for individual prompt templates."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    """Configuration for an individual prompt template."""

    template: str
    variables: list[str]
