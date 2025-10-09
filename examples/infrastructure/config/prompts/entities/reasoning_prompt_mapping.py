"""Reasoning prompt mapping entity."""

from dataclasses import dataclass

from infrastructure.config.prompts.entities.prompt_template import PromptTemplate


@dataclass(frozen=True)
class ReasoningPromptMapping:
    """Mapping of prompt name to reasoning prompt template."""

    prompt_name: str
    template: PromptTemplate
