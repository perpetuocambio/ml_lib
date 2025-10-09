"""System prompt mapping entity."""

from dataclasses import dataclass

from infrastructure.config.prompts.entities.prompt_template import PromptTemplate


@dataclass(frozen=True)
class SystemPromptMapping:
    """Mapping of prompt name to system prompt template."""

    prompt_name: str
    template: PromptTemplate
