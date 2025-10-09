"""Configuration entity for all prompt templates."""

from dataclasses import dataclass

from infrastructure.config.prompts.entities.proposal_template_mapping import (
    ProposalTemplateMapping,
)
from infrastructure.config.prompts.entities.reasoning_prompt_mapping import (
    ReasoningPromptMapping,
)
from infrastructure.config.prompts.entities.system_prompt_mapping import (
    SystemPromptMapping,
)


@dataclass(frozen=True)
class PromptsConfig:
    """Complete configuration for prompt templates system."""

    system_prompts: list[SystemPromptMapping]
    reasoning_prompts: list[ReasoningPromptMapping]
    proposal_templates: list[ProposalTemplateMapping]
