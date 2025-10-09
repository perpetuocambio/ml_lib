"""Proposal template mapping entity."""

from dataclasses import dataclass

from infrastructure.config.prompts.entities.proposal_template_data import (
    ProposalTemplateData,
)


@dataclass(frozen=True)
class ProposalTemplateMapping:
    """Mapping of proposal type to template data."""

    proposal_type: str
    templates: list[ProposalTemplateData]
