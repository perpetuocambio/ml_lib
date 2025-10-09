"""Proposal template data entity."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProposalTemplateData:
    """Data for proposal template."""

    template_key: str
    content: str
