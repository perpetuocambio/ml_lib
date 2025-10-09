"""Entity for proposal template fields."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProposalTemplateFields:
    """
    Proposal template fields.
    Replaces generic dict violations.
    """

    description: str
    expected_outcome: str
