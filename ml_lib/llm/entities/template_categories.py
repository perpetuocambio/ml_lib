"""Entity for listing available template categories."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TemplateCategoriesListing:
    """
    Listing of available template categories.
    Replaces Dict[str, List[str]] violation.
    """

    system_prompts: list[str]
    reasoning_prompts: list[str]
    proposal_templates: list[str]
