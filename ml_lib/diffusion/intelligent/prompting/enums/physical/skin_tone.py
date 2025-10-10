"""Skin tone enum - replaces skin_tones from YAML."""

from typing import TYPE_CHECKING

from ..base_prompt_enum import BasePromptEnum

if TYPE_CHECKING:
    from .ethnicity import Ethnicity


class SkinTone(BasePromptEnum):
    """Skin tone options for character generation.

    Values extracted from character_attributes.yaml.
    All valid skin tones have equal probability (uniform distribution).

    Inherits prompt-friendly __str__() from BasePromptEnum.
    Each enum value provides metadata through properties.
    """

    FAIR = "fair"
    """Fair/pale skin tone."""

    LIGHT = "light"
    """Light to medium-light skin tone."""

    MEDIUM = "medium"
    """Medium/olive/tan skin tone."""

    MEDIUM_DARK = "medium_dark"
    """Medium-dark/olive-brown skin tone."""

    DARK = "dark"
    """Dark/deep/ebony skin tone."""

    @property
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation (literal strings for prompts).

        Examples:
            >>> SkinTone.FAIR.keywords
            ('fair skin', 'light skin', 'pale skin')
        """
        from .ethnicity import Ethnicity  # Avoid circular import

        return {
            SkinTone.FAIR: ("fair skin", "light skin", "pale skin"),
            SkinTone.LIGHT: ("light skin", "light-medium skin", "medium-fair skin"),
            SkinTone.MEDIUM: ("medium skin", "olive skin", "tan skin"),
            SkinTone.MEDIUM_DARK: ("medium-dark skin", "dark olive", "olive-brown skin"),
            SkinTone.DARK: ("dark skin", "deep skin", "rich skin", "ebony skin"),
        }[self]

    @property
    def prompt_weight(self) -> float:
        """Weight/emphasis for this attribute in prompts.

        Examples:
            >>> SkinTone.FAIR.prompt_weight
            1.1
        """
        return {
            SkinTone.FAIR: 1.1,
            SkinTone.LIGHT: 1.1,
            SkinTone.MEDIUM: 1.2,
            SkinTone.MEDIUM_DARK: 1.2,
            SkinTone.DARK: 1.2,
        }[self]

    @property
    def ethnicity_associations(self) -> tuple["Ethnicity", ...]:
        """Ethnicities commonly associated with this skin tone (strongly-typed enums).

        Examples:
            >>> SkinTone.MEDIUM.ethnicity_associations
            (Ethnicity.MIDDLE_EASTERN, Ethnicity.SOUTH_ASIAN, Ethnicity.HISPANIC_LATINX)
        """
        from .ethnicity import Ethnicity  # Avoid circular import

        return {
            SkinTone.FAIR: (Ethnicity.CAUCASIAN,),
            SkinTone.LIGHT: (Ethnicity.CAUCASIAN, Ethnicity.MIDDLE_EASTERN),
            SkinTone.MEDIUM: (
                Ethnicity.MIDDLE_EASTERN,
                Ethnicity.SOUTH_ASIAN,
                Ethnicity.HISPANIC_LATINX,
            ),
            SkinTone.MEDIUM_DARK: (
                Ethnicity.SOUTH_ASIAN,
                Ethnicity.MIDDLE_EASTERN,
                Ethnicity.AFRICAN_AMERICAN,
                Ethnicity.HISPANIC_LATINX,
            ),
            SkinTone.DARK: (Ethnicity.AFRICAN_AMERICAN,),
        }[self]

    @property
    def min_age(self) -> int:
        """Minimum age for this attribute."""
        return 18

    @property
    def max_age(self) -> int:
        """Maximum age for this attribute."""
        return 80

    def get_ethnicity_prompts(self) -> tuple[str, ...]:
        """Get all ethnicity associations as prompt-ready strings.

        Returns:
            Tuple of prompt-friendly ethnicity strings.

        Examples:
            >>> SkinTone.MEDIUM.get_ethnicity_prompts()
            ('middle eastern', 'south asian', 'hispanic latinx')
        """
        return tuple(str(eth) for eth in self.ethnicity_associations)
