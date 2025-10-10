"""Activity enum - replaces activities from YAML."""

from ..base_prompt_enum import BasePromptEnum


class Activity(BasePromptEnum):
    """Activity options for character generation.

    Values extracted from character_attributes.yaml.
    Each enum value provides metadata through properties.
    All valid activities have equal probability (uniform distribution).
    """

    INTIMATE = "intimate"
    """Intimate/erotic/sensual activity (explicit)."""

    SEXUAL = "sexual"
    """Sexual/intercourse/penetration (explicit)."""

    FOREPLAY = "foreplay"
    """Foreplay/caressing/touching (explicit)."""

    BDSM = "bdsm"
    """BDSM/domination/spanking (explicit)."""

    @property
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation."""
        _keywords: dict[Activity, tuple[str, ...]] = {
            Activity.INTIMATE: ("intimate position", "erotic pose", "sensual activity", "romantic setting", "intimate moment", "erotic scene"),
            Activity.SEXUAL: ("sex", "intercourse", "penetration", "oral", "anal", "position", "erotic act", "sexual activity", "lovemaking"),
            Activity.FOREPLAY: ("foreplay", "caressing", "touching", "kissing", "sensual massage", "erotic massage", "passionate kissing", "making out"),
            Activity.BDSM: ("bdsm", "domination", "submissive", "spanking", "impact play", "bondage scene", "role play"),
        }
        return _keywords[self]

    @property
    def min_age(self) -> int:
        """Minimum age for this attribute."""
        return 18

    @property
    def max_age(self) -> int:
        """Maximum age for this attribute."""
        return 80
