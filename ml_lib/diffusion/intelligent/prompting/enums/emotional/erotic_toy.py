"""Erotic toy enum - replaces erotic_toys from YAML."""

from ..base_prompt_enum import BasePromptEnum


class EroticToy(BasePromptEnum):
    """Erotic toy options for character generation.

    Values extracted from character_attributes.yaml.
    Each enum value provides metadata through properties.
    All valid erotic toys have equal probability (uniform distribution).
    """

    DILDOS = "dildos"
    """Dildo/penetrative toy."""

    VIBRATORS = "vibrators"
    """Vibrator/clitoral/rabbit vibrator."""

    ANAL_TOYS = "anal_toys"
    """Anal plug/anal beads/prostate massager."""

    BDSM = "bdsm"
    """Bondage/handcuffs/restraints."""

    OTHER = "other"
    """Sex toys/erotic accessories/pleasure toys."""

    @property
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation."""
        _keywords: dict[EroticToy, tuple[str, ...]] = {
            EroticToy.DILDOS: ("dildo", "penetrative toy", "anal dildo", "vaginal dildo", "realistic dildo", "non-realistic dildo", "glass dildo", "silicone dildo"),
            EroticToy.VIBRATORS: ("vibrator", "clitoral vibrator", "rabbit vibrator", "bullet vibrator", "wand vibrator", "couples vibrator", "g-spot vibrator"),
            EroticToy.ANAL_TOYS: ("anal plug", "anal beads", "prostate massager", "anal dildo", "butt plug", "anal vibrator", "prostate toy"),
            EroticToy.BDSM: ("bondage", "handcuffs", "blindfold", "gags", "restraints", "ropes", "collars", "spreader bar", "paddles"),
            EroticToy.OTHER: ("sex toys", "erotic accessories", "kinky items", "adult toys", "pleasure toys", "sensual items"),
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
