"""Pose enum - replaces poses from YAML."""

from ..base_prompt_enum import BasePromptEnum


class Pose(BasePromptEnum):
    """Pose options for character generation.

    Values extracted from character_attributes.yaml.
    Each enum value provides metadata through properties.
    All valid poses have equal probability (uniform distribution).
    """

    SITTING = "sitting"
    """Sitting/seated/lounging pose."""

    STANDING = "standing"
    """Standing/upright/erect pose."""

    KNEELING = "kneeling"
    """Kneeling/on knees pose."""

    LYING = "lying"
    """Lying down/reclining pose."""

    INTIMATE_CLOSE = "intimate_close"
    """Intimate/close up/erotic pose (explicit)."""

    EXPLICIT_SEXUAL = "explicit_sexual"
    """Explicit sexual position/act (explicit)."""

    @property
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation."""
        _keywords: dict[Pose, tuple[str, ...]] = {
            Pose.SITTING: ("sitting", "seated", "lounging", "reclining", "relaxed pose"),
            Pose.STANDING: ("standing", "erect", "upright", "posed", "straight posture"),
            Pose.KNEELING: ("kneeling", "kneeling position", "on knees", "bent position"),
            Pose.LYING: ("lying down", "reclining", "laying", "horizontal position"),
            Pose.INTIMATE_CLOSE: ("intimate pose", "close up", "erotic pose", "sensual position"),
            Pose.EXPLICIT_SEXUAL: ("sexual position", "sexual act", "intimate act", "erotic activity"),
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
