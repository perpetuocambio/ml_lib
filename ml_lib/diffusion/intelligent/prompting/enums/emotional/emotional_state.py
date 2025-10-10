"""Emotional state enum - replaces emotional_states from YAML."""

from ..base_prompt_enum import BasePromptEnum


class EmotionalState(BasePromptEnum):
    """Emotional state options for character generation.

    Values extracted from character_attributes.yaml.
    Each enum value provides metadata through properties.
    All valid emotional states have equal probability (uniform distribution).
    """

    NEUTRAL = "neutral"
    """Neutral/calm expression."""

    HAPPY = "happy"
    """Happy/smiling/joyful."""

    SENSUAL = "sensual"
    """Sensual/seductive/sultry."""

    ORGASM = "orgasm"
    """Orgasm/ecstasy/pleasure expression."""

    INTENSE = "intense"
    """Intense/focused/passionate."""

    SURPRISED = "surprised"
    """Surprised/shocked expression."""

    @property
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation."""
        _keywords: dict[EmotionalState, tuple[str, ...]] = {
            EmotionalState.NEUTRAL: ("neutral expression", "calm expression", "normal expression", "relaxed face", "calm face"),
            EmotionalState.HAPPY: ("happy expression", "smiling", "joyful", "smile", "joy", "content", "pleased expression"),
            EmotionalState.SENSUAL: ("sensual expression", "seductive look", "sultry expression", "come hither look", "seductive gaze", "alluring look"),
            EmotionalState.ORGASM: ("orgasm face", "orgasm expression", "orgasmic look", "face of pleasure", "ecstasy expression", "pleasure face", "ecstatic expression"),
            EmotionalState.INTENSE: ("intense expression", "focused look", "intense gaze", "passionate look", "intense expression", "focused eyes"),
            EmotionalState.SURPRISED: ("surprised expression", "surprised look", "shocked look", "surprised face", "wide eyes", "shocked expression"),
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
