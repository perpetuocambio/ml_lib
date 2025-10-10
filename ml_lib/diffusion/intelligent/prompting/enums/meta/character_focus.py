"""Character focus enum for framing and composition."""

from ..base_prompt_enum import BasePromptEnum


class CharacterFocus(BasePromptEnum):
    """Focus type for character generation.

    Determines the framing and composition of the generated character.
    """

    PORTRAIT = "portrait"
    """Portrait view - head and shoulders, close-up focus."""

    FULL_BODY = "full_body"
    """Full body view - entire character visible, head to toe."""

    SCENE = "scene"
    """Scene view - character in environment context, wider framing."""

    @property
    def description(self) -> str:
        """Get detailed description of this focus type."""
        _descriptions: dict[CharacterFocus, str] = {
            CharacterFocus.PORTRAIT: "Close-up portrait showing head and shoulders with detailed facial features",
            CharacterFocus.FULL_BODY: "Complete character view from head to toe showing full body proportions",
            CharacterFocus.SCENE: "Character integrated into environment with contextual scene elements",
        }
        return _descriptions[self]

    @property
    def crop_ratio(self) -> tuple[int, int]:
        """Get recommended aspect ratio (width, height) for this focus."""
        _ratios: dict[CharacterFocus, tuple[int, int]] = {
            CharacterFocus.PORTRAIT: (3, 4),  # Vertical portrait
            CharacterFocus.FULL_BODY: (2, 3),  # Slightly vertical
            CharacterFocus.SCENE: (16, 9),  # Cinematic landscape
        }
        return _ratios[self]

    @property
    def detail_level(self) -> int:
        """Get detail focus level (1-10, higher = more character detail)."""
        _levels: dict[CharacterFocus, int] = {
            CharacterFocus.PORTRAIT: 10,  # Maximum facial detail
            CharacterFocus.FULL_BODY: 7,  # Balanced body detail
            CharacterFocus.SCENE: 5,  # Context over character detail
        }
        return _levels[self]
