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
