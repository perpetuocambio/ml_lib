from enum import Enum


class BasePromptEnum(Enum):
    """Base class for all enums that need prompt-friendly string conversion.

    Provides automatic conversion of underscore-separated values to space-separated
    strings suitable for AI prompt generation.

    All enums in the prompting module should inherit from this base class for
    consistent string representation behavior.

    Examples:
        >>> class Color(BasePromptEnum):
        ...     DARK_BLUE = "dark_blue"
        >>> str(Color.DARK_BLUE)
        'dark blue'
        >>> f"Use {Color.DARK_BLUE} color"
        'Use dark blue color'
    """

    def __str__(self) -> str:
        """Get the prompt-friendly string representation.

        Automatically converts underscores to spaces for natural language output.

        Returns:
            Prompt-friendly string with underscores replaced by spaces.
        """
        return self.value.replace("_", " ")
