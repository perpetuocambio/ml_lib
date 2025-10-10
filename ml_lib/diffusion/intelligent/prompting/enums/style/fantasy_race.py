"""Fantasy race enum - replaces fantasy_races from YAML."""

from ..base_prompt_enum import BasePromptEnum


class FantasyRace(BasePromptEnum):
    """Fantasy race options for character generation.

    Values extracted from character_attributes.yaml.
    Each enum value provides metadata through properties.
    All valid fantasy races have equal probability (uniform distribution).
    """

    ELF = "elf"
    """Elf/elven/pointed ears."""

    DEMON = "demon"
    """Demon/demonic/demon horns."""

    MONSTER_GIRL = "monster_girl"
    """Monster girl/supernatural creature."""

    @property
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation."""
        _keywords: dict[FantasyRace, tuple[str, ...]] = {
            FantasyRace.ELF: ("elf", "elven", "elf ears", "pointed ears", "elven features", "elven beauty", "elvish", "elven woman"),
            FantasyRace.DEMON: ("demon", "demonic", "demon horns", "demon tail", "demon features", "demoness", "devil", "supernatural being", "evil being"),
            FantasyRace.MONSTER_GIRL: ("monster girl", "monster girl features", "supernatural girl", "fantasy creature", "non-human", "creature girl", "beast girl", "monster woman"),
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
