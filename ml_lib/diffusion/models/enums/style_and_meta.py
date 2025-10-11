"""Consolidated enums for style, emotional, and meta attributes.

This module consolidates all enums from:
- ml_lib/diffusion/intelligent/prompting/enums/style/
- ml_lib/diffusion/intelligent/prompting/enums/emotional/
- ml_lib/diffusion/intelligent/prompting/enums/meta/
"""

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


# ============================================================================
# STYLE ENUMS
# ============================================================================


class SpecialEffect(BasePromptEnum):
    """Special effect options for character generation.

    Values extracted from character_attributes.yaml.
    Each enum value provides metadata through properties.
    All valid special effects have equal probability (uniform distribution).
    """

    WET = "wet"
    """Wet/damp/sweaty skin."""

    CUM = "cum"
    """Cum/semen/body fluids."""

    STICKY = "sticky"
    """Sticky/slimy/coated skin."""

    @property
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation."""
        _keywords: dict[SpecialEffect, tuple[str, ...]] = {
            SpecialEffect.WET: ("wet", "wet skin", "damp", "moist", "wet look", "wet clothes", "sweaty", "drenched", "sopping wet"),
            SpecialEffect.CUM: ("cum", "semen", "bodily fluids", "cum on skin", "cum on face", "cum on body", "ejaculation", "body fluids"),
            SpecialEffect.STICKY: ("sticky", "sticky skin", "sticky body", "gooey", "slimy", "coated", "covered in substance", "oily", "greasy"),
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


class ArtisticStyle(BasePromptEnum):
    """Artistic style options for character generation.

    Values extracted from character_attributes.yaml.
    Each enum value provides metadata through properties.
    All valid artistic styles have equal probability (uniform distribution).
    """

    PHOTOREALISTIC = "photorealistic"
    """Photorealistic/hyperrealistic/ultra realistic."""

    ANIME = "anime"
    """Anime/manga style."""

    CARTOON = "cartoon"
    """Cartoon/illustration/comic book style."""

    FANTASY = "fantasy"
    """Fantasy art/concept art."""

    VINTAGE = "vintage"
    """Vintage/retro/1950s-1970s style."""

    GOTHIC = "gothic"
    """Gothic/dark aesthetic."""

    @property
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation."""
        _keywords: dict[ArtisticStyle, tuple[str, ...]] = {
            ArtisticStyle.PHOTOREALISTIC: ("photorealistic", "hyperrealistic", "ultra realistic", "lifelike", "realistic", "photo", "photography", "professional photography", "film photography", "cinematic", "cinematic lighting", "studio photography"),
            ArtisticStyle.ANIME: ("anime style", "manga style", "japanese animation", "otaku art", "anime aesthetic", "chibi", "kawaii"),
            ArtisticStyle.CARTOON: ("cartoon style", "illustration", "comic book style", "hand-drawn", "animated", "disney style", "western cartoon", "comic art"),
            ArtisticStyle.FANTASY: ("fantasy art", "concept art", "fantasy illustration", "mythical art", "magical art style", "enchanted art", "fairy tale art"),
            ArtisticStyle.VINTAGE: ("vintage photo", "retro style", "vintage aesthetic", "1950s style", "1960s style", "1970s style", "vintage fashion", "retro photography", "old photo", "aged photo"),
            ArtisticStyle.GOTHIC: ("gothic style", "dark aesthetic", "goth", "dark fantasy", "macabre art", "dark romanticism", "victorian gothic"),
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


class AestheticStyle(BasePromptEnum):
    """Aesthetic style options for character generation.

    Values extracted from character_attributes.yaml.
    Each enum value provides metadata through properties.
    All valid aesthetic styles have equal probability (uniform distribution).
    """

    GOTH = "goth"
    """Goth/gothic/dark aesthetic."""

    PUNK = "punk"
    """Punk/punk rock style."""

    NURSE = "nurse"
    """Nurse outfit/medical costume."""

    WITCH = "witch"
    """Witch/magical costume."""

    NUN = "nun"
    """Nun/religious costume."""

    @property
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation."""
        _keywords: dict[AestheticStyle, tuple[str, ...]] = {
            AestheticStyle.GOTH: ("goth", "gothic", "dark makeup", "dark clothing", "black clothing", "goth style", "dark aesthetic", "dark fashion", "black makeup"),
            AestheticStyle.PUNK: ("punk", "punk style", "punk fashion", "punk look", "leather jacket", "punk clothes", "punk makeup", "punk hair", "rebellious style"),
            AestheticStyle.NURSE: ("nurse", "nurse outfit", "medical costume", "nurse uniform", "hospital uniform", "nurse hat", "nurse shoes", "medical attire", "hospital costume"),
            AestheticStyle.WITCH: ("witch", "witch costume", "witch outfit", "witch hat", "witch aesthetic", "magical costume", "wizard", "sorceress", "spellcaster"),
            AestheticStyle.NUN: ("nun", "nun outfit", "religious costume", "nun habit", "convent clothing", "catholic costume", "religious attire", "ascetic clothing"),
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


# ============================================================================
# EMOTIONAL ENUMS
# ============================================================================


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


# ============================================================================
# META ENUMS
# ============================================================================


class SafetyLevel(BasePromptEnum):
    """Safety level for content generation.

    Controls how strictly content is filtered for inappropriate material.
    """

    STRICT = "strict"
    """Strict filtering - blocks all potentially inappropriate content."""

    MODERATE = "moderate"
    """Moderate filtering - allows some mature content with restrictions."""

    RELAXED = "relaxed"
    """Relaxed filtering - minimal content restrictions."""

    @property
    def description(self) -> str:
        """Get detailed description of this safety level."""
        _descriptions: dict[SafetyLevel, str] = {
            SafetyLevel.STRICT: "Strict content filtering with maximum safety restrictions",
            SafetyLevel.MODERATE: "Balanced filtering allowing mature content with age verification",
            SafetyLevel.RELAXED: "Minimal filtering for adult audiences with content awareness",
        }
        return _descriptions[self]

    @property
    def filter_strength(self) -> int:
        """Get numeric filter strength (0-10, higher = stricter)."""
        _strengths: dict[SafetyLevel, int] = {
            SafetyLevel.STRICT: 10,
            SafetyLevel.MODERATE: 5,
            SafetyLevel.RELAXED: 2,
        }
        return _strengths[self]

    @property
    def blocks_explicit(self) -> bool:
        """Check if this level blocks explicit content."""
        _blocks: dict[SafetyLevel, bool] = {
            SafetyLevel.STRICT: True,
            SafetyLevel.MODERATE: False,
            SafetyLevel.RELAXED: False,
        }
        return _blocks[self]


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


class QualityTarget(BasePromptEnum):
    """Target quality level for generation.

    Determines the overall quality and detail level of the generated output.
    """

    LOW = "low"
    """Low quality - faster generation, less detail."""

    MEDIUM = "medium"
    """Medium quality - balanced speed and detail."""

    HIGH = "high"
    """High quality - slower generation, high detail."""

    MASTERPIECE = "masterpiece"
    """Masterpiece quality - maximum detail and quality, slowest generation."""

    @property
    def description(self) -> str:
        """Get detailed description of this quality level."""
        _descriptions: dict[QualityTarget, str] = {
            QualityTarget.LOW: "Fast generation with basic quality suitable for testing and previews",
            QualityTarget.MEDIUM: "Balanced quality with good detail and reasonable generation time",
            QualityTarget.HIGH: "High-quality output with excellent detail and fine rendering",
            QualityTarget.MASTERPIECE: "Ultimate quality with maximum detail, refinement, and artistic excellence",
        }
        return _descriptions[self]

    @property
    def steps_multiplier(self) -> float:
        """Get generation steps multiplier (1.0 = baseline)."""
        _multipliers: dict[QualityTarget, float] = {
            QualityTarget.LOW: 0.5,  # 50% of baseline steps
            QualityTarget.MEDIUM: 1.0,  # Baseline
            QualityTarget.HIGH: 1.5,  # 150% of baseline
            QualityTarget.MASTERPIECE: 2.0,  # 200% of baseline
        }
        return _multipliers[self]

    @property
    def quality_score(self) -> int:
        """Get numeric quality score (1-10)."""
        _scores: dict[QualityTarget, int] = {
            QualityTarget.LOW: 3,
            QualityTarget.MEDIUM: 6,
            QualityTarget.HIGH: 8,
            QualityTarget.MASTERPIECE: 10,
        }
        return _scores[self]


class ComplexityLevel(BasePromptEnum):
    """Complexity level for attributes and generation.

    Indicates the complexity and detail level of an attribute or generation task.
    """

    LOW = "low"
    """Low complexity - simple, straightforward."""

    MEDIUM = "medium"
    """Medium complexity - moderate detail and intricacy."""

    HIGH = "high"
    """High complexity - detailed, intricate, complex."""

    @property
    def description(self) -> str:
        """Get detailed description of this complexity level."""
        _descriptions: dict[ComplexityLevel, str] = {
            ComplexityLevel.LOW: "Simple and straightforward with minimal intricacy or special features",
            ComplexityLevel.MEDIUM: "Moderate complexity with balanced detail and some intricate elements",
            ComplexityLevel.HIGH: "Highly detailed and intricate with complex patterns and features",
        }
        return _descriptions[self]

    @property
    def complexity_score(self) -> int:
        """Get numeric complexity score (1-10)."""
        _scores: dict[ComplexityLevel, int] = {
            ComplexityLevel.LOW: 3,
            ComplexityLevel.MEDIUM: 6,
            ComplexityLevel.HIGH: 9,
        }
        return _scores[self]

    @property
    def token_weight(self) -> float:
        """Get prompt token weight multiplier for this complexity."""
        _weights: dict[ComplexityLevel, float] = {
            ComplexityLevel.LOW: 0.8,  # Less emphasis in prompt
            ComplexityLevel.MEDIUM: 1.0,  # Normal weight
            ComplexityLevel.HIGH: 1.3,  # Stronger emphasis
        }
        return _weights[self]
