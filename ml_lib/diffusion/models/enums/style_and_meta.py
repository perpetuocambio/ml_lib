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
        return self.value.replace["_", " "]


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
    def keywords(self) -> list[str]:
        """Keywords used in prompt generation."""
        if self == SpecialEffect.WET:
            return ["wet", "wet skin", "damp", "moist", "wet look", "wet clothes", "sweaty", "drenched", "sopping wet"]
        elif self == SpecialEffect.CUM:
            return ["cum", "semen", "bodily fluids", "cum on skin", "cum on face", "cum on body", "ejaculation", "body fluids"]
        elif self == SpecialEffect.STICKY:
            return ["sticky", "sticky skin", "sticky body", "gooey", "slimy", "coated", "covered in substance", "oily", "greasy"]
        else:
            raise ValueError(f"Unexpected SpecialEffect: {self}")

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
    def keywords(self) -> list[str]:
        """Keywords used in prompt generation."""
        if self == FantasyRace.ELF:
            return ["elf", "elven", "elf ears", "pointed ears", "elven features", "elven beauty", "elvish", "elven woman"]
        elif self == FantasyRace.DEMON:
            return ["demon", "demonic", "demon horns", "demon tail", "demon features", "demoness", "devil", "supernatural being", "evil being"]
        elif self == FantasyRace.MONSTER_GIRL:
            return ["monster girl", "monster girl features", "supernatural girl", "fantasy creature", "non-human", "creature girl", "beast girl", "monster woman"]
        else:
            raise ValueError(f"Unexpected FantasyRace: {self}")

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
    def keywords(self) -> list[str]:
        """Keywords used in prompt generation."""
        if self == ArtisticStyle.PHOTOREALISTIC:
            return ["photorealistic", "hyperrealistic", "ultra realistic", "lifelike", "realistic", "photo", "photography", "professional photography", "film photography", "cinematic", "cinematic lighting", "studio photography"]
        elif self == ArtisticStyle.ANIME:
            return ["anime style", "manga style", "japanese animation", "otaku art", "anime aesthetic", "chibi", "kawaii"]
        elif self == ArtisticStyle.CARTOON:
            return ["cartoon style", "illustration", "comic book style", "hand-drawn", "animated", "disney style", "western cartoon", "comic art"]
        elif self == ArtisticStyle.FANTASY:
            return ["fantasy art", "concept art", "fantasy illustration", "mythical art", "magical art style", "enchanted art", "fairy tale art"]
        elif self == ArtisticStyle.VINTAGE:
            return ["vintage photo", "retro style", "vintage aesthetic", "1950s style", "1960s style", "1970s style", "vintage fashion", "retro photography", "old photo", "aged photo"]
        elif self == ArtisticStyle.GOTHIC:
            return ["gothic style", "dark aesthetic", "goth", "dark fantasy", "macabre art", "dark romanticism", "victorian gothic"]
        else:
            raise ValueError(f"Unexpected ArtisticStyle: {self}")

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
    def keywords(self) -> list[str]:
        """Keywords used in prompt generation."""
        if self == AestheticStyle.GOTH:
            return ["goth", "gothic", "dark makeup", "dark clothing", "black clothing", "goth style", "dark aesthetic", "dark fashion", "black makeup"]
        elif self == AestheticStyle.PUNK:
            return ["punk", "punk style", "punk fashion", "punk look", "leather jacket", "punk clothes", "punk makeup", "punk hair", "rebellious style"]
        elif self == AestheticStyle.NURSE:
            return ["nurse", "nurse outfit", "medical costume", "nurse uniform", "hospital uniform", "nurse hat", "nurse shoes", "medical attire", "hospital costume"]
        elif self == AestheticStyle.WITCH:
            return ["witch", "witch costume", "witch outfit", "witch hat", "witch aesthetic", "magical costume", "wizard", "sorceress", "spellcaster"]
        elif self == AestheticStyle.NUN:
            return ["nun", "nun outfit", "religious costume", "nun habit", "convent clothing", "catholic costume", "religious attire", "ascetic clothing"]
        else:
            raise ValueError(f"Unexpected AestheticStyle: {self}")

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
    def keywords(self) -> list[str]:
        """Keywords used in prompt generation."""
        if self == EroticToy.DILDOS:
            return ["dildo", "penetrative toy", "anal dildo", "vaginal dildo", "realistic dildo", "non-realistic dildo", "glass dildo", "silicone dildo"]
        elif self == EroticToy.VIBRATORS:
            return ["vibrator", "clitoral vibrator", "rabbit vibrator", "bullet vibrator", "wand vibrator", "couples vibrator", "g-spot vibrator"]
        elif self == EroticToy.ANAL_TOYS:
            return ["anal plug", "anal beads", "prostate massager", "anal dildo", "butt plug", "anal vibrator", "prostate toy"]
        elif self == EroticToy.BDSM:
            return ["bondage", "handcuffs", "blindfold", "gags", "restraints", "ropes", "collars", "spreader bar", "paddles"]
        elif self == EroticToy.OTHER:
            return ["sex toys", "erotic accessories", "kinky items", "adult toys", "pleasure toys", "sensual items"]
        else:
            raise ValueError(f"Unexpected EroticToy: {self}")

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
    def keywords(self) -> list[str]:
        """Keywords used in prompt generation."""
        if self == EmotionalState.NEUTRAL:
            return ["neutral expression", "calm expression", "normal expression", "relaxed face", "calm face"]
        elif self == EmotionalState.HAPPY:
            return ["happy expression", "smiling", "joyful", "smile", "joy", "content", "pleased expression"]
        elif self == EmotionalState.SENSUAL:
            return ["sensual expression", "seductive look", "sultry expression", "come hither look", "seductive gaze", "alluring look"]
        elif self == EmotionalState.ORGASM:
            return ["orgasm face", "orgasm expression", "orgasmic look", "face of pleasure", "ecstasy expression", "pleasure face", "ecstatic expression"]
        elif self == EmotionalState.INTENSE:
            return ["intense expression", "focused look", "intense gaze", "passionate look", "intense expression", "focused eyes"]
        elif self == EmotionalState.SURPRISED:
            return ["surprised expression", "surprised look", "shocked look", "surprised face", "wide eyes", "shocked expression"]
        else:
            raise ValueError(f"Unexpected EmotionalState: {self}")

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
        if self == SafetyLevel.STRICT:
            return "Strict content filtering with maximum safety restrictions"
        elif self == SafetyLevel.MODERATE:
            return "Balanced filtering allowing mature content with age verification"
        elif self == SafetyLevel.RELAXED:
            return "Minimal filtering for adult audiences with content awareness"
        else:
            raise ValueError(f"Unexpected SafetyLevel: {self}")

    @property
    def filter_strength(self) -> int:
        """Get numeric filter strength (0-10, higher = stricter)."""
        if self == SafetyLevel.STRICT:
            return 10
        elif self == SafetyLevel.MODERATE:
            return 5
        elif self == SafetyLevel.RELAXED:
            return 2
        else:
            raise ValueError(f"Unexpected SafetyLevel: {self}")

    @property
    def blocks_explicit(self) -> bool:
        """Check if this level blocks explicit content."""
        if self == SafetyLevel.STRICT:
            return True
        elif self == SafetyLevel.MODERATE:
            return False
        elif self == SafetyLevel.RELAXED:
            return False
        else:
            raise ValueError(f"Unexpected SafetyLevel: {self}")


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
        if self == CharacterFocus.PORTRAIT:
            return "Close-up portrait showing head and shoulders with detailed facial features"
        elif self == CharacterFocus.FULL_BODY:
            return "Complete character view from head to toe showing full body proportions"
        elif self == CharacterFocus.SCENE:
            return "Character integrated into environment with contextual scene elements"
        else:
            raise ValueError(f"Unexpected CharacterFocus: {self}")

    @property
    def crop_ratio(self) -> list[int]:
        """Get recommended aspect ratio (width, height) for this focus."""
        if self == CharacterFocus.PORTRAIT:
            return (3, 4)  # Vertical portrait
        elif self == CharacterFocus.FULL_BODY:
            return (2, 3)  # Slightly vertical
        elif self == CharacterFocus.SCENE:
            return (16, 9)  # Cinematic landscape
        else:
            raise ValueError(f"Unexpected CharacterFocus: {self}")

    @property
    def detail_level(self) -> int:
        """Get detail focus level (1-10, higher = more character detail)."""
        if self == CharacterFocus.PORTRAIT:
            return 10  # Maximum facial detail
        elif self == CharacterFocus.FULL_BODY:
            return 7  # Balanced body detail
        elif self == CharacterFocus.SCENE:
            return 5  # Context over character detail
        else:
            raise ValueError(f"Unexpected CharacterFocus: {self}")


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
        if self == QualityTarget.LOW:
            return "Fast generation with basic quality suitable for testing and previews"
        elif self == QualityTarget.MEDIUM:
            return "Balanced quality with good detail and reasonable generation time"
        elif self == QualityTarget.HIGH:
            return "High-quality output with excellent detail and fine rendering"
        elif self == QualityTarget.MASTERPIECE:
            return "Ultimate quality with maximum detail, refinement, and artistic excellence"
        else:
            raise ValueError(f"Unexpected QualityTarget: {self}")

    @property
    def steps_multiplier(self) -> float:
        """Get generation steps multiplier (1.0 = baseline)."""
        if self == QualityTarget.LOW:
            return 0.5  # 50% of baseline steps
        elif self == QualityTarget.MEDIUM:
            return 1.0  # Baseline
        elif self == QualityTarget.HIGH:
            return 1.5  # 150% of baseline
        elif self == QualityTarget.MASTERPIECE:
            return 2.0  # 200% of baseline
        else:
            raise ValueError(f"Unexpected QualityTarget: {self}")

    @property
    def quality_score(self) -> int:
        """Get numeric quality score (1-10)."""
        if self == QualityTarget.LOW:
            return 3
        elif self == QualityTarget.MEDIUM:
            return 6
        elif self == QualityTarget.HIGH:
            return 8
        elif self == QualityTarget.MASTERPIECE:
            return 10
        else:
            raise ValueError(f"Unexpected QualityTarget: {self}")


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
        if self == ComplexityLevel.LOW:
            return "Simple and straightforward with minimal intricacy or special features"
        elif self == ComplexityLevel.MEDIUM:
            return "Moderate complexity with balanced detail and some intricate elements"
        elif self == ComplexityLevel.HIGH:
            return "Highly detailed and intricate with complex patterns and features"
        else:
            raise ValueError(f"Unexpected ComplexityLevel: {self}")

    @property
    def complexity_score(self) -> int:
        """Get numeric complexity score (1-10)."""
        if self == ComplexityLevel.LOW:
            return 3
        elif self == ComplexityLevel.MEDIUM:
            return 6
        elif self == ComplexityLevel.HIGH:
            return 9
        else:
            raise ValueError(f"Unexpected ComplexityLevel: {self}")

    @property
    def token_weight(self) -> float:
        """Get prompt token weight multiplier for this complexity."""
        if self == ComplexityLevel.LOW:
            return 0.8  # Less emphasis in prompt
        elif self == ComplexityLevel.MEDIUM:
            return 1.0  # Normal weight
        elif self == ComplexityLevel.HIGH:
            return 1.3  # Stronger emphasis
        else:
            raise ValueError(f"Unexpected ComplexityLevel: {self}")
