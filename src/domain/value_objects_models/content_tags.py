"""Content tags and NSFW classification enums."""

import re
from enum import Enum
from dataclasses import dataclass, field


class NSFWCategory(Enum):
    """Categories of NSFW content."""
    ORAL = "oral"
    ANAL = "anal"
    VAGINAL = "vaginal"
    CUM = "cum"
    BONDAGE = "bondage"
    HANDJOB = "handjob"
    FOOTJOB = "footjob"
    TITJOB = "titjob"
    GROUP = "group"
    MASTURBATION = "masturbation"
    NUDITY = "nudity"
    POSING = "posing"
    EXPRESSION = "expression"
    CLOTHING = "clothing"
    BODY_PART = "body_part"
    POSITION = "position"
    OTHER = "other"


class PromptTokenPriority(Enum):
    """Priority levels for prompt token preservation during compaction."""
    CRITICAL = 4  # Core content (characters, main action)
    HIGH = 3      # NSFW acts, specific features
    MEDIUM = 2    # Context, setting, modifiers
    LOW = 1       # Quality tags, redundant descriptors
    DISCARD = 0   # Can be removed if needed


class QualityTag(Enum):
    """Common quality tags in prompts."""
    MASTERPIECE = "masterpiece"
    BEST_QUALITY = "best quality"
    HIGH_QUALITY = "high quality"
    AMAZING_QUALITY = "amazing quality"
    VERY_AESTHETIC = "very aesthetic"
    ABSURDRES = "absurdres"
    DETAILED = "detailed"
    HIGHLY_DETAILED = "highly detailed"
    SHARP_FOCUS = "sharp focus"
    ULTRA_DETAILED = "ultra detailed"
    K8 = "8k"
    K4 = "4k"
    UHD = "uhd"
    RAW_PHOTO = "raw photo"
    PROFESSIONAL = "professional"


# Content tag mappings
NSFW_KEYWORDS: dict[NSFWCategory, list[str]] = {
    NSFWCategory.ORAL: [
        "fellatio", "blowjob", "oral", "deepthroat", "throat", "mouth",
        "licking penis", "sucking", "oral sex", "cock sucking", "irrumatio"
    ],
    NSFWCategory.ANAL: [
        "anal", "anal sex", "anal penetration", "anus", "asshole",
        "anal insertion", "rimjob", "anilingus", "ass licking"
    ],
    NSFWCategory.VAGINAL: [
        "vaginal", "sex", "fucking", "penetration", "intercourse",
        "pussy", "vaginal sex", "missionary", "vaginal penetration"
    ],
    NSFWCategory.CUM: [
        "cum", "cumshot", "facial", "bukkake", "creampie", "ejaculation",
        "cum on face", "cum in mouth", "cum on body", "cum inside",
        "excessive cum", "cum drip", "semen"
    ],
    NSFWCategory.BONDAGE: [
        "bondage", "bdsm", "tied", "restrained", "rope", "bound",
        "chains", "handcuffs", "submissive", "dominant", "hogtied"
    ],
    NSFWCategory.HANDJOB: [
        "handjob", "hand job", "stroking", "penis grab", "cock grab",
        "jerking", "hand on penis", "manual stimulation"
    ],
    NSFWCategory.FOOTJOB: [
        "footjob", "foot job", "feet", "foot fetish", "foot worship",
        "soles", "toes on penis"
    ],
    NSFWCategory.TITJOB: [
        "titjob", "paizuri", "breast sex", "titty fuck", "between breasts",
        "breast press", "breast sandwich"
    ],
    NSFWCategory.GROUP: [
        "threesome", "gangbang", "group sex", "orgy", "multiple partners",
        "ffm", "mmf", "multiple boys", "multiple girls", "group"
    ],
    NSFWCategory.MASTURBATION: [
        "masturbation", "masturbating", "touching self", "self pleasure",
        "fingering", "self stimulation", "solo", "hands between legs"
    ],
    NSFWCategory.NUDITY: [
        "nude", "naked", "completely nude", "fully nude", "no clothes",
        "undressed", "bare", "exposed"
    ],
    NSFWCategory.POSING: [
        "spread legs", "spread pussy", "presenting", "displaying",
        "jack-o pose", "all fours", "bent over", "legs up", "squatting",
        "kneeling", "laying down", "on back", "on stomach"
    ],
    NSFWCategory.EXPRESSION: [
        "ahegao", "orgasm face", "pleasure face", "aroused", "horny",
        "seductive", "lustful", "blushing", "moaning", "tongue out",
        "eyes rolling", "naughty face", "slutty", "sultry"
    ],
    NSFWCategory.CLOTHING: [
        "lingerie", "underwear", "panties", "bra", "stockings", "garter",
        "fishnet", "transparent", "see-through", "torn clothes", "partially clothed",
        "topless", "bottomless", "skirt lift", "shirt lift", "exposed breasts"
    ],
    NSFWCategory.BODY_PART: [
        "breasts", "nipples", "tits", "boobs", "ass", "butt", "buttocks",
        "penis", "cock", "dick", "pussy", "vagina", "clitoris", "labia",
        "testicles", "balls", "scrotum", "thighs", "hips", "crotch"
    ],
    NSFWCategory.POSITION: [
        "missionary", "doggystyle", "cowgirl", "reverse cowgirl", "spitroast",
        "standing sex", "against wall", "from behind", "69", "sideways",
        "prone bone", "mating press", "full nelson", "lotus position"
    ],
}


# Character and core content keywords (CRITICAL priority)
CORE_CONTENT_KEYWORDS = [
    # Characters
    "girl", "boy", "woman", "man", "male", "female", "person", "character",
    "1girl", "2girls", "3girls", "4girls", "1boy", "2boys", "3boys",
    "multiple girls", "multiple boys", "multiple people",
    # Basic descriptors
    "face", "body", "portrait", "full body", "upper body", "close-up",
]


# Context keywords (MEDIUM priority)
CONTEXT_KEYWORDS = [
    # Settings
    "bedroom", "bathroom", "kitchen", "office", "classroom", "library",
    "outdoor", "indoor", "beach", "forest", "city", "room",
    # Lighting
    "natural light", "sunlight", "moonlight", "candlelight", "dim light",
    "bright", "dark", "shadows", "volumetric lighting", "cinematic lighting",
    # Camera
    "pov", "from above", "from below", "from behind", "side view",
    "close up", "wide shot", "dynamic angle",
]


class NSFWKeywordRegistry:
    """
    Registry of NSFW keywords organized by category.

    Provides type-safe access to NSFW keywords with efficient lookups.
    Encapsulates the keyword dictionary to enable future extensibility.
    """

    def __init__(self):
        """Initialize registry with default NSFW keywords."""
        self._keywords = NSFW_KEYWORDS

    def get_keywords(self, category: NSFWCategory) -> list[str]:
        """
        Get keywords for a specific category.

        Args:
            category: NSFW category

        Returns:
            List of keywords for the category
        """
        return self._keywords.get(category, [])

    def get_all_keywords(self) -> dict[NSFWCategory, list[str]]:
        """
        Get all keywords organized by category.

        Returns:
            Dictionary mapping categories to keywords
        """
        return self._keywords.copy()

    def find_category_for_keyword(self, keyword: str) -> NSFWCategory | None:
        """
        Find which category a keyword belongs to.

        Args:
            keyword: Keyword to search for (case-insensitive)

        Returns:
            Category containing the keyword, or None if not found
        """
        keyword_lower = keyword.lower()
        for category, keywords in self._keywords.items():
            if keyword_lower in keywords:
                return category
        return None

    def matches_any_keyword(self, text: str, category: NSFWCategory | None = None) -> bool:
        """
        Check if text contains any keyword.

        Args:
            text: Text to check (case-insensitive)
            category: Optional category to restrict search to

        Returns:
            True if text contains any keyword from specified category (or any category if None)
        """
        text_lower = text.lower()

        if category:
            # Check only specific category
            keywords = self._keywords.get(category, [])
            return any(kw in text_lower for kw in keywords)
        else:
            # Check all categories
            for keywords in self._keywords.values():
                if any(kw in text_lower for kw in keywords):
                    return True
            return False

    def find_matching_keywords(
        self, text: str, category: NSFWCategory | None = None
    ) -> list[str]:
        """
        Find all keywords that match in the text.

        Args:
            text: Text to search in (case-insensitive)
            category: Optional category to restrict search to

        Returns:
            List of matched keywords
        """
        text_lower = text.lower()
        matches = []

        if category:
            # Search only specific category
            keywords = self._keywords.get(category, [])
            matches.extend(kw for kw in keywords if kw in text_lower)
        else:
            # Search all categories
            for keywords in self._keywords.values():
                matches.extend(kw for kw in keywords if kw in text_lower)

        return matches

    def get_all_categories(self) -> list[NSFWCategory]:
        """
        Get list of all registered categories.

        Returns:
            List of NSFW categories
        """
        return list(self._keywords.keys())

    def add_keyword(self, category: NSFWCategory, keyword: str):
        """
        Add a new keyword to a category.

        Args:
            category: Category to add keyword to
            keyword: Keyword to add (will be lowercased)
        """
        keyword_lower = keyword.lower()
        if category not in self._keywords:
            self._keywords[category] = []
        if keyword_lower not in self._keywords[category]:
            self._keywords[category].append(keyword_lower)


# Global singleton instance
NSFW_REGISTRY = NSFWKeywordRegistry()


@dataclass
class TokenClassification:
    """Classification of a prompt token."""
    token: str
    priority: PromptTokenPriority
    category: NSFWCategory | None = None
    is_quality_tag: bool = False
    is_weight_syntax: bool = False
    weight: float = 1.0

    @property
    def should_keep(self) -> bool:
        """Whether this token should be kept during compaction."""
        return self.priority.value >= PromptTokenPriority.MEDIUM.value


@dataclass
class PromptCompactionResult:
    """Result of prompt compaction operation."""

    # Input
    original_prompt: str
    original_token_count: int
    max_tokens: int

    # Output
    compacted_prompt: str
    compacted_token_count: int

    # What was kept/removed
    kept_tokens: list[TokenClassification] = field(default_factory=list)
    removed_tokens: list[TokenClassification] = field(default_factory=list)

    # Analysis
    nsfw_categories_found: list[NSFWCategory] = field(default_factory=list)
    core_content_preserved: bool = True
    nsfw_content_preserved: bool = True
    quality_tags_removed: int = 0

    # Warnings
    warnings: list[str] = field(default_factory=list)

    @property
    def was_compacted(self) -> bool:
        """Whether compaction was performed."""
        return self.original_token_count > self.max_tokens

    @property
    def tokens_removed(self) -> int:
        """Number of tokens removed."""
        return self.original_token_count - self.compacted_token_count

    @property
    def compression_ratio(self) -> float:
        """Compression ratio (0-1, lower = more compression)."""
        if self.original_token_count == 0:
            return 1.0
        return self.compacted_token_count / self.original_token_count

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)


@dataclass
class DetectedActs:
    """
    Collection of detected NSFW acts organized by category.

    Value object that encapsulates detected acts with type-safe access.
    """

    _acts: dict[NSFWCategory, list[str]] = field(default_factory=dict)

    def add_act(self, category: NSFWCategory, keyword: str):
        """Add a detected act to a category."""
        if category not in self._acts:
            self._acts[category] = []
        if keyword not in self._acts[category]:
            self._acts[category].append(keyword)

    def add_acts(self, category: NSFWCategory, keywords: list[str]):
        """Add multiple detected acts to a category."""
        for keyword in keywords:
            self.add_act(category, keyword)

    def get_acts(self, category: NSFWCategory) -> list[str]:
        """Get acts for a specific category."""
        return self._acts.get(category, [])

    def get_all_acts(self) -> dict[NSFWCategory, list[str]]:
        """Get all detected acts."""
        return self._acts.copy()

    def get_categories(self) -> list[NSFWCategory]:
        """Get categories with detected acts."""
        return list(self._acts.keys())

    def has_acts_in_category(self, category: NSFWCategory) -> bool:
        """Check if category has detected acts."""
        return category in self._acts and len(self._acts[category]) > 0

    def total_acts_count(self) -> int:
        """Get total number of detected acts across all categories."""
        return sum(len(acts) for acts in self._acts.values())

    def is_empty(self) -> bool:
        """Check if no acts detected."""
        return len(self._acts) == 0

    def __len__(self) -> int:
        """Number of categories with detected acts."""
        return len(self._acts)

    def __iter__(self):
        """Iterate over (category, acts) tuples."""
        return iter(self._acts.items())


@dataclass
class NSFWAnalysis:
    """Analysis of NSFW content in a prompt."""

    # Detection
    is_nsfw: bool
    confidence: float
    categories: list[NSFWCategory] = field(default_factory=list)

    # Specific acts found
    detected_acts: DetectedActs = field(default_factory=DetectedActs)

    # For LoRA matching
    recommended_lora_tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate analysis."""
        assert 0.0 <= self.confidence <= 1.0, "Confidence must be between 0 and 1"

    @property
    def category_names(self) -> list[str]:
        """Get category names as strings."""
        return [cat.value for cat in self.categories]

    @property
    def has_explicit_content(self) -> bool:
        """Whether explicit sexual content is present."""
        explicit_categories = {
            NSFWCategory.ORAL, NSFWCategory.ANAL, NSFWCategory.VAGINAL,
            NSFWCategory.CUM, NSFWCategory.GROUP, NSFWCategory.MASTURBATION
        }
        return any(cat in explicit_categories for cat in self.categories)

    @property
    def has_nudity_only(self) -> bool:
        """Whether only nudity is present (no sexual acts)."""
        return (self.is_nsfw and
                NSFWCategory.NUDITY in self.categories and
                not self.has_explicit_content)


def classify_token(token: str) -> TokenClassification:
    """
    Classify a prompt token by priority and category.

    Args:
        token: Token to classify

    Returns:
        Token classification with priority and category
    """
    token_lower = token.lower().strip()

    # Remove weight syntax for matching
    clean_token = token_lower
    weight = 1.0
    is_weight_syntax = False

    # Check for (token:weight) syntax
    if ":" in token_lower and "(" in token_lower:
        is_weight_syntax = True
        # Extract token and weight
        match = re.match(r'\(([^:]+):([0-9.]+)\)', token_lower)
        if match:
            clean_token = match.group(1).strip()
            weight = float(match.group(2))

    # Remove parentheses for matching
    clean_token = clean_token.replace("(", "").replace(")", "").strip()

    # Check if quality tag
    is_quality = any(qt.value in clean_token for qt in QualityTag)

    # Determine category and priority
    category = None
    priority = PromptTokenPriority.MEDIUM  # Default

    # Check NSFW categories (HIGH priority)
    for nsfw_cat in NSFW_REGISTRY.get_all_categories():
        if NSFW_REGISTRY.matches_any_keyword(clean_token, nsfw_cat):
            category = nsfw_cat
            priority = PromptTokenPriority.HIGH
            break

    # Check core content (CRITICAL priority)
    if not category:
        if any(kw in clean_token for kw in CORE_CONTENT_KEYWORDS):
            priority = PromptTokenPriority.CRITICAL

    # Check context (MEDIUM priority)
    if not category and priority == PromptTokenPriority.MEDIUM:
        if any(kw in clean_token for kw in CONTEXT_KEYWORDS):
            priority = PromptTokenPriority.MEDIUM

    # Quality tags = LOW priority
    if is_quality:
        priority = PromptTokenPriority.LOW

    return TokenClassification(
        token=token,
        priority=priority,
        category=category,
        is_quality_tag=is_quality,
        is_weight_syntax=is_weight_syntax,
        weight=weight
    )


def analyze_nsfw_content(prompt: str) -> NSFWAnalysis:
    """
    Analyze prompt for NSFW content.

    Args:
        prompt: Prompt to analyze

    Returns:
        NSFW analysis with detected categories and acts
    """
    prompt_lower = prompt.lower()

    detected_acts = DetectedActs()
    categories: list[NSFWCategory] = []

    # Check each NSFW category
    for category in NSFW_REGISTRY.get_all_categories():
        found_keywords = NSFW_REGISTRY.find_matching_keywords(prompt_lower, category)
        if found_keywords:
            detected_acts.add_acts(category, found_keywords)
            categories.append(category)

    # Determine if NSFW and confidence
    is_nsfw = len(categories) > 0

    # Confidence based on number of categories and keywords found
    if not is_nsfw:
        confidence = 0.0
    else:
        total_keywords = detected_acts.total_acts_count()
        # More keywords = higher confidence
        confidence = min(0.5 + (total_keywords * 0.1), 1.0)

    # Generate recommended LoRA tags
    recommended_tags = []
    for category in categories:
        recommended_tags.append(category.value)
        # Add most common keyword from this category
        category_acts = detected_acts.get_acts(category)
        if category_acts:
            recommended_tags.append(category_acts[0])

    # Add generic NSFW tags if explicit content
    if any(cat in {NSFWCategory.ORAL, NSFWCategory.ANAL, NSFWCategory.VAGINAL}
           for cat in categories):
        recommended_tags.extend(["nsfw", "explicit", "sex"])

    return NSFWAnalysis(
        is_nsfw=is_nsfw,
        confidence=confidence,
        categories=categories,
        detected_acts=detected_acts,
        recommended_lora_tags=list(set(recommended_tags))  # Remove duplicates
    )
