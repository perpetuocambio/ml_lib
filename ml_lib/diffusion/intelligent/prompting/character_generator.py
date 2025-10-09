"""Character generator with attribute shuffling for diversity.

PURPOSE: Counter CivitAI model bias toward white subjects by generating
diverse characters with consistent ethnic features.
"""

import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class GeneratedCharacter:
    """A generated character with all attributes."""

    # Core identity
    age: int
    age_keywords: List[str]

    # Ethnicity and skin (MUST be consistent)
    skin_tone: str
    skin_keywords: List[str]
    skin_prompt_weight: float
    ethnicity: str
    ethnicity_keywords: List[str]
    ethnicity_prompt_weight: float

    # Physical features
    eye_color: str
    eye_keywords: List[str]
    hair_color: str
    hair_keywords: List[str]
    hair_texture: str
    hair_texture_keywords: List[str]
    hair_texture_weight: float

    # Body
    body_type: str
    body_keywords: List[str]
    breast_size: str
    breast_keywords: List[str]

    # Scene
    setting: str
    setting_keywords: List[str]
    lighting_suggestions: List[str]
    pose: str
    pose_keywords: List[str]
    pose_complexity: str
    pose_explicit: bool

    # Age-related features
    age_features: List[str]

    def to_prompt(self, include_explicit: bool = True) -> str:
        """
        Generate prompt from character attributes.

        Args:
            include_explicit: Include explicit pose keywords

        Returns:
            Formatted prompt string
        """
        parts = []

        # Age (weighted)
        age_str = ", ".join(self.age_keywords)
        parts.append(f"({age_str}:1.2)")

        # Ethnicity (HIGH weight to counter bias)
        ethnicity_str = ", ".join(self.ethnicity_keywords)
        parts.append(f"({ethnicity_str}:{self.ethnicity_prompt_weight})")

        # Skin tone (HIGH weight to counter bias)
        skin_str = ", ".join(self.skin_keywords)
        parts.append(f"({skin_str}:{self.skin_prompt_weight})")

        # Eye color
        eye_str = ", ".join(self.eye_keywords)
        parts.append(eye_str)

        # Hair (color + texture with weight for non-straight hair)
        hair_color_str = ", ".join(self.hair_keywords)
        hair_texture_str = ", ".join(self.hair_texture_keywords)

        if self.hair_texture_weight > 1.0:
            parts.append(f"{hair_color_str}, ({hair_texture_str}:{self.hair_texture_weight})")
        else:
            parts.append(f"{hair_color_str}, {hair_texture_str}")

        # Age features
        if self.age_features:
            age_features_str = ", ".join(self.age_features)
            parts.append(f"({age_features_str}:1.1)")

        # Body
        body_str = ", ".join(self.body_keywords)
        parts.append(body_str)

        breast_str = ", ".join(self.breast_keywords)
        parts.append(f"({breast_str}:1.2)")

        # Pose
        if include_explicit or not self.pose_explicit:
            pose_str = ", ".join(self.pose_keywords)
            if self.pose_complexity == "high":
                parts.append(f"({pose_str}:1.3)")
            elif self.pose_complexity == "medium":
                parts.append(f"({pose_str}:1.2)")
            else:
                parts.append(pose_str)

        # Setting
        setting_str = ", ".join(self.setting_keywords)
        parts.append(setting_str)

        # Lighting
        lighting_str = ", ".join(self.lighting_suggestions[:2])  # Max 2
        parts.append(f"({lighting_str}:1.1)")

        return ", ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "age": self.age,
            "ethnicity": self.ethnicity,
            "skin_tone": self.skin_tone,
            "eye_color": self.eye_color,
            "hair_color": self.hair_color,
            "hair_texture": self.hair_texture,
            "body_type": self.body_type,
            "breast_size": self.breast_size,
            "setting": self.setting,
            "pose": self.pose,
            "prompt": self.to_prompt(),
        }


class CharacterGenerator:
    """Generates diverse characters by shuffling attributes from YAML config.

    CRITICAL: Enforces ethnic consistency and counters racial bias.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize character generator.

        Args:
            config_path: Path to character_attributes.yaml
        """
        if config_path is None:
            # Default to central config location
            module_dir = Path(__file__).parent
            project_root = module_dir.parent.parent.parent.parent
            config_path = project_root / "config" / "intelligent_prompting" / "character_attributes.yaml"

        self.config_path = config_path

        # Load configuration
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # Extract sections
        self.skin_tones = self.config.get("skin_tones", {})
        self.ethnicities = self.config.get("ethnicities", {})
        self.eye_colors = self.config.get("eye_colors", {})
        self.hair_colors = self.config.get("hair_colors", {})
        self.hair_textures = self.config.get("hair_textures", {})
        self.body_types = self.config.get("body_types", {})
        self.breast_sizes = self.config.get("breast_sizes", {})
        self.age_ranges = self.config.get("age_ranges", {})
        self.settings = self.config.get("settings", {})
        self.poses = self.config.get("poses", {})

        # Randomization rules
        self.randomization_rules = self.config.get("randomization_rules", {})

        # Diversity enforcement
        self.enforce_diversity = True
        self.diversity_targets = self.randomization_rules.get("diversity_targets", {})

        logger.info(f"CharacterGenerator initialized from {config_path}")
        logger.info(
            f"Diversity enforcement: {self.enforce_diversity} "
            f"(min non-white: {self.diversity_targets.get('min_non_white_percentage', 0):.0%})"
        )

    def generate(
        self,
        enforce_diversity: Optional[bool] = None,
        explicit_poses_only: bool = False,
        age_range: Optional[tuple[int, int]] = None,
    ) -> GeneratedCharacter:
        """
        Generate a random character with consistent attributes.

        Args:
            enforce_diversity: Override diversity enforcement
            explicit_poses_only: Only select explicit sexual poses
            age_range: Optional (min_age, max_age) constraint

        Returns:
            Generated character
        """
        if enforce_diversity is None:
            enforce_diversity = self.enforce_diversity

        # 1. Select ethnicity and skin tone (MUST be consistent)
        ethnicity_key, ethnicity_data = self._select_ethnicity(enforce_diversity)
        skin_tone_key, skin_tone_data = self._select_skin_tone(ethnicity_key, enforce_diversity)

        # 2. Select age
        age, age_keywords, age_features = self._select_age(age_range)

        # 3. Select hair (color + texture consistent with ethnicity)
        hair_color_key, hair_color_data = self._select_hair_color(age, ethnicity_key)
        hair_texture_key, hair_texture_data = self._select_hair_texture(ethnicity_key)

        # 4. Select eye color
        eye_color_key, eye_color_data = self._select_eye_color()

        # 5. Select body
        body_type_key, body_type_data = self._select_body_type()
        breast_size_key, breast_size_data = self._select_breast_size()

        # 6. Select setting and pose
        setting_key, setting_data = self._select_setting()
        pose_key, pose_data = self._select_pose(explicit_poses_only)

        # Build character
        character = GeneratedCharacter(
            age=age,
            age_keywords=age_keywords,
            skin_tone=skin_tone_key,
            skin_keywords=skin_tone_data["keywords"],
            skin_prompt_weight=skin_tone_data.get("prompt_weight", 1.2),
            ethnicity=ethnicity_key,
            ethnicity_keywords=ethnicity_data["keywords"],
            ethnicity_prompt_weight=ethnicity_data.get("prompt_weight", 1.0),
            eye_color=eye_color_key,
            eye_keywords=eye_color_data["keywords"],
            hair_color=hair_color_key,
            hair_keywords=hair_color_data["keywords"],
            hair_texture=hair_texture_key,
            hair_texture_keywords=hair_texture_data["keywords"],
            hair_texture_weight=hair_texture_data.get("prompt_weight", 1.0),
            body_type=body_type_key,
            body_keywords=body_type_data["keywords"],
            breast_size=breast_size_key,
            breast_keywords=breast_size_data["keywords"],
            setting=setting_key,
            setting_keywords=setting_data["keywords"],
            lighting_suggestions=setting_data.get("lighting_suggestions", ["natural light"]),
            pose=pose_key,
            pose_keywords=pose_data["keywords"],
            pose_complexity=pose_data.get("complexity", "medium"),
            pose_explicit=pose_data.get("explicit", False),
            age_features=age_features,
        )

        logger.info(
            f"Generated character: {character.age}y {character.ethnicity} "
            f"with {character.skin_tone} skin, {character.hair_texture} {character.hair_color} hair"
        )

        return character

    def _select_ethnicity(self, enforce_diversity: bool) -> tuple[str, Dict]:
        """Select ethnicity with diversity enforcement."""
        if enforce_diversity:
            # Apply diversity targets
            min_non_white = self.diversity_targets.get("min_non_white_percentage", 0.70)

            # Roll for non-white
            if random.random() < min_non_white:
                # Select from non-white ethnicities
                non_white = {k: v for k, v in self.ethnicities.items() if k != "caucasian"}
                return self._weighted_choice(non_white)
            else:
                # Allow white
                return self._weighted_choice(self.ethnicities)
        else:
            return self._weighted_choice(self.ethnicities)

    def _select_skin_tone(self, ethnicity_key: str, enforce_diversity: bool) -> tuple[str, Dict]:
        """Select skin tone consistent with ethnicity."""
        # Get ethnicity associations
        ethnicity_data = self.ethnicities[ethnicity_key]

        # Filter skin tones compatible with ethnicity
        compatible_skin_tones = {}

        for skin_key, skin_data in self.skin_tones.items():
            ethnicity_assocs = skin_data.get("ethnicity_associations", [])

            # Check if ethnicity matches
            if any(eth in ethnicity_key or ethnicity_key in eth for eth in ethnicity_assocs):
                compatible_skin_tones[skin_key] = skin_data

        # If no compatible found, allow all
        if not compatible_skin_tones:
            compatible_skin_tones = self.skin_tones

        # Enforce diversity (prefer darker skin tones)
        if enforce_diversity:
            min_dark_skin = self.diversity_targets.get("min_dark_skin_percentage", 0.30)

            if random.random() < min_dark_skin:
                # Force medium-dark or dark
                dark_tones = {
                    k: v
                    for k, v in compatible_skin_tones.items()
                    if k in ["medium", "medium_dark", "dark"]
                }
                if dark_tones:
                    compatible_skin_tones = dark_tones

        return self._weighted_choice(compatible_skin_tones)

    def _select_age(self, age_range: Optional[tuple[int, int]]) -> tuple[int, List[str], List[str]]:
        """Select age and related keywords."""
        # Filter age ranges
        if age_range:
            min_age, max_age = age_range
            compatible_ages = {
                k: v
                for k, v in self.age_ranges.items()
                if v["age_min"] >= min_age and v["age_max"] <= max_age
            }
        else:
            compatible_ages = self.age_ranges

        # Select age range
        age_range_key, age_range_data = self._weighted_choice(compatible_ages)

        # Random age within range
        age = random.randint(age_range_data["age_min"], age_range_data["age_max"])

        # Get age features
        age_features = self._get_age_features(age)

        return age, age_range_data["keywords"], age_features

    def _get_age_features(self, age: int) -> List[str]:
        """Get age-appropriate features."""
        if age < 40:
            return ["minimal age lines", "youthful mature", "smooth skin"]
        elif age < 50:
            return ["subtle age lines", "laugh lines", "mature beauty"]
        elif age < 60:
            return ["age lines", "mature features", "natural aging"]
        else:
            return ["prominent age lines", "grey hair", "senior features", "aged beauty"]

    def _select_hair_color(self, age: int, ethnicity_key: str) -> tuple[str, Dict]:
        """Select hair color appropriate for age."""
        # For 60+, increase probability of grey/white
        if age >= 60:
            grey_options = {
                k: v for k, v in self.hair_colors.items() if k in ["grey_silver", "white"]
            }

            if random.random() < 0.6:  # 60% chance of grey for 60+
                return self._weighted_choice(grey_options)

        # For 50-60, possible grey
        elif age >= 50:
            if random.random() < 0.3:  # 30% chance of grey
                return "grey_silver", self.hair_colors["grey_silver"]

        # Otherwise normal distribution
        return self._weighted_choice(self.hair_colors)

    def _select_hair_texture(self, ethnicity_key: str) -> tuple[str, Dict]:
        """Select hair texture compatible with ethnicity."""
        # Filter compatible textures
        compatible_textures = {}

        for texture_key, texture_data in self.hair_textures.items():
            ethnicity_fit = texture_data.get("ethnicity_fit", [])

            if not ethnicity_fit or ethnicity_key in ethnicity_fit:
                compatible_textures[texture_key] = texture_data

        if not compatible_textures:
            compatible_textures = self.hair_textures

        return self._weighted_choice(compatible_textures)

    def _select_eye_color(self) -> tuple[str, Dict]:
        """Select eye color."""
        return self._weighted_choice(self.eye_colors)

    def _select_body_type(self) -> tuple[str, Dict]:
        """Select body type."""
        return self._weighted_choice(self.body_types)

    def _select_breast_size(self) -> tuple[str, Dict]:
        """Select breast size."""
        return self._weighted_choice(self.breast_sizes)

    def _select_setting(self) -> tuple[str, Dict]:
        """Select setting/location."""
        return self._weighted_choice(self.settings)

    def _select_pose(self, explicit_only: bool = False) -> tuple[str, Dict]:
        """Select pose."""
        if explicit_only:
            explicit_poses = {k: v for k, v in self.poses.items() if v.get("explicit", False)}
            return self._weighted_choice(explicit_poses)
        else:
            return self._weighted_choice(self.poses)

    def _weighted_choice(self, items: Dict[str, Dict]) -> tuple[str, Dict]:
        """Select item based on probability weights."""
        if not items:
            raise ValueError("No items to choose from")

        # Extract probabilities
        keys = list(items.keys())
        weights = [items[k].get("probability", 1.0 / len(items)) for k in keys]

        # Normalize weights
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]

        # Random choice
        selected_key = random.choices(keys, weights=probabilities, k=1)[0]

        return selected_key, items[selected_key]

    def generate_batch(self, count: int, **kwargs) -> List[GeneratedCharacter]:
        """
        Generate multiple characters.

        Args:
            count: Number of characters to generate
            **kwargs: Arguments passed to generate()

        Returns:
            List of characters
        """
        return [self.generate(**kwargs) for _ in range(count)]
