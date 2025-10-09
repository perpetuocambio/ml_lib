"""Character generator with attribute shuffling for diversity.

PURPOSE: Counter CivitAI model bias toward white subjects by generating
diverse characters with consistent ethnic features.
"""

import logging
import random
from typing import Dict, List, Any, Optional

import yaml
from pathlib import Path

from ml_lib.diffusion.intelligent.prompting.entities import (
    GeneratedCharacter,
    AttributeConfig,
    CharacterAttributeSet
)

logger = logging.getLogger(__name__)


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
            raw_config = yaml.safe_load(f)

        # Convert raw config to structured objects
        self.attribute_set = self._load_attribute_set(raw_config)
        
        # Extract sections for backward compatibility
        self.skin_tones = {k: self._attr_to_dict(v) for k, v in self.attribute_set.skin_tones.items()}
        self.ethnicities = {k: self._attr_to_dict(v) for k, v in self.attribute_set.ethnicities.items()}
        self.eye_colors = {k: self._attr_to_dict(v) for k, v in self.attribute_set.eye_colors.items()}
        self.hair_colors = {k: self._attr_to_dict(v) for k, v in self.attribute_set.hair_colors.items()}
        self.hair_textures = {k: self._attr_to_dict(v) for k, v in self.attribute_set.hair_textures.items()}
        self.body_types = {k: self._attr_to_dict(v) for k, v in self.attribute_set.body_types.items()}
        self.breast_sizes = {k: self._attr_to_dict(v) for k, v in self.attribute_set.breast_sizes.items()}
        self.age_ranges = {k: self._attr_to_dict(v) for k, v in self.attribute_set.age_ranges.items()}
        self.settings = {k: self._attr_to_dict(v) for k, v in self.attribute_set.settings.items()}
        self.poses = {k: self._attr_to_dict(v) for k, v in self.attribute_set.poses.items()}
        self.clothing_styles = {k: self._attr_to_dict(v) for k, v in raw_config.get("clothing_styles", {}).items()}
        self.clothing_conditions = {k: self._attr_to_dict(v) for k, v in raw_config.get("clothing_conditions", {}).items()}
        self.clothing_details = {k: self._attr_to_dict(v) for k, v in raw_config.get("clothing_details", {}).items()}
        self.cosplay_styles = {k: self._attr_to_dict(v) for k, v in raw_config.get("cosplay_styles", {}).items()}
        self.accessories = {k: self._attr_to_dict(v) for k, v in raw_config.get("accessories", {}).items()}
        self.erotic_toys = {k: self._attr_to_dict(v) for k, v in raw_config.get("erotic_toys", {}).items()}
        self.activities = {k: self._attr_to_dict(v) for k, v in raw_config.get("activities", {}).items()}
        self.weather_conditions = {k: self._attr_to_dict(v) for k, v in raw_config.get("weather_conditions", {}).items()}
        self.emotional_states = {k: self._attr_to_dict(v) for k, v in raw_config.get("emotional_states", {}).items()}
        self.environment_details = {k: self._attr_to_dict(v) for k, v in raw_config.get("environment_details", {}).items()}
        self.artistic_styles = {k: self._attr_to_dict(v) for k, v in raw_config.get("artistic_styles", {}).items()}
        self.physical_features = {k: self._attr_to_dict(v) for k, v in raw_config.get("physical_features", {}).items()}
        self.body_sizes = {k: self._attr_to_dict(v) for k, v in raw_config.get("body_sizes", {}).items()}
        self.aesthetic_styles = {k: self._attr_to_dict(v) for k, v in raw_config.get("aesthetic_styles", {}).items()}
        self.fantasy_races = {k: self._attr_to_dict(v) for k, v in raw_config.get("fantasy_races", {}).items()}
        self.special_effects = {k: self._attr_to_dict(v) for k, v in raw_config.get("special_effects", {}).items()}

        # Randomization rules
        self.randomization_rules = self.attribute_set.randomization_rules

        # Diversity enforcement
        self.enforce_diversity = True
        self.diversity_targets = self.randomization_rules.get("diversity_targets", {})

        logger.info(f"CharacterGenerator initialized from {config_path}")
        logger.info(
            f"Diversity enforcement: {self.enforce_diversity} "
            f"(min non-white: {self.diversity_targets.get('min_non_white_percentage', 0):.0%})"
        )
    
    def _load_attribute_set(self, raw_config: Dict[str, Any]) -> CharacterAttributeSet:
        """Load and convert raw config to structured attribute set."""
        def dict_to_attr_config(attr_dict: Dict[str, Any]) -> AttributeConfig:
            """Convert dictionary to AttributeConfig."""
            return AttributeConfig(
                keywords=attr_dict.get("keywords", []),
                probability=attr_dict.get("probability", 1.0),
                prompt_weight=attr_dict.get("prompt_weight", 1.0),
                ethnicity_associations=attr_dict.get("ethnicity_associations"),
                min_age=attr_dict.get("min_age", 18),
                max_age=attr_dict.get("max_age", 80),
                ethnicity_fit=attr_dict.get("ethnicity_fit"),
                age_features=attr_dict.get("age_features"),
                lighting_suggestions=attr_dict.get("lighting_suggestions"),
                complexity=attr_dict.get("complexity", "medium"),
                explicit=attr_dict.get("explicit", False),
                age_min=attr_dict.get("age_min"),
                age_max=attr_dict.get("age_max")
            )
        
        return CharacterAttributeSet(
            skin_tones={k: dict_to_attr_config(v) for k, v in raw_config.get("skin_tones", {}).items()},
            ethnicities={k: dict_to_attr_config(v) for k, v in raw_config.get("ethnicities", {}).items()},
            eye_colors={k: dict_to_attr_config(v) for k, v in raw_config.get("eye_colors", {}).items()},
            hair_colors={k: dict_to_attr_config(v) for k, v in raw_config.get("hair_colors", {}).items()},
            hair_textures={k: dict_to_attr_config(v) for k, v in raw_config.get("hair_textures", {}).items()},
            body_types={k: dict_to_attr_config(v) for k, v in raw_config.get("body_types", {}).items()},
            breast_sizes={k: dict_to_attr_config(v) for k, v in raw_config.get("breast_sizes", {}).items()},
            age_ranges={k: dict_to_attr_config(v) for k, v in raw_config.get("age_ranges", {}).items()},
            settings={k: dict_to_attr_config(v) for k, v in raw_config.get("settings", {}).items()},
            poses={k: dict_to_attr_config(v) for k, v in raw_config.get("poses", {}).items()},
            clothing_styles={k: dict_to_attr_config(v) for k, v in raw_config.get("clothing_styles", {}).items()},
            clothing_conditions={k: dict_to_attr_config(v) for k, v in raw_config.get("clothing_conditions", {}).items()},
            clothing_details={k: dict_to_attr_config(v) for k, v in raw_config.get("clothing_details", {}).items()},
            cosplay_styles={k: dict_to_attr_config(v) for k, v in raw_config.get("cosplay_styles", {}).items()},
            accessories={k: dict_to_attr_config(v) for k, v in raw_config.get("accessories", {}).items()},
            erotic_toys={k: dict_to_attr_config(v) for k, v in raw_config.get("erotic_toys", {}).items()},
            activities={k: dict_to_attr_config(v) for k, v in raw_config.get("activities", {}).items()},
            weather_conditions={k: dict_to_attr_config(v) for k, v in raw_config.get("weather_conditions", {}).items()},
            emotional_states={k: dict_to_attr_config(v) for k, v in raw_config.get("emotional_states", {}).items()},
            environment_details={k: dict_to_attr_config(v) for k, v in raw_config.get("environment_details", {}).items()},
            artistic_styles={k: dict_to_attr_config(v) for k, v in raw_config.get("artistic_styles", {}).items()},
            physical_features={k: dict_to_attr_config(v) for k, v in raw_config.get("physical_features", {}).items()},
            body_sizes={k: dict_to_attr_config(v) for k, v in raw_config.get("body_sizes", {}).items()},
            aesthetic_styles={k: dict_to_attr_config(v) for k, v in raw_config.get("aesthetic_styles", {}).items()},
            fantasy_races={k: dict_to_attr_config(v) for k, v in raw_config.get("fantasy_races", {}).items()},
            special_effects={k: dict_to_attr_config(v) for k, v in raw_config.get("special_effects", {}).items()},
            randomization_rules=raw_config.get("randomization_rules", {})
        )
    
    def _attr_to_dict(self, attr: AttributeConfig) -> Dict[str, Any]:
        """Convert AttributeConfig back to dictionary for backward compatibility."""
        return {
            "keywords": attr.keywords,
            "probability": attr.probability,
            "prompt_weight": attr.prompt_weight,
            "ethnicity_associations": attr.ethnicity_associations,
            "min_age": attr.min_age,
            "max_age": attr.max_age,
            "ethnicity_fit": attr.ethnicity_fit,
            "age_features": attr.age_features,
            "lighting_suggestions": attr.lighting_suggestions,
            "complexity": attr.complexity,
            "explicit": attr.explicit,
            "age_min": attr.age_min,
            "age_max": attr.age_max
        }

    def generate(
        self,
        enforce_diversity: Optional[bool] = None,
        explicit_poses_only: bool = False,
        age_range: Optional[tuple[int, int]] = None,
        include_accessories: bool = True,
        include_toys: bool = True,
        include_clothing: bool = True,
    ) -> GeneratedCharacter:
        """
        Generate a random character with consistent attributes.

        Args:
            enforce_diversity: Override diversity enforcement
            explicit_poses_only: Only select explicit sexual poses
            age_range: Optional (min_age, max_age) constraint
            include_accessories: Whether to include accessories
            include_toys: Whether to include erotic toys
            include_clothing: Whether to include clothing

        Returns:
            Generated character
        """
        if enforce_diversity is None:
            enforce_diversity = self.enforce_diversity

        # 1. Select ethnicity and skin tone (MUST be consistent)
        ethnicity_key, ethnicity_config = self._select_ethnicity_config(enforce_diversity)
        skin_tone_key, skin_tone_config = self._select_skin_tone_config(ethnicity_key, enforce_diversity)

        # 2. Select age
        age, age_keywords, age_features = self._select_age_config(age_range)

        # 3. Select hair (color + texture consistent with ethnicity)
        hair_color_key, hair_color_config = self._select_hair_color_config(age, ethnicity_key)
        hair_texture_key, hair_texture_config = self._select_hair_texture_config(ethnicity_key)

        # 4. Select eye color
        eye_color_key, eye_color_config = self._select_eye_color_config()

        # 5. Select body
        body_type_key, body_type_config = self._select_body_type_config()
        breast_size_key, breast_size_config = self._select_breast_size_config()

        # 6. Select clothing style
        clothing_style_key, clothing_style_config = self._select_clothing_config() if include_clothing else ("nude", self.attribute_set.clothing_styles["nude"])

        # 7. Select accessories
        accessory_list = self._select_accessories_config() if include_accessories else []

        # 8. Select erotic toys (only if explicit content)
        toy_list = self._select_erotic_toys_config() if include_toys else []

        # 9. Select activity
        activity_key, activity_config = self._select_activity_config(explicit_poses_only)

        # 10. Select setting and pose
        setting_key, setting_config = self._select_setting_config()
        pose_key, pose_config = self._select_pose_config(explicit_poses_only)

        # 11. Select weather conditions
        weather_key, weather_config = self._select_weather_config()

        # 12. Select emotional state
        emotional_key, emotional_config = self._select_emotional_state_config()

        # 13. Select environment details
        environment_key, environment_config = self._select_environment_config()

        # 11. Select clothing condition
        clothing_condition_key, clothing_condition_config = self._select_clothing_condition_config()

        # 12. Select clothing details
        clothing_detail_key, clothing_detail_config = self._select_clothing_detail_config()

        # 13. Select cosplay style
        cosplay_key, cosplay_config = self._select_cosplay_config()

        # 14. Select weather conditions
        weather_key, weather_config = self._select_weather_config()

        # 15. Select emotional state
        emotional_key, emotional_config = self._select_emotional_state_config()

        # 16. Select environment details
        environment_key, environment_config = self._select_environment_config()

        # 17. Select artistic style
        artistic_style_key, artistic_style_config = self._select_artistic_style_config()

        # 18. Select physical features
        physical_features_key, physical_features_config = self._select_physical_features_config()

        # 19. Select body size
        body_size_key, body_size_config = self._select_body_size_config()

        # 20. Select aesthetic style
        aesthetic_style_key, aesthetic_style_config = self._select_aesthetic_style_config()

        # 21. Select fantasy race
        fantasy_race_key, fantasy_race_config = self._select_fantasy_race_config()

        # 22. Select special effects
        special_effects_key, special_effects_config = self._select_special_effects_config()

        # Build character using new approach
        character = GeneratedCharacter(
            age=age,
            age_keywords=age_keywords,
            skin_tone=skin_tone_key,
            skin_keywords=skin_tone_config.keywords,
            skin_prompt_weight=skin_tone_config.prompt_weight,
            ethnicity=ethnicity_key,
            ethnicity_keywords=ethnicity_config.keywords,
            ethnicity_prompt_weight=ethnicity_config.prompt_weight,
            artistic_style=artistic_style_key,
            artistic_keywords=artistic_style_config.keywords,
            eye_color=eye_color_key,
            eye_keywords=eye_color_config.keywords,
            hair_color=hair_color_key,
            hair_keywords=hair_color_config.keywords,
            hair_texture=hair_texture_key,
            hair_texture_keywords=hair_texture_config.keywords,
            hair_texture_weight=hair_texture_config.prompt_weight,
            body_type=body_type_key,
            body_keywords=body_type_config.keywords,
            breast_size=breast_size_key,
            breast_keywords=breast_size_config.keywords,
            body_size=body_size_key,
            body_size_keywords=body_size_config.keywords,
            physical_features=physical_features_key,
            physical_feature_keywords=physical_features_config.keywords,
            clothing_style=clothing_style_key,
            clothing_keywords=clothing_style_config.keywords,
            clothing_condition=clothing_condition_key,
            clothing_condition_keywords=clothing_condition_config.keywords,
            clothing_details=clothing_detail_key,
            clothing_detail_keywords=clothing_detail_config.keywords,
            aesthetic_style=aesthetic_style_key,
            aesthetic_keywords=aesthetic_style_config.keywords,
            fantasy_race=fantasy_race_key,
            fantasy_race_keywords=fantasy_race_config.keywords,
            special_effects=special_effects_key,
            special_effect_keywords=special_effects_config.keywords,
            cosplay_style=cosplay_key,
            cosplay_keywords=cosplay_config.keywords,
            accessories=accessory_list,
            accessory_keywords=[word for acc in accessory_list for word in self.attribute_set.accessories[acc].keywords],
            erotic_toys=toy_list,
            toy_keywords=[word for toy in toy_list for word in self.attribute_set.erotic_toys[toy].keywords] if toy_list else [],
            activity=activity_key,
            activity_keywords=activity_config.keywords,
            weather=weather_key,
            weather_keywords=weather_config.keywords,
            emotional_state=emotional_key,
            emotional_keywords=emotional_config.keywords,
            environment=environment_key,
            environment_keywords=environment_config.keywords,
            setting=setting_key,
            setting_keywords=setting_config.keywords,
            lighting_suggestions=setting_config.lighting_suggestions,
            pose=pose_key,
            pose_keywords=pose_config.keywords,
            pose_complexity=pose_config.complexity,
            pose_explicit=pose_config.explicit,
            age_features=age_features,
        )

        logger.info(
            f"Generated character: {character.age}y {character.ethnicity} "
            f"with {character.skin_tone} skin, {character.hair_texture} {character.hair_color} hair, "
            f"clothing: {character.clothing_style}, accessories: {len(character.accessories)}, "
            f"toys: {len(character.erotic_toys)}, activity: {character.activity}"
        )

        return character

    def _select_ethnicity_config(self, enforce_diversity: bool) -> tuple[str, AttributeConfig]:
        """Select ethnicity with diversity enforcement."""
        if enforce_diversity:
            # Apply diversity targets
            min_non_white = self.diversity_targets.get("min_non_white_percentage", 0.60)

            # Roll for non-white
            if random.random() < min_non_white:
                # Select from non-white ethnicities
                non_white_configs = {
                    k: v for k, v in self.attribute_set.ethnicities.items() 
                    if k != "caucasian"
                }
                return self._weighted_choice_config(non_white_configs)
            else:
                # Allow white
                return self._weighted_choice_config(self.attribute_set.ethnicities)
        else:
            return self._weighted_choice_config(self.attribute_set.ethnicities)

    def _select_skin_tone_config(self, ethnicity_key: str, enforce_diversity: bool) -> tuple[str, AttributeConfig]:
        """Select skin tone consistent with ethnicity."""
        # Get ethnicity associations
        ethnicity_config = self.attribute_set.ethnicities[ethnicity_key]

        # Filter skin tones compatible with ethnicity
        compatible_skin_tones = {}

        for skin_key, skin_config in self.attribute_set.skin_tones.items():
            ethnicity_assocs = skin_config.ethnicity_associations

            # Check if ethnicity matches
            if any(eth in ethnicity_key or ethnicity_key in eth for eth in ethnicity_assocs):
                compatible_skin_tones[skin_key] = skin_config

        # If no compatible found, allow all
        if not compatible_skin_tones:
            compatible_skin_tones = self.attribute_set.skin_tones

        # Enforce diversity (prefer darker skin tones)
        if enforce_diversity:
            min_dark_skin = self.diversity_targets.get("min_dark_skin_percentage", 0.25)

            if random.random() < min_dark_skin:
                # Force medium-dark or dark
                dark_tones = {
                    k: v
                    for k, v in compatible_skin_tones.items()
                    if k in ["medium", "medium_dark", "dark"]
                }
                if dark_tones:
                    compatible_skin_tones = dark_tones

        return self._weighted_choice_config(compatible_skin_tones)

    def _select_age_config(self, age_range: Optional[tuple[int, int]]) -> tuple[int, List[str], List[str]]:
        """Select age and related keywords."""
        # Filter age ranges
        if age_range:
            min_age, max_age = age_range
            compatible_ages = {
                k: v
                for k, v in self.attribute_set.age_ranges.items()
                if v.age_min >= min_age and v.age_max <= max_age
            }
        else:
            compatible_ages = self.attribute_set.age_ranges

        # Select age range
        age_range_key, age_range_config = self._weighted_choice_config(compatible_ages)

        # Random age within range
        age = random.randint(age_range_config.age_min or 18, age_range_config.age_max or 80)

        # Get age features
        age_features = self._get_age_features_config(age)

        return age, age_range_config.keywords, age_features

    def _get_age_features_config(self, age: int) -> List[str]:
        """Get age-appropriate features."""
        if age < 40:
            return ["minimal age lines", "youthful mature", "smooth skin"]
        elif age < 50:
            return ["subtle age lines", "laugh lines", "mature beauty"]
        elif age < 60:
            return ["age lines", "mature features", "natural aging"]
        else:
            return ["prominent age lines", "grey hair", "senior features", "aged beauty"]

    def _select_hair_color_config(self, age: int, ethnicity_key: str) -> tuple[str, AttributeConfig]:
        """Select hair color appropriate for age."""
        # For 60+, increase probability of grey/white
        if age >= 60:
            grey_options = {
                k: v for k, v in self.attribute_set.hair_colors.items() 
                if k in ["grey_silver", "white"]
            }

            if random.random() < 0.6:  # 60% chance of grey for 60+
                return self._weighted_choice_config(grey_options)

        # For 50-60, possible grey
        elif age >= 50:
            if random.random() < 0.3:  # 30% chance of grey
                return "grey_silver", self.attribute_set.hair_colors["grey_silver"]

        # Otherwise normal distribution
        return self._weighted_choice_config(self.attribute_set.hair_colors)

    def _select_hair_texture_config(self, ethnicity_key: str) -> tuple[str, AttributeConfig]:
        """Select hair texture compatible with ethnicity."""
        # Filter compatible textures
        compatible_textures = {}

        for texture_key, texture_config in self.attribute_set.hair_textures.items():
            ethnicity_fit = texture_config.ethnicity_fit

            if not ethnicity_fit or ethnicity_key in ethnicity_fit:
                compatible_textures[texture_key] = texture_config

        if not compatible_textures:
            compatible_textures = self.attribute_set.hair_textures

        return self._weighted_choice_config(compatible_textures)

    def _select_eye_color_config(self) -> tuple[str, AttributeConfig]:
        """Select eye color."""
        return self._weighted_choice_config(self.attribute_set.eye_colors)

    def _select_body_type_config(self) -> tuple[str, AttributeConfig]:
        """Select body type."""
        return self._weighted_choice_config(self.attribute_set.body_types)

    def _select_breast_size_config(self) -> tuple[str, AttributeConfig]:
        """Select breast size."""
        return self._weighted_choice_config(self.attribute_set.breast_sizes)

    def _select_setting_config(self) -> tuple[str, AttributeConfig]:
        """Select setting/location."""
        return self._weighted_choice_config(self.attribute_set.settings)

    def _select_pose_config(self, explicit_only: bool = False) -> tuple[str, AttributeConfig]:
        """Select pose."""
        if explicit_only:
            explicit_poses = {k: v for k, v in self.attribute_set.poses.items() if v.explicit}
            return self._weighted_choice_config(explicit_poses) if explicit_poses else self._weighted_choice_config(self.attribute_set.poses)
        else:
            return self._weighted_choice_config(self.attribute_set.poses)

    def _select_clothing_config(self) -> tuple[str, AttributeConfig]:
        """Select clothing style."""
        return self._weighted_choice_config(self.attribute_set.clothing_styles)

    def _select_accessories_config(self) -> List[str]:
        """Select accessories (multiple can be selected)."""
        selected_accessories = []
        
        # Probability of selecting each accessory category
        for acc_category, acc_config in self.attribute_set.accessories.items():
            # Higher probability if the config has higher probability value
            if random.random() < acc_config.probability * 0.5:  # Scale down for multiple selection
                selected_accessories.append(acc_category)
        
        # If no accessories selected but jewelry is common, add some jewelry
        if not selected_accessories and random.random() < 0.3:
            if "jewelry" in self.attribute_set.accessories:
                selected_accessories.append("jewelry")
        
        return selected_accessories

    def _select_erotic_toys_config(self) -> List[str]:
        """Select erotic toys (multiple can be selected)."""
        selected_toys = []
        
        # Probability of selecting each toy category
        for toy_category, toy_config in self.attribute_set.erotic_toys.items():
            # Higher probability if the config has higher probability value
            if random.random() < toy_config.probability * 0.3:  # Lower scale for explicit content
                selected_toys.append(toy_category)
        
        return selected_toys

    def _select_activity_config(self, explicit_only: bool = False) -> tuple[str, AttributeConfig]:
        """Select activity."""
        if explicit_only:
            # Only select explicit activities
            explicit_activities = {k: v for k, v in self.attribute_set.activities.items() if v.explicit}
            if explicit_activities:
                return self._weighted_choice_config(explicit_activities)
            else:
                # If no explicit activities, return any activity
                return self._weighted_choice_config(self.attribute_set.activities)
        else:
            return self._weighted_choice_config(self.attribute_set.activities)

    def _select_weather_config(self) -> tuple[str, AttributeConfig]:
        """Select weather conditions."""
        return self._weighted_choice_config(self.attribute_set.weather_conditions)

    def _select_clothing_condition_config(self) -> tuple[str, AttributeConfig]:
        """Select clothing condition."""
        return self._weighted_choice_config(self.attribute_set.clothing_conditions)

    def _select_clothing_detail_config(self) -> tuple[str, AttributeConfig]:
        """Select clothing details."""
        return self._weighted_choice_config(self.attribute_set.clothing_details)

    def _select_artistic_style_config(self) -> tuple[str, AttributeConfig]:
        """Select artistic style."""
        return self._weighted_choice_config(self.attribute_set.artistic_styles)

    def _select_physical_features_config(self) -> tuple[str, AttributeConfig]:
        """Select physical features."""
        return self._weighted_choice_config(self.attribute_set.physical_features)

    def _select_body_size_config(self) -> tuple[str, AttributeConfig]:
        """Select body size."""
        return self._weighted_choice_config(self.attribute_set.body_sizes)

    def _select_aesthetic_style_config(self) -> tuple[str, AttributeConfig]:
        """Select aesthetic style."""
        return self._weighted_choice_config(self.attribute_set.aesthetic_styles)

    def _select_fantasy_race_config(self) -> tuple[str, AttributeConfig]:
        """Select fantasy race."""
        return self._weighted_choice_config(self.attribute_set.fantasy_races)

    def _select_special_effects_config(self) -> tuple[str, AttributeConfig]:
        """Select special effects."""
        return self._weighted_choice_config(self.attribute_set.special_effects)

    def _select_environment_config(self) -> tuple[str, AttributeConfig]:
        """Select environment details."""
        return self._weighted_choice_config(self.attribute_set.environment_details)

    def _weighted_choice_config(self, items: Dict[str, AttributeConfig]) -> tuple[str, AttributeConfig]:
        """Select item based on probability weights."""
        if not items:
            raise ValueError("No items to choose from")

        # Extract probabilities
        keys = list(items.keys())
        weights = [items[k].probability for k in keys]

        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            # If all probabilities are 0, use equal weights
            probabilities = [1.0 / len(items) for _ in keys]
        else:
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
