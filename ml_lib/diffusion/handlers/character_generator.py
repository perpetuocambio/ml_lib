"""Character generator using class-based attribute system."""

import logging
import random
from typing import List, Optional

# Import from new models structure
from ml_lib.diffusion.models import (
    GeneratedCharacter,
    ValidationResult,
    GenerationPreferences,
)

# Import core attribute classes from models
from ml_lib.diffusion.models import AttributeType, AttributeDefinition
from ml_lib.diffusion.handlers import (
    CharacterAttributeSet,
    AttributeCollection,
    ConfigLoader,
)

logger = logging.getLogger(__name__)


class CharacterGenerator:
    """Character generator using class-based attribute system."""

    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        """
        Initialize enhanced character generator.

        Args:
            config_loader: Configuration loader (creates default if None)
        """
        if config_loader is None:
            config_loader = ConfigLoader()

        self.config_loader = config_loader
        self.attribute_set = config_loader.get_attribute_set()
        self._initialize_generation_rules()

    def _initialize_generation_rules(self):
        """Initialize generation rules and constraints."""
        # Define attribute priorities and dependencies
        self.attribute_priorities = {
            AttributeType.AGE_RANGE: 10,
            AttributeType.ETHNICITY: 9,
            AttributeType.SKIN_TONE: 8,
            AttributeType.BODY_TYPE: 7,
            AttributeType.BREAST_SIZE: 6,
            AttributeType.HAIR_COLOR: 5,
            AttributeType.HAIR_TEXTURE: 4,
            AttributeType.EYE_COLOR: 3,
            AttributeType.CLOTHING_STYLE: 2,
            AttributeType.AESTHETIC_STYLE: 1,
        }

        # Define attribute groups that should be selected together
        self.attribute_groups = [
            [AttributeType.AGE_RANGE, AttributeType.BODY_TYPE, AttributeType.BREAST_SIZE],
            [AttributeType.ETHNICITY, AttributeType.SKIN_TONE, AttributeType.HAIR_COLOR],
            [AttributeType.CLOTHING_STYLE, AttributeType.AESTHETIC_STYLE],
        ]

    def generate_character(self, preferences: Optional[GenerationPreferences] = None) -> GeneratedCharacter:
        """
        Generate an enhanced character with improved attribute selection.

        Args:
            preferences: Generation preferences

        Returns:
            Generated character
        """
        if preferences is None:
            preferences = GenerationPreferences()

        # Step 1: Select core identity attributes
        selected_attributes = self._select_core_attributes(preferences)

        # Step 2: Select compatible physical features
        physical_attributes = self._select_physical_attributes(selected_attributes, preferences)
        selected_attributes.extend(physical_attributes)

        # Step 3: Select clothing and style (with enhanced compatibility checking)
        style_attributes = self._select_style_attributes(selected_attributes, preferences)
        selected_attributes.extend(style_attributes)

        # Step 4: Select accessories and details
        detail_attributes = self._select_detail_attributes(selected_attributes, preferences)
        selected_attributes.extend(detail_attributes)

        # Step 5: Validate all selections
        validation_result = self.config_loader.validate_character_selection(selected_attributes)

        if not validation_result.is_valid:
            logger.warning(f"Character generation issues: {validation_result.issues}")
            # Attempt to resolve conflicts
            selected_attributes = self._resolve_conflicts(selected_attributes, validation_result)

        # Step 6: Create final character
        character = self._create_character_from_attributes(selected_attributes, preferences)

        return character

    def _select_core_attributes(self, preferences: GenerationPreferences) -> List[AttributeDefinition]:
        """
        Select core identity attributes with enhanced logic.

        Args:
            preferences: Generation preferences

        Returns:
            List of selected core attributes
        """
        selected = []

        # Get collections
        age_collection = self.attribute_set.get_collection(AttributeType.AGE_RANGE)
        ethnicity_collection = self.attribute_set.get_collection(AttributeType.ETHNICITY)

        if not age_collection or not ethnicity_collection:
            raise ValueError("Required attribute collections not found")

        # Select age based on preferences
        age_attribute = self._select_age_attribute(age_collection, preferences)
        if age_attribute:
            selected.append(age_attribute)

        # Select ethnicity with diversity consideration
        ethnicity_attribute = self._select_ethnicity_attribute(
            ethnicity_collection, preferences, age_attribute
        )
        if ethnicity_attribute:
            selected.append(ethnicity_attribute)

        return selected

    def _select_age_attribute(self, collection: AttributeCollection,
                              preferences: GenerationPreferences) -> Optional[AttributeDefinition]:
        """
        Select age attribute considering preferences.

        Args:
            collection: Age attribute collection
            preferences: Generation preferences

        Returns:
            Selected age attribute or None
        """
        # If specific age requested, find matching attribute
        if preferences.target_age:
            for attribute in collection.all_attributes():
                if attribute.validate_age(preferences.target_age):
                    return attribute

        # Otherwise, uniform random selection
        return collection.select_random()

    def _select_ethnicity_attribute(self, collection: AttributeCollection,
                                  preferences: GenerationPreferences,
                                  age_attribute: Optional[AttributeDefinition]) -> Optional[AttributeDefinition]:
        """
        Select ethnicity attribute with diversity considerations.

        Args:
            collection: Ethnicity attribute collection
            preferences: Generation preferences
            age_attribute: Selected age attribute (for context)

        Returns:
            Selected ethnicity attribute or None
        """
        # Get available attributes (excluding blocked ones)
        available_attributes = [
            attr for attr in collection.all_attributes()
            if not attr.is_blocked
        ]

        if not available_attributes:
            return None

        # Apply diversity weighting
        weights = []
        for attribute in available_attributes:
            weight = attribute.probability

            # Adjust probability based on diversity target
            if preferences.diversity_target > 0.5:
                # Favor non-white ethnicities
                if attribute.name not in ["caucasian"]:
                    weight *= (preferences.diversity_target / 0.5)

            weights.append(weight)

        # Weighted selection
        total_weight = sum(weights)
        if total_weight <= 0:
            return random.choice(available_attributes)

        normalized_weights = [w / total_weight for w in weights]
        return random.choices(available_attributes, weights=normalized_weights, k=1)[0]

    def _select_physical_attributes(self, selected_attributes: List[AttributeDefinition],
                                   preferences: GenerationPreferences) -> List[AttributeDefinition]:
        """
        Select physical attributes that are compatible with core identity.

        Args:
            selected_attributes: Already selected attributes
            preferences: Generation preferences

        Returns:
            List of selected physical attributes
        """
        selected = []

        # Get core attributes
        age_attr = next((attr for attr in selected_attributes if attr.attribute_type == AttributeType.AGE_RANGE), None)
        ethnicity_attr = next((attr for attr in selected_attributes if attr.attribute_type == AttributeType.ETHNICITY), None)

        # Select skin tone based on ethnicity
        skin_tone_attr = self._select_skin_tone(ethnicity_attr, selected_attributes)
        if skin_tone_attr:
            selected.append(skin_tone_attr)

        # Select hair color and texture
        hair_color_attr = self._select_hair_color(ethnicity_attr, age_attr, selected_attributes)
        if hair_color_attr:
            selected.append(hair_color_attr)

        hair_texture_attr = self._select_hair_texture(ethnicity_attr, selected_attributes)
        if hair_texture_attr:
            selected.append(hair_texture_attr)

        # Select eye color
        eye_color_attr = self._select_eye_color(ethnicity_attr, selected_attributes)
        if eye_color_attr:
            selected.append(eye_color_attr)

        # Select body type and breast size
        body_type_attr = self._select_body_type(age_attr, selected_attributes)
        if body_type_attr:
            selected.append(body_type_attr)

        breast_size_attr = self._select_breast_size(age_attr, body_type_attr, selected_attributes)
        if breast_size_attr:
            selected.append(breast_size_attr)

        return selected

    def _select_skin_tone(self, ethnicity_attr: Optional[AttributeDefinition],
                         selected_attributes: List[AttributeDefinition]) -> Optional[AttributeDefinition]:
        """Select compatible skin tone."""
        skin_collection = self.attribute_set.get_collection(AttributeType.SKIN_TONE)
        if not skin_collection:
            return None

        # Get compatible skin tones
        compatible_skins = skin_collection.get_compatible_attributes(selected_attributes)

        # Filter by ethnicity association if ethnicity is selected
        if ethnicity_attr and compatible_skins:
            ethnic_skins = []
            for skin in compatible_skins:
                if (ethnicity_attr.name in skin.ethnicity_associations or
                    not skin.ethnicity_associations):  # No specific association means universal
                    ethnic_skins.append(skin)

            if ethnic_skins:
                compatible_skins = ethnic_skins

        # Select from compatible options
        if compatible_skins:
            weights = [skin.probability for skin in compatible_skins]
            total_weight = sum(weights)

            if total_weight > 0:
                normalized_weights = [w / total_weight for w in weights]
                return random.choices(compatible_skins, weights=normalized_weights, k=1)[0]
            else:
                return random.choice(compatible_skins)

        return None

    def _select_hair_color(self, ethnicity_attr: Optional[AttributeDefinition],
                          age_attr: Optional[AttributeDefinition],
                          selected_attributes: List[AttributeDefinition]) -> Optional[AttributeDefinition]:
        """Select compatible hair color."""
        hair_collection = self.attribute_set.get_collection(AttributeType.HAIR_COLOR)
        if not hair_collection:
            return None

        # Get compatible options
        compatible_hair = hair_collection.get_compatible_attributes(selected_attributes)

        return hair_collection.select_random() if compatible_hair else None

    def _select_hair_texture(self, ethnicity_attr: Optional[AttributeDefinition],
                           selected_attributes: List[AttributeDefinition]) -> Optional[AttributeDefinition]:
        """Select compatible hair texture."""
        texture_collection = self.attribute_set.get_collection(AttributeType.HAIR_TEXTURE)
        if not texture_collection:
            return None

        return texture_collection.select_random()

    def _select_eye_color(self, ethnicity_attr: Optional[AttributeDefinition],
                         selected_attributes: List[AttributeDefinition]) -> Optional[AttributeDefinition]:
        """Select compatible eye color."""
        eye_collection = self.attribute_set.get_collection(AttributeType.EYE_COLOR)
        if not eye_collection:
            return None

        return eye_collection.select_random()

    def _select_body_type(self, age_attr: Optional[AttributeDefinition],
                         selected_attributes: List[AttributeDefinition]) -> Optional[AttributeDefinition]:
        """Select compatible body type."""
        body_collection = self.attribute_set.get_collection(AttributeType.BODY_TYPE)
        if not body_collection:
            return None

        return body_collection.select_random()

    def _select_breast_size(self, age_attr: Optional[AttributeDefinition],
                           body_type_attr: Optional[AttributeDefinition],
                           selected_attributes: List[AttributeDefinition]) -> Optional[AttributeDefinition]:
        """Select compatible breast size."""
        breast_collection = self.attribute_set.get_collection(AttributeType.BREAST_SIZE)
        if not breast_collection:
            return None

        return breast_collection.select_random()

    def _select_style_attributes(self, selected_attributes: List[AttributeDefinition],
                                preferences: GenerationPreferences) -> List[AttributeDefinition]:
        """
        Select style attributes with enhanced compatibility.

        Args:
            selected_attributes: Already selected attributes
            preferences: Generation preferences

        Returns:
            List of selected style attributes
        """
        selected = []

        # Select clothing style
        clothing_attr = self._select_clothing_style(preferences, selected_attributes)
        if clothing_attr:
            selected.append(clothing_attr)

        # Select aesthetic style
        aesthetic_attr = self._select_aesthetic_style(preferences, selected_attributes)
        if aesthetic_attr:
            selected.append(aesthetic_attr)

        return selected

    def _select_clothing_style(self, preferences: GenerationPreferences,
                              selected_attributes: List[AttributeDefinition]) -> Optional[AttributeDefinition]:
        """Select clothing style."""
        clothing_collection = self.attribute_set.get_collection(AttributeType.CLOTHING_STYLE)
        if not clothing_collection:
            return None

        # Get compatible options
        compatible_clothing = clothing_collection.get_compatible_attributes(selected_attributes)

        # Filter out blocked content
        valid_clothing = [cloth for cloth in compatible_clothing if not cloth.is_blocked]

        if not valid_clothing:
            return None

        # Weighted selection
        weights = [cloth.probability for cloth in valid_clothing]
        total_weight = sum(weights)

        if total_weight > 0:
            normalized_weights = [w / total_weight for w in weights]
            return random.choices(valid_clothing, weights=normalized_weights, k=1)[0]
        else:
            return random.choice(valid_clothing)

    def _select_aesthetic_style(self, preferences: GenerationPreferences,
                               selected_attributes: List[AttributeDefinition]) -> Optional[AttributeDefinition]:
        """Select aesthetic style."""
        aesthetic_collection = self.attribute_set.get_collection(AttributeType.AESTHETIC_STYLE)
        if not aesthetic_collection:
            return None

        # Get compatible options
        compatible_styles = aesthetic_collection.get_compatible_attributes(selected_attributes)

        # Filter out blocked content
        valid_styles = [style for style in compatible_styles if not style.is_blocked]

        if not valid_styles:
            return None

        # Weighted selection
        weights = [style.probability for style in valid_styles]
        total_weight = sum(weights)

        if total_weight > 0:
            normalized_weights = [w / total_weight for w in weights]
            return random.choices(valid_styles, weights=normalized_weights, k=1)[0]
        else:
            return random.choice(valid_styles)

    def _select_detail_attributes(self, selected_attributes: List[AttributeDefinition],
                                preferences: GenerationPreferences) -> List[AttributeDefinition]:
        """
        Select detail attributes (accessories, effects, etc.).

        Args:
            selected_attributes: Already selected attributes
            preferences: Generation preferences

        Returns:
            List of selected detail attributes
        """
        selected = []

        # Select accessories
        accessory_attr = self._select_accessory(selected_attributes)
        if accessory_attr:
            selected.append(accessory_attr)

        # Select special effects if explicit content allowed
        if preferences.explicit_content_allowed:
            effect_attr = self._select_special_effect(selected_attributes)
            if effect_attr:
                selected.append(effect_attr)

        return selected

    def _select_accessory(self, selected_attributes: List[AttributeDefinition]) -> Optional[AttributeDefinition]:
        """Select compatible accessory."""
        accessory_collection = self.attribute_set.get_collection(AttributeType.ACCESSORY)
        if not accessory_collection:
            return None

        return accessory_collection.select_random()

    def _select_special_effect(self, selected_attributes: List[AttributeDefinition]) -> Optional[AttributeDefinition]:
        """Select special effect."""
        effect_collection = self.attribute_set.get_collection(AttributeType.SPECIAL_EFFECT)
        if not effect_collection:
            return None

        return effect_collection.select_random()

    def _resolve_conflicts(self, attributes: List[AttributeDefinition],
                          validation_result: ValidationResult) -> List[AttributeDefinition]:
        """
        Resolve attribute conflicts.

        Args:
            attributes: Current attributes
            validation_result: Validation result

        Returns:
            Resolved attributes
        """
        resolved_attributes = attributes.copy()

        # Handle blocked content issues
        if validation_result.has_blocking_issues:
            # Remove blocked attributes
            resolved_attributes = [
                attr for attr in resolved_attributes
                if not attr.is_blocked
            ]

        # Handle compatibility issues
        if not validation_result.is_valid:
            # This would implement more sophisticated conflict resolution
            # For now, we'll just log the issues
            logger.info(f"Compatibility issues found: {len(validation_result.issues)}")

        return resolved_attributes

    def _create_character_from_attributes(self, attributes: List[AttributeDefinition],
                                         preferences: GenerationPreferences) -> GeneratedCharacter:
        """
        Create a GeneratedCharacter from selected attributes.

        Args:
            attributes: Selected attributes
            preferences: Generation preferences

        Returns:
            Generated character
        """
        # Extract key attributes for character creation
        age_attr = next((attr for attr in attributes if attr.attribute_type == AttributeType.AGE_RANGE), None)
        ethnicity_attr = next((attr for attr in attributes if attr.attribute_type == AttributeType.ETHNICITY), None)
        skin_tone_attr = next((attr for attr in attributes if attr.attribute_type == AttributeType.SKIN_TONE), None)
        eye_color_attr = next((attr for attr in attributes if attr.attribute_type == AttributeType.EYE_COLOR), None)
        hair_color_attr = next((attr for attr in attributes if attr.attribute_type == AttributeType.HAIR_COLOR), None)
        hair_texture_attr = next((attr for attr in attributes if attr.attribute_type == AttributeType.HAIR_TEXTURE), None)
        body_type_attr = next((attr for attr in attributes if attr.attribute_type == AttributeType.BODY_TYPE), None)
        breast_size_attr = next((attr for attr in attributes if attr.attribute_type == AttributeType.BREAST_SIZE), None)
        clothing_style_attr = next((attr for attr in attributes if attr.attribute_type == AttributeType.CLOTHING_STYLE), None)
        aesthetic_style_attr = next((attr for attr in attributes if attr.attribute_type == AttributeType.AESTHETIC_STYLE), None)

        # Create character with extracted attributes
        # (This is a simplified version - a full implementation would be more complex)

        # Determine age from selected attributes
        age = 35  # Default
        if age_attr:
            age = (age_attr.min_age + age_attr.max_age) // 2

        character = GeneratedCharacter(
            age=age,
            age_keywords=age_attr.keywords if age_attr else ["adult"],
            skin_tone=skin_tone_attr.name if skin_tone_attr else "medium",
            skin_keywords=skin_tone_attr.keywords if skin_tone_attr else ["medium skin"],
            skin_prompt_weight=skin_tone_attr.prompt_weight if skin_tone_attr else 1.2,
            ethnicity=ethnicity_attr.name if ethnicity_attr else "caucasian",
            ethnicity_keywords=ethnicity_attr.keywords if ethnicity_attr else ["caucasian"],
            ethnicity_prompt_weight=ethnicity_attr.prompt_weight if ethnicity_attr else 1.0,
            artistic_style="photorealistic",  # Default
            artistic_keywords=["photorealistic", "detailed"],
            eye_color=eye_color_attr.name if eye_color_attr else "brown",
            eye_keywords=eye_color_attr.keywords if eye_color_attr else ["brown eyes"],
            hair_color=hair_color_attr.name if hair_color_attr else "brown",
            hair_keywords=hair_color_attr.keywords if hair_color_attr else ["brown hair"],
            hair_texture=hair_texture_attr.name if hair_texture_attr else "straight",
            hair_texture_keywords=hair_texture_attr.keywords if hair_texture_attr else ["straight hair"],
            hair_texture_weight=hair_texture_attr.prompt_weight if hair_texture_attr else 1.0,
            body_type=body_type_attr.name if body_type_attr else "curvy",
            body_keywords=body_type_attr.keywords if body_type_attr else ["curvy body"],
            breast_size=breast_size_attr.name if breast_size_attr else "large",
            breast_keywords=breast_size_attr.keywords if breast_size_attr else ["large breasts"],
            body_size="average",  # Default
            body_size_keywords=["average build"],
            physical_features="none",  # Default
            physical_feature_keywords=[],
            clothing_style=clothing_style_attr.name if clothing_style_attr else "nude",
            clothing_keywords=clothing_style_attr.keywords if clothing_style_attr else ["nude"],
            clothing_condition="intact",  # Default
            clothing_condition_keywords=["intact clothes"],
            clothing_details="none",  # Default
            clothing_detail_keywords=[],
            aesthetic_style=aesthetic_style_attr.name if aesthetic_style_attr else "none",
            aesthetic_keywords=aesthetic_style_attr.keywords if aesthetic_style_attr else [],
            fantasy_race="none",  # Default
            fantasy_race_keywords=[],
            special_effects="none",  # Default
            special_effect_keywords=[],
            cosplay_style="original_character",  # Default
            cosplay_keywords=[],
            accessories=[],  # Would populate from selected accessories
            accessory_keywords=[],
            erotic_toys=[],  # Default empty
            toy_keywords=[],
            activity="intimate",  # Default
            activity_keywords=["intimate"],
            weather="sunny",  # Default
            weather_keywords=["sunny day"],
            emotional_state="sensual",  # Default
            emotional_keywords=["sensual expression"],
            environment="indoor",  # Default
            environment_keywords=["indoor"],
            setting="bedroom",  # Default
            setting_keywords=["bedroom"],
            lighting_suggestions=["soft lighting"],
            pose="intimate_close",  # Default
            pose_keywords=["intimate pose"],
            pose_complexity="high",
            pose_explicit=True,
            age_features=["mature beauty"] if age >= 40 else ["youthful skin"]
        )

        return character

    def generate_batch(self, count: int,
                      preferences: Optional[GenerationPreferences] = None) -> List[GeneratedCharacter]:
        """
        Generate multiple characters.

        Args:
            count: Number of characters to generate
            preferences: Generation preferences (applied to all)

        Returns:
            List of generated characters
        """
        return [self.generate_character(preferences) for _ in range(count)]
