"""Intelligent character generator using object-oriented architecture."""

import logging
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from ml_lib.diffusion.intelligent.prompting.smart_attributes import (
    SmartAttribute, AttributeCategory, AttributeConfig,
    CharacterAttributeManager, AgeAttribute, EthnicityAttribute,
    SkinToneAttribute, ClothingStyleAttribute, AestheticStyleAttribute
)
from ml_lib.diffusion.intelligent.prompting.attribute_groups import (
    AttributeGroupManager, create_standard_groups, CompatibilityChecker
)
from ml_lib.diffusion.intelligent.prompting.entities import GeneratedCharacter

logger = logging.getLogger(__name__)


@dataclass
class CharacterGenerationContext:
    """Context for character generation."""
    
    target_age: Optional[int] = None
    target_ethnicity: Optional[str] = None
    target_style: Optional[str] = None
    explicit_content_allowed: bool = True
    safety_level: str = "strict"  # strict, moderate, relaxed
    character_focus: str = "portrait"  # portrait, full_body, scene
    quality_target: str = "high"  # low, medium, high, masterpiece


class IntelligentCharacterGenerator:
    """Next-generation character generator with intelligent attribute management."""
    
    def __init__(self):
        """Initialize intelligent character generator."""
        self.attribute_manager = CharacterAttributeManager()
        self.group_manager = create_standard_groups()
        self.compatibility_checker = CompatibilityChecker()
        self._initialize_attributes()
    
    def _initialize_attributes(self):
        """Initialize all smart attributes."""
        # This would load from configuration files
        # For now, we'll create some sample attributes
        pass
    
    def generate_character(self, context: Optional[CharacterGenerationContext] = None) -> GeneratedCharacter:
        """
        Generate an intelligent character with coherent attributes.
        
        Args:
            context: Generation context
            
        Returns:
            Generated character
        """
        if context is None:
            context = CharacterGenerationContext()
        
        # Step 1: Select core identity attributes
        core_attributes = self._select_core_identity(context)
        
        # Step 2: Select compatible physical features
        physical_attributes = self._select_physical_features(core_attributes, context)
        
        # Step 3: Select clothing and style (with compatibility checking)
        style_attributes = self._select_style_attributes(core_attributes, physical_attributes, context)
        
        # Step 4: Select accessories and details
        detail_attributes = self._select_detail_attributes(core_attributes, physical_attributes, style_attributes, context)
        
        # Step 5: Validate all selections
        all_attributes = {**core_attributes, **physical_attributes, **style_attributes, **detail_attributes}
        is_valid, issues = self.compatibility_checker.check_compatibility(all_attributes)
        
        if not is_valid:
            logger.warning(f"Character generation issues: {issues}")
            # Attempt to resolve conflicts or regenerate
            all_attributes = self._resolve_conflicts(all_attributes, issues)
        
        # Step 6: Create final character
        character = self._create_character_from_attributes(all_attributes, context)
        
        return character
    
    def _select_core_identity(self, context: CharacterGenerationContext) -> Dict[str, str]:
        """
        Select core identity attributes (age, ethnicity, etc.).
        
        Args:
            context: Generation context
            
        Returns:
            Dictionary of selected attributes
        """
        # This would use the attribute manager to make intelligent selections
        # For now, we'll return sample values
        return {
            "age_ranges": "milf" if (context.target_age and context.target_age >= 40) else "adult",
            "ethnicities": context.target_ethnicity or "caucasian",
            "skin_tones": "medium" if context.target_ethnicity == "middle_eastern" else "fair"
        }
    
    def _select_physical_features(self, core_attributes: Dict[str, str], 
                                 context: CharacterGenerationContext) -> Dict[str, str]:
        """
        Select physical features compatible with core identity.
        
        Args:
            core_attributes: Previously selected core attributes
            context: Generation context
            
        Returns:
            Dictionary of selected physical attributes
        """
        # Ensure features match ethnicity and age
        ethnicity = core_attributes.get("ethnicities", "caucasian")
        age_group = core_attributes.get("age_ranges", "adult")
        
        # Match skin tone to ethnicity
        skin_tone_map = {
            "caucasian": "fair",
            "east_asian": "light",
            "south_asian": "medium",
            "hispanic_latinx": "medium",
            "african_american": "dark",
            "middle_eastern": "medium_dark"
        }
        
        # Match hair color to ethnicity
        hair_color_map = {
            "caucasian": "blonde",
            "east_asian": "black",
            "south_asian": "black",
            "hispanic_latinx": "brown",
            "african_american": "black",
            "middle_eastern": "brown"
        }
        
        # Match eye color to ethnicity
        eye_color_map = {
            "caucasian": "blue",
            "east_asian": "brown",
            "south_asian": "brown",
            "hispanic_latinx": "brown",
            "african_american": "brown",
            "middle_eastern": "brown"
        }
        
        return {
            "skin_tones": skin_tone_map.get(ethnicity, "medium"),
            "hair_colors": hair_color_map.get(ethnicity, "brown"),
            "eye_colors": eye_color_map.get(ethnicity, "brown"),
            "body_types": "curvy" if age_group == "milf" else "slim",
            "breast_sizes": "large" if age_group == "milf" else "medium"
        }
    
    def _select_style_attributes(self, core_attributes: Dict[str, str], 
                               physical_attributes: Dict[str, str],
                               context: CharacterGenerationContext) -> Dict[str, str]:
        """
        Select style attributes ensuring compatibility.
        
        Args:
            core_attributes: Core identity attributes
            physical_attributes: Physical features
            context: Generation context
            
        Returns:
            Dictionary of selected style attributes
        """
        # Check for blocked combinations
        blocked_styles = ["schoolgirl"]  # Explicitly blocked
        
        # Select clothing style
        if context.target_style and context.target_style not in blocked_styles:
            clothing_style = context.target_style
        else:
            # Default selection avoiding blocked styles
            clothing_options = ["nude", "lingerie", "casual", "formal", "fetish"]
            clothing_style = random.choice([opt for opt in clothing_options if opt not in blocked_styles])
        
        # Select aesthetic style
        aesthetic_options = ["goth", "punk", "nurse", "witch", "nun", "fantasy", "vintage"]
        aesthetic_style = context.target_style or random.choice(aesthetic_options)
        
        # Ensure aesthetic style is not blocked
        if aesthetic_style in blocked_styles:
            aesthetic_style = "goth"  # Safe default
        
        return {
            "clothing_styles": clothing_style,
            "aesthetic_styles": aesthetic_style,
            "artistic_styles": "photorealistic"  # Default to photorealistic
        }
    
    def _select_detail_attributes(self, core_attributes: Dict[str, str],
                                physical_attributes: Dict[str, str],
                                style_attributes: Dict[str, str],
                                context: CharacterGenerationContext) -> Dict[str, str]:
        """
        Select detail attributes (accessories, effects, etc.).
        
        Args:
            core_attributes: Core identity attributes
            physical_attributes: Physical features
            style_attributes: Style attributes
            context: Generation context
            
        Returns:
            Dictionary of selected detail attributes
        """
        # Select accessories based on style
        style = style_attributes.get("aesthetic_styles", "goth")
        accessories_map = {
            "goth": ["jewelry", "headwear", "fetish_accessories"],
            "punk": ["jewelry", "headwear", "bags"],
            "nurse": ["headwear", "bags"],
            "witch": ["headwear", "jewelry"],
            "nun": ["headwear"],
            "fantasy": ["headwear", "jewelry"],
            "vintage": ["jewelry", "bags", "eyewear"]
        }
        
        accessories = accessories_map.get(style, ["jewelry"])
        
        # Select special effects if explicit content is allowed
        special_effects = []
        if context.explicit_content_allowed:
            effects_options = ["wet", "cum", "sticky"]
            if random.random() < 0.3:  # 30% chance of effects
                special_effects = [random.choice(effects_options)]
        
        return {
            "accessories": random.choice(accessories) if accessories else "jewelry",
            "special_effects": special_effects[0] if special_effects else "none"
        }
    
    def _resolve_conflicts(self, attributes: Dict[str, str], 
                          issues: List[str]) -> Dict[str, str]:
        """
        Resolve attribute conflicts.
        
        Args:
            attributes: Current attribute selections
            issues: List of compatibility issues
            
        Returns:
            Resolved attributes
        """
        resolved_attributes = attributes.copy()
        
        # Handle specific known conflicts
        for issue in issues:
            if "schoolgirl" in issue.lower() or "blocked" in issue.lower():
                # Replace blocked attributes
                if resolved_attributes.get("aesthetic_styles") == "schoolgirl":
                    resolved_attributes["aesthetic_styles"] = "goth"
                if resolved_attributes.get("clothing_styles") == "school_uniform":
                    resolved_attributes["clothing_styles"] = "lingerie"
        
        return resolved_attributes
    
    def _create_character_from_attributes(self, attributes: Dict[str, str],
                                         context: CharacterGenerationContext) -> GeneratedCharacter:
        """
        Create a GeneratedCharacter from selected attributes.
        
        Args:
            attributes: Selected attributes
            context: Generation context
            
        Returns:
            Generated character
        """
        # This would create a proper GeneratedCharacter with all the attributes
        # For now, we'll create a simplified version
        
        # Extract age (for demonstration)
        age_keyword = attributes.get("age_ranges", "adult")
        age_map = {
            "young_adult": 22,
            "adult": 35,
            "milf": 45,
            "mature": 60
        }
        age = age_map.get(age_keyword, 35)
        
        # Create character with selected attributes
        character = GeneratedCharacter(
            age=age,
            age_keywords=[age_keyword],
            skin_tone=attributes.get("skin_tones", "medium"),
            skin_keywords=[f"{attributes.get('skin_tones', 'medium')} skin"],
            skin_prompt_weight=1.2,
            ethnicity=attributes.get("ethnicities", "caucasian"),
            ethnicity_keywords=[attributes.get("ethnicities", "caucasian")],
            ethnicity_prompt_weight=1.0,
            eye_color=attributes.get("eye_colors", "brown"),
            eye_keywords=[f"{attributes.get('eye_colors', 'brown')} eyes"],
            hair_color=attributes.get("hair_colors", "brown"),
            hair_keywords=[f"{attributes.get('hair_colors', 'brown')} hair"],
            hair_texture="straight",  # Default
            hair_texture_keywords=["straight hair"],
            hair_texture_weight=1.0,
            body_type=attributes.get("body_types", "curvy"),
            body_keywords=[f"{attributes.get('body_types', 'curvy')} body"],
            breast_size=attributes.get("breast_sizes", "large"),
            breast_keywords=[f"{attributes.get('breast_sizes', 'large')} breasts"],
            clothing_style=attributes.get("clothing_styles", "nude"),
            clothing_keywords=[attributes.get("clothing_styles", "nude")],
            clothing_condition="intact",  # Default
            clothing_condition_keywords=["intact clothes"],
            clothing_details="none",  # Default
            clothing_detail_keywords=["undetailed"],
            cosplay_style="original_character",  # Default
            cosplay_keywords=["original character"],
            accessories=[attributes.get("accessories", "jewelry")],
            accessory_keywords=[attributes.get("accessories", "jewelry")],
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
                      context: Optional[CharacterGenerationContext] = None) -> List[GeneratedCharacter]:
        """
        Generate multiple characters.
        
        Args:
            count: Number of characters to generate
            context: Generation context (applied to all)
            
        Returns:
            List of generated characters
        """
        return [self.generate_character(context) for _ in range(count)]