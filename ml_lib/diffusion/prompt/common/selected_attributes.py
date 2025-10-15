from dataclasses import dataclass
from ml_lib.diffusion.prompt.core import AttributeDefinition


@dataclass
class SelectedAttributes:
    """Container for selected character attributes.

    This class replaces Dict[str, str] that was previously used to store
    selected attributes in character generators.

    Each attribute type is a strongly-typed field instead of a dictionary entry.
    """

    # Core identity
    age_range: AttributeDefinition | None = None
    ethnicity: AttributeDefinition | None = None
    skin_tone: AttributeDefinition | None = None

    # Hair
    hair_color: AttributeDefinition | None = None
    hair_texture: AttributeDefinition | None = None

    # Eyes
    eye_color: AttributeDefinition | None = None

    # Body
    body_type: AttributeDefinition | None = None
    body_size: AttributeDefinition | None = None
    breast_size: AttributeDefinition | None = None
    physical_features: AttributeDefinition | None = None

    # Style
    clothing_style: AttributeDefinition | None = None
    clothing_condition: AttributeDefinition | None = None
    clothing_details: AttributeDefinition | None = None
    aesthetic_style: AttributeDefinition | None = None
    artistic_style: AttributeDefinition | None = None

    # Fantasy
    fantasy_race: AttributeDefinition | None = None
    cosplay_style: AttributeDefinition | None = None

    # Accessories
    accessories: AttributeDefinition | None = None
    erotic_toys: AttributeDefinition | None = None

    # Scene
    activity: AttributeDefinition | None = None
    pose: AttributeDefinition | None = None
    setting: AttributeDefinition | None = None
    environment: AttributeDefinition | None = None
    weather: AttributeDefinition | None = None
    emotional_state: AttributeDefinition | None = None

    # Effects
    special_effects: AttributeDefinition | None = None

    def to_list(self) -> list[AttributeDefinition]:
        """Convert to list of non-None attributes."""
        return [
            attr
            for attr in [
                self.age_range,
                self.ethnicity,
                self.skin_tone,
                self.hair_color,
                self.hair_texture,
                self.eye_color,
                self.body_type,
                self.body_size,
                self.breast_size,
                self.physical_features,
                self.clothing_style,
                self.clothing_condition,
                self.clothing_details,
                self.aesthetic_style,
                self.artistic_style,
                self.fantasy_race,
                self.cosplay_style,
                self.accessories,
                self.erotic_toys,
                self.activity,
                self.pose,
                self.setting,
                self.environment,
                self.weather,
                self.emotional_state,
                self.special_effects,
            ]
            if attr is not None
        ]

    @property
    def count(self) -> int:
        """Number of selected attributes (non-None)."""
        return len(self.to_list())

    @property
    def is_empty(self) -> bool:
        """Whether no attributes have been selected."""
        return self.count == 0
