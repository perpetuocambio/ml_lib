#!/usr/bin/env python3
"""Auto-generate properties from YAML data for all enums."""

import yaml
from pathlib import Path
from typing import Any

YAML_PATH = Path("/src/perpetuocambio/ml_lib/config/intelligent_prompting/character_attributes.yaml")
ENUMS_DIR = Path("/src/perpetuocambio/ml_lib/ml_lib/diffusion/intelligent/prompting/enums")

# Map YAML keys to enum file paths and class names
ENUM_MAP = {
    "body_types": ("physical/body_type.py", "BodyType"),
    "body_sizes": ("physical/body_size.py", "BodySize"),
    "breast_sizes": ("physical/breast_size.py", "BreastSize"),
    "age_ranges": ("physical/age_range.py", "AgeRange"),
    "physical_features": ("physical/physical_feature.py", "PhysicalFeature"),
    "settings": ("scene/setting.py", "Setting"),
    "poses": ("scene/pose.py", "Pose"),
    "activities": ("scene/activity.py", "Activity"),
    "weather_conditions": ("scene/weather_condition.py", "WeatherCondition"),
    "environment_details": ("scene/environment.py", "Environment"),
    "artistic_styles": ("style/artistic_style.py", "ArtisticStyle"),
    "aesthetic_styles": ("style/aesthetic_style.py", "AestheticStyle"),
    "special_effects": ("style/special_effect.py", "SpecialEffect"),
    "fantasy_races": ("style/fantasy_race.py", "FantasyRace"),
    "clothing_styles": ("appearance/clothing_style.py", "ClothingStyle"),
    "clothing_conditions": ("appearance/clothing_condition.py", "ClothingCondition"),
    "clothing_details": ("appearance/clothing_detail.py", "ClothingDetail"),
    "accessories": ("appearance/accessory.py", "Accessory"),
    "cosplay_styles": ("appearance/cosplay_style.py", "CosplayStyle"),
    "emotional_states": ("emotional/emotional_state.py", "EmotionalState"),
    "erotic_toys": ("emotional/erotic_toy.py", "EroticToy"),
}


def load_yaml() -> dict[str, Any]:
    """Load YAML configuration."""
    with open(YAML_PATH) as f:
        return yaml.safe_load(f)


def generate_keywords_property(enum_name: str, data: dict[str, Any]) -> str:
    """Generate keywords property code."""
    lines = [
        "    @property",
        "    def keywords(self) -> tuple[str, ...]:",
        '        """Keywords used in prompt generation."""',
        f"        _keywords: dict[{enum_name}, tuple[str, ...]] = {{",
    ]

    for key, value in data.items():
        enum_key = key.upper()
        keywords = value.get("keywords", [key.replace("_", " ")])
        keywords_str = ", ".join(f'"{k}"' for k in keywords)
        lines.append(f"            {enum_name}.{enum_key}: ({keywords_str}),")

    lines.append("        }")
    lines.append("        return _keywords[self]")

    return "\n".join(lines)


def generate_min_age_property(enum_name: str, data: dict[str, Any]) -> str:
    """Generate min_age property if data has varying ages."""
    ages = {value.get("min_age", 18) for value in data.values()}

    if len(ages) == 1:
        # All same age, simple property
        age = ages.pop()
        return f"""    @property
    def min_age(self) -> int:
        \"\"\"Minimum age for this attribute.\"\"\"
        return {age}"""

    # Varying ages, use dict
    lines = [
        "    @property",
        "    def min_age(self) -> int:",
        '        """Minimum age for this attribute."""',
        f"        _min_ages: dict[{enum_name}, int] = {{",
    ]

    for key, value in data.items():
        enum_key = key.upper()
        min_age = value.get("min_age", 18)
        lines.append(f"            {enum_name}.{enum_key}: {min_age},")

    lines.append("        }")
    lines.append("        return _min_ages[self]")

    return "\n".join(lines)


def generate_properties(yaml_key: str, enum_name: str, data: dict[str, Any]) -> str:
    """Generate all properties for an enum."""
    properties = []

    # Keywords property (always)
    properties.append(generate_keywords_property(enum_name, data))
    properties.append("")

    # Age properties if present
    if any("min_age" in v for v in data.values()):
        properties.append(generate_min_age_property(enum_name, data))
        properties.append("")
        properties.append("""    @property
    def max_age(self) -> int:
        \"\"\"Maximum age for this attribute.\"\"\"
        return 80""")
        properties.append("")

    # Prompt weight if present
    if any("prompt_weight" in v for v in data.values()):
        weights = {v.get("prompt_weight", 1.0) for v in data.values()}
        if len(weights) == 1:
            weight = weights.pop()
            properties.append(f"""    @property
    def prompt_weight(self) -> float:
        \"\"\"Weight/emphasis for this attribute in prompts.\"\"\"
        return {weight}""")
            properties.append("")

    return "\n".join(properties)


def add_properties_to_enum(file_path: Path, enum_name: str, properties_code: str):
    """Add properties to an enum file."""
    content = file_path.read_text()

    if "@property" in content:
        print(f"  ✓ {file_path.name} already has properties")
        return

    # Add metadata note to docstring
    content = content.replace(
        '    All valid',
        '    Each enum value provides metadata through properties.\n    All valid'
    )

    # Add properties at the end
    content = content.rstrip() + "\n\n" + properties_code

    file_path.write_text(content)
    print(f"  ✓ Added properties to {file_path.name}")


def main():
    """Generate properties for all enums."""
    print("Auto-generating properties from YAML...\n")

    yaml_data = load_yaml()
    count = 0

    for yaml_key, (file_rel, enum_name) in ENUM_MAP.items():
        if yaml_key not in yaml_data:
            print(f"  ✗ {yaml_key} not found in YAML")
            continue

        file_path = ENUMS_DIR / file_rel
        if not file_path.exists():
            print(f"  ✗ {file_path} does not exist")
            continue

        print(f"Processing {enum_name}...")
        properties = generate_properties(yaml_key, enum_name, yaml_data[yaml_key])
        add_properties_to_enum(file_path, enum_name, properties)
        count += 1

    print(f"\n✅ Generated properties for {count} enums")


if __name__ == "__main__":
    main()
