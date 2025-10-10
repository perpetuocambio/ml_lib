#!/usr/bin/env python3
"""Script to update all enums to inherit from BasePromptEnum."""

import re
from pathlib import Path

# Root directory
ENUMS_DIR = Path("/src/perpetuocambio/ml_lib/ml_lib/diffusion/intelligent/prompting/enums")

# Files to update
files_to_update = [
    "physical/body_size.py",
    "physical/eye_color.py",
    "physical/hair_texture.py",
    "physical/hair_color.py",
    "physical/physical_feature.py",
    "physical/breast_size.py",
    "physical/body_type.py",
    "physical/age_range.py",
    "style/special_effect.py",
    "style/artistic_style.py",
    "style/fantasy_race.py",
    "style/aesthetic_style.py",
    "emotional/erotic_toy.py",
    "emotional/emotional_state.py",
    "appearance/clothing_condition.py",
    "appearance/accessory.py",
    "appearance/clothing_detail.py",
    "appearance/cosplay_style.py",
    "appearance/clothing_style.py",
    "scene/weather_condition.py",
    "scene/activity.py",
    "scene/pose.py",
    "scene/environment.py",
    "scene/setting.py",
    "meta/character_focus.py",
    "meta/complexity_level.py",
    "meta/quality_target.py",
    "meta/safety_level.py",
]


def update_enum_file(file_path: Path) -> None:
    """Update a single enum file to use BasePromptEnum."""
    content = file_path.read_text()

    # Skip if already uses BasePromptEnum
    if "BasePromptEnum" in content:
        print(f"✓ {file_path.name} already uses BasePromptEnum")
        return

    # Replace 'from enum import Enum' with BasePromptEnum import
    content = re.sub(
        r'from enum import Enum',
        'from ..base_prompt_enum import BasePromptEnum',
        content
    )

    # For meta enums, use different relative path
    if "/meta/" in str(file_path):
        content = re.sub(
            r'from \.\.base_prompt_enum import BasePromptEnum',
            'from ..base_prompt_enum import BasePromptEnum',
            content
        )

    # Replace class definition
    content = re.sub(
        r'class (\w+)\(Enum\):',
        r'class \1(BasePromptEnum):',
        content
    )

    # Write back
    file_path.write_text(content)
    print(f"✓ Updated {file_path.name}")


def main():
    """Update all enum files."""
    print("Updating enums to use BasePromptEnum...\n")

    updated_count = 0
    for file_rel_path in files_to_update:
        file_path = ENUMS_DIR / file_rel_path
        if file_path.exists():
            update_enum_file(file_path)
            updated_count += 1
        else:
            print(f"✗ File not found: {file_path}")

    print(f"\n✅ Updated {updated_count} enum files")


if __name__ == "__main__":
    main()
