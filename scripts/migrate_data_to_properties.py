#!/usr/bin/env python3
"""Script to migrate data from *_data.py files to enum properties."""

from pathlib import Path

# Datos de HairColor
HAIR_COLOR_PROPERTIES = '''
    @property
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation."""
        _keywords: dict[HairColor, tuple[str, ...]] = {
            HairColor.BLACK: ("black hair", "raven hair", "ebony hair"),
            HairColor.DARK_BROWN: ("dark brown hair", "black-brown hair", "rich brown hair"),
            HairColor.BROWN: ("brown hair", "light brown hair", "chestnut hair", "auburn"),
            HairColor.BLONDE: ("blonde hair", "blond hair", "golden hair", "honey hair"),
            HairColor.RED: ("red hair", "ginger hair", "auburn hair", "strawberry blonde"),
            HairColor.GREY_SILVER: ("grey hair", "gray hair", "silver hair", "salt and pepper"),
            HairColor.WHITE: ("white hair", "pure white hair", "snow white hair"),
        }
        return _keywords[self]

    @property
    def min_age(self) -> int:
        """Minimum age for this attribute."""
        _min_ages: dict[HairColor, int] = {
            HairColor.BLACK: 18,
            HairColor.DARK_BROWN: 18,
            HairColor.BROWN: 18,
            HairColor.BLONDE: 18,
            HairColor.RED: 18,
            HairColor.GREY_SILVER: 45,
            HairColor.WHITE: 60,
        }
        return _min_ages[self]

    @property
    def max_age(self) -> int:
        """Maximum age for this attribute."""
        return 80
'''

# Datos de HairTexture
HAIR_TEXTURE_PROPERTIES = '''
    @property
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation."""
        _keywords: dict[HairTexture, tuple[str, ...]] = {
            HairTexture.STRAIGHT: ("straight hair", "smooth hair", "sleek hair"),
            HairTexture.WAVY: ("wavy hair", "flowing hair", "loose waves"),
            HairTexture.CURLY: ("curly hair", "curled hair", "ringlets"),
            HairTexture.COILY: ("coily hair", "kinky hair", "tight coils"),
            HairTexture.TEXTURED: ("textured hair", "natural hair", "afro"),
        }
        return _keywords[self]

    @property
    def prompt_weight(self) -> float:
        """Weight/emphasis for this attribute in prompts."""
        return 1.0
'''


def add_hair_color_properties():
    """Add properties to HairColor enum."""
    file_path = Path("/src/perpetuocambio/ml_lib/ml_lib/diffusion/intelligent/prompting/enums/physical/hair_color.py")
    content = file_path.read_text()

    if "@property" in content:
        print("✓ HairColor already has properties")
        return

    # Add docstring note
    content = content.replace(
        '    All valid hair colors have equal probability (uniform distribution).\n    """',
        '    All valid hair colors have equal probability (uniform distribution).\n\n    Each enum value provides metadata through properties.\n    """'
    )

    # Add properties before the last line
    content = content.rstrip() + HAIR_COLOR_PROPERTIES + "\n"

    file_path.write_text(content)
    print("✓ Added properties to HairColor")


def add_hair_texture_properties():
    """Add properties to HairTexture enum."""
    file_path = Path("/src/perpetuocambio/ml_lib/ml_lib/diffusion/intelligent/prompting/enums/physical/hair_texture.py")
    content = file_path.read_text()

    if "@property" in content:
        print("✓ HairTexture already has properties")
        return

    # Add docstring note
    content = content.replace(
        '    All valid hair textures have equal probability (uniform distribution).\n    """',
        '    All valid hair textures have equal probability (uniform distribution).\n\n    Each enum value provides metadata through properties.\n    """'
    )

    # Add properties before the last line
    content = content.rstrip() + HAIR_TEXTURE_PROPERTIES + "\n"

    file_path.write_text(content)
    print("✓ Added properties to HairTexture")


def main():
    """Migrate data to properties."""
    print("Migrating data classes to enum properties...\n")
    add_hair_color_properties()
    add_hair_texture_properties()
    print("\n✅ Migration complete")


if __name__ == "__main__":
    main()
