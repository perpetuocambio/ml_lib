"""
Diffusion-based image generation module.

This module provides intelligent image generation capabilities with:
- Automatic prompt analysis and optimization
- LoRA recommendation and management
- Memory-efficient generation
- Character generation with ethnic consistency
- Learning from user feedback

Quick Start:
    >>> from ml_lib.diffusion import ImageGenerator
    >>> generator = ImageGenerator()
    >>> image = generator.generate_character()
    >>> image.save("character.png")

Advanced Usage:
    >>> from ml_lib.diffusion import ImageGenerator, GenerationOptions
    >>> options = GenerationOptions(steps=50, cfg_scale=8.0, enable_learning=True)
    >>> generator = ImageGenerator(
    ...     model="stabilityai/stable-diffusion-xl-base-1.0",
    ...     options=options
    ... )
    >>> image = generator.generate_from_prompt("a beautiful landscape")
    >>> image.save("landscape.png")
"""

# Public facade API (recommended for all users)
from ml_lib.diffusion.facade import ImageGenerator, Generator, GenerationOptions

# Advanced API (for users who need fine-grained control)
# Note: These are subject to change and may have breaking changes
# Most users should use the facade instead
__advanced_api__ = [
    "services.intelligent_pipeline",
    "handlers.character_generator",
    "handlers.memory_manager",
]

__all__ = [
    # Facade (recommended)
    "ImageGenerator",
    "Generator",
    "GenerationOptions",

    # Advanced (use with caution)
    "__advanced_api__",
]

__version__ = "0.2.0"
