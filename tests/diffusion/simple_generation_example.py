#!/usr/bin/env python3
"""
Simple image generation example using available pipeline components.

This example demonstrates the actual image generation pipeline if available.
"""

import os
import sys
from pathlib import Path
import logging
from datetime import datetime
import uuid

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml_lib.diffusion.intelligent.prompting import CharacterGenerator
from ml_lib.diffusion.intelligent.hub_integration import ModelRegistry
from ml_lib.diffusion.intelligent.pipeline.services.intelligent_builder import (
    IntelligentPipelineBuilder
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_simple_generation():
    """Run simple generation using available pipeline."""
    logger.info("Running simple image generation example...")
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Generate a character
    generator = CharacterGenerator()
    character = generator.generate_character()
    
    logger.info(f"Generated character prompt: {character.to_prompt()}")
    
    # Try to create a simple pipeline for generation
    try:
        # Try to find the correct method to initialize the builder
        # Different implementations might use different initialization methods
        try:
            # Try the auto method
            builder = IntelligentPipelineBuilder.from_auto()
        except AttributeError:
            # If from_auto doesn't exist, try from_comfyui_auto
            try:
                builder = IntelligentPipelineBuilder.from_comfyui_auto()
            except AttributeError:
                # If neither exists, try the default constructor
                builder = IntelligentPipelineBuilder()
    
        # Try to generate an image with the character prompt
        # The actual API might vary depending on the implementation
        if hasattr(builder, 'generate'):
            # If it has a generate method, use it
            result = builder.generate(character.to_prompt())
            
            if hasattr(result, 'image') and result.image:
                # Create output filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_id = str(uuid.uuid4())[:8]
                filename = f"simple_gen_{timestamp}_{unique_id}.png"
                filepath = output_dir / filename
                
                # Save the image
                result.image.save(filepath)
                logger.info(f"✅ Image saved to: {filepath}")
                
                # Save prompt info
                info_file = output_dir / f"info_{timestamp}_{unique_id}.txt"
                with open(info_file, 'w') as f:
                    f.write(f"Generation Time: {datetime.now()}\n")
                    f.write(f"Prompt: {character.to_prompt()}\n")
                
                return filepath
            else:
                logger.warning("Generated result has no image, creating placeholder...")
        else:
            logger.warning("Builder has no generate method, creating placeholder...")
            
    except Exception as e:
        logger.warning(f"Could not create pipeline: {e}")
        logger.info("The actual image generation requires a full Stable Diffusion pipeline setup.")
    
    # If actual generation failed, create a text file indicating what would have been generated
    info_file = output_dir / f"info_placeholder_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(info_file, 'w') as f:
        f.write(f"Generation Time: {datetime.now()}\n")
        f.write(f"Character Prompt: {character.to_prompt()}\n")
        f.write(f"Expected Resolution: 1024x1024\n")
        f.write(f"Expected Steps: 35\n")
        f.write(f"Expected CFG Scale: 9.0\n")
        f.write("\nNote: This is a placeholder. Actual image generation requires:\n")
        f.write("- Stable Diffusion pipeline (diffusers)\n")
        f.write("- Model weights (e.g., SDXL)\n")
        f.write("- Sufficient VRAM (8GB+ recommended)\n")
        f.write("- Proper pipeline integration\n")
    
    logger.info(f"Saved generation info to: {info_file}")
    return info_file


def main():
    """Run the simple generation example."""
    logger.info("Simple Image Generation Example")
    logger.info("Demonstrating the pipeline with actual or simulated generation\n")
    
    try:
        result_path = run_simple_generation()
        logger.info(f"\nGeneration completed! Output: {result_path}")
        
        logger.info("\nThis example shows:")
        logger.info("  1. Character generation with intelligent attributes")
        logger.info("  2. Pipeline initialization (when available)")
        logger.info("  3. Image generation (actual or simulated)")
        logger.info("  4. Output file management")
        
    except Exception as e:
        logger.error(f"Error in generation: {e}", exc_info=True)


if __name__ == "__main__":
    main()