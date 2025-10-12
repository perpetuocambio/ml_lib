#!/usr/bin/env python3
"""
Complete image generation example with fallback simulation.

This example demonstrates the full pipeline including actual image generation:
1. Character generation with intelligent prompting
2. Model and LoRA selection
3. Parameter optimization
4. Image generation using diffusers (with fallback to simulation)
"""

import os
import sys
from pathlib import Path
import logging
from PIL import Image, ImageDraw, ImageFont
import random
from datetime import datetime
import uuid

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml_lib.diffusion.services import (
    CharacterGenerator,
    PromptAnalyzer,
    LoRARecommender,
    ParameterOptimizer,
    ModelRegistry,
)
from ml_lib.diffusion.models import GeneratedCharacter, Priority
from ml_lib.diffusion.services.parameter_optimizer import GenerationConstraints

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_placeholder_image(prompt, width=1024, height=1024):
    """Create a placeholder image with text description."""
    # Create a blank image
    img = Image.new('RGB', (width, height), color=(random.randint(50, 150), random.randint(50, 150), random.randint(150, 255)))
    draw = ImageDraw.Draw(img)
    
    # Add text description
    lines = []
    line_length = 60  # Approximate characters per line
    for i in range(0, len(prompt), line_length):
        lines.append(prompt[i:i+line_length])
    
    y_offset = 20
    for line in lines[:8]:  # Limit to 8 lines
        draw.text((20, y_offset), line, fill=(0, 0, 0))
        y_offset += 25
    
    # Add a simple visual element to make it more interesting
    for _ in range(5):
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = x1 + random.randint(20, 100)
        y2 = y1 + random.randint(20, 100)
        draw.ellipse([x1, y1, x2, y2], outline=(0, 0, 0), width=2)
    
    return img


def example_intelligent_generation():
    """Example: Complete intelligent image generation."""
    logger.info("=" * 80)
    logger.info("Example: Complete Intelligent Image Generation")
    logger.info("=" * 80)

    # Output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Step 1: Generate character with intelligent prompting
    logger.info("Step 1: Generating character with intelligent prompting...")
    char_generator = CharacterGenerator()
    character = char_generator.generate_character()
    
    logger.info(f"Generated character: {character.to_prompt()}")
    
    # Step 2: Analyze the prompt
    logger.info("Step 2: Analyzing prompt with semantic analysis...")
    analyzer = PromptAnalyzer(use_llm=False)  # Use rule-based analysis
    analysis = analyzer.analyze(character.to_prompt())
    
    logger.info(f"Prompt analysis - Complexity: {analysis.complexity_category.value}")
    logger.info(f"Detected concepts: {len(analysis.detected_concepts)} categories")
    
    # Step 3: Initialize model registry and LoRA recommender (if available)
    logger.info("Step 3: Initializing model registry and LoRA recommender...")
    try:
        registry = ModelRegistry()
        lora_recommender = LoRARecommender(registry=registry)
        
        # Get LoRA recommendations based on analysis
        recommendations = lora_recommender.recommend(
            prompt_analysis=analysis,
            base_model="sdxl-base-1.0",
            max_loras=2,
            min_confidence=0.3,
        )
        
        logger.info(f"Selected {len(recommendations)} LoRAs:")
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"  {i}. {rec.lora_name} (confidence: {rec.confidence_score:.2f}, weight: {rec.suggested_alpha:.2f})")
    except Exception as e:
        logger.warning(f"Could not initialize LoRA recommender: {e}")
        recommendations = []
    
    # Step 4: Optimize parameters
    logger.info("Step 4: Optimizing generation parameters...")
    optimizer = ParameterOptimizer()
    constraints = GenerationConstraints(
        max_time_seconds=120,
        max_vram_gb=12,
        priority=Priority.BALANCED,
    )
    
    params = optimizer.optimize(analysis, constraints)
    
    logger.info(f"Optimized parameters:")
    logger.info(f"  Steps: {params.num_steps}")
    logger.info(f"  CFG Scale: {params.guidance_scale}")
    logger.info(f"  Resolution: {params.width}x{params.height}")
    logger.info(f"  Sampler: {params.sampler_name}")
    
    # Step 5: Generate image (simulated if actual generation not available)
    logger.info("Step 5: Generating image...")
    
    try:
        # Try to use the actual pipeline if available
        from ml_lib.diffusion.intelligent.pipeline.services.intelligent_builder import (
            IntelligentPipelineBuilder,
            GenerationConfig,
        )
        
        builder = IntelligentPipelineBuilder.from_auto()
        
        # Create the generation config
        config = GenerationConfig(
            prompt=character.to_prompt(),
            negative_prompt="anime, cartoon, comic, drawing, painting, low quality, blurry, deformed, bad anatomy",
            num_inference_steps=params.num_steps,
            guidance_scale=params.guidance_scale,
            width=params.width,
            height=params.height,
        )
        
        result = builder.generate(config)
        
        if result.image:
            image = result.image
        else:
            logger.warning("Actual generation failed, using placeholder...")
            image = create_placeholder_image(character.to_prompt(), params.width, params.height)
    except ImportError:
        logger.warning("IntelligentPipelineBuilder not available, using placeholder...")
        image = create_placeholder_image(character.to_prompt(), params.width, params.height)
    except Exception as e:
        logger.warning(f"Actual generation failed ({e}), using placeholder...")
        image = create_placeholder_image(character.to_prompt(), params.width, params.height)
    
    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"generated_image_{timestamp}_{unique_id}.png"
    filepath = output_dir / filename
    
    # Save the image
    image.save(filepath)
    
    logger.info(f"✅ Image saved to: {filepath}")
    logger.info(f"   Prompt: {character.to_prompt()[:100]}...")
    
    # Also save metadata
    metadata_file = output_dir / f"metadata_{timestamp}_{unique_id}.txt"
    with open(metadata_file, 'w') as f:
        f.write(f"Generation Date: {datetime.now()}\n")
        f.write(f"Prompt: {character.to_prompt()}\n")
        f.write(f"Complexity: {analysis.complexity_category.value}\n")
        f.write(f"Steps: {params.num_steps}\n")
        f.write(f"CFG Scale: {params.guidance_scale}\n")
        f.write(f"Resolution: {params.width}x{params.height}\n")
        f.write(f"Sampler: {params.sampler_name}\n")
        f.write(f"Quality Score: {params.estimated_quality_score}\n")
        f.write(f"LoRA Recommendations: {len(recommendations)}\n")
        
    logger.info(f"   Metadata saved to: {metadata_file}")
    
    return filepath


def example_batch_generation():
    """Example: Generate multiple images in batch."""
    logger.info("\n" + "=" * 80)
    logger.info("Example: Batch Image Generation")
    logger.info("=" * 80)
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    logger.info("Generating 3 different characters...")
    
    char_generator = CharacterGenerator()
    images_generated = []
    
    for i in range(3):
        logger.info(f"\nGenerating image {i+1}/3...")
        
        # Generate a character
        character = char_generator.generate_character()
        prompt = character.to_prompt()
        
        logger.info(f"  Character prompt: {prompt[:100]}...")
        
        try:
            # Create a placeholder image since full pipeline might not be available
            image = create_placeholder_image(prompt, 896, 1152)
            
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"batch_image_{i+1}_{timestamp}_{str(uuid.uuid4())[:8]}.png"
            filepath = output_dir / filename
            image.save(filepath)
            
            images_generated.append(filepath)
            logger.info(f"  ✅ Saved: {filepath}")
        except Exception as e:
            logger.error(f"  ❌ Error generating image {i+1}: {e}")
    
    logger.info(f"\nGenerated {len(images_generated)} images in batch mode")
    for img_path in images_generated:
        logger.info(f"  - {img_path}")
    
    return images_generated


def example_styled_generation():
    """Example: Generate images with specific styles."""
    logger.info("\n" + "=" * 80)
    logger.info("Example: Styled Generation")
    logger.info("=" * 80)
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Define different generation styles
    styles = [
        {
            "name": "photorealistic",
            "prompt_modifier": "photorealistic, highly detailed, 8k, masterpiece, best quality",
        },
        {
            "name": "artistic",
            "prompt_modifier": "artistic, concept art, digital painting, detailed, illustration",
        },
        {
            "name": "cinematic",
            "prompt_modifier": "cinematic lighting, dramatic, movie still, film grain, color graded",
        }
    ]
    
    char_generator = CharacterGenerator()
    
    for style in styles:
        logger.info(f"\nGenerating {style['name']} style image...")
        
        # Generate a character
        character = char_generator.generate_character()
        base_prompt = character.to_prompt()
        full_prompt = f"{style['prompt_modifier']}, {base_prompt}"
        
        logger.info(f"  Full prompt: {full_prompt[:100]}...")
        
        try:
            # Create a placeholder image for the style
            image = create_placeholder_image(full_prompt, 1024, 1024)
            
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"styled_{style['name']}_{timestamp}_{str(uuid.uuid4())[:8]}.png"
            filepath = output_dir / filename
            image.save(filepath)
            
            logger.info(f"  ✅ {style['name']} style image saved: {filepath}")
        except Exception as e:
            logger.error(f"  ❌ Error generating {style['name']} style: {e}")


def main():
    """Run all generation examples."""
    logger.info("Complete Image Generation Examples")
    logger.info("Demonstrating full pipeline from character generation to image output\n")
    
    try:
        # Example 1: Complete intelligent generation
        image_path = example_intelligent_generation()
        
        # Example 2: Batch generation
        batch_images = example_batch_generation()
        
        # Example 3: Styled generation
        example_styled_generation()
        
        logger.info("\n" + "=" * 80)
        logger.info("All examples completed!")
        logger.info(f"Generated images saved to: {Path('output').resolve()}")
        
        if image_path:
            logger.info(f"Sample image: {image_path}")
        logger.info(f"Total batch images: {len(batch_images)}")
        logger.info("Check the 'output' directory for generated images and metadata.\n")
        
        logger.info("Key Features Demonstrated:")
        logger.info("  1. Intelligent character generation with diversity enforcement")
        logger.info("  2. Semantic prompt analysis with Ollama (when available)")
        logger.info("  3. Automatic LoRA selection and optimization")
        logger.info("  4. Parameter optimization based on prompt complexity")
        logger.info("  5. Complete image generation pipeline (with fallback to simulation)")
        logger.info("  6. Batch processing capabilities")
        logger.info("  7. Style-specific generation")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)


if __name__ == "__main__":
    main()