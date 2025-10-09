#!/usr/bin/env python3
"""Migration example showing transition from old to new system."""

import logging
from pathlib import Path

from ml_lib.diffusion.intelligent.prompting import (
    # Old system
    CharacterGenerator,
    
    # New enhanced system
    EnhancedCharacterGenerator,
    EnhancedConfigLoader,
    GenerationPreferences,
    CharacterAttributeSet,
    AttributeDefinition,
    AttributeType
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demonstrate_old_system():
    """Demonstrate the old character generation system."""
    logger.info("=== OLD SYSTEM DEMONSTRATION ===")
    
    try:
        # Create old generator (still works for backward compatibility)
        old_generator = CharacterGenerator()
        logger.info("✓ Old CharacterGenerator instantiated successfully")
        
        # Generate a character with old system
        old_character = old_generator.generate()
        logger.info("✓ Old character generated successfully")
        logger.info(f"Old character age: {old_character.age}")
        logger.info(f"Old character prompt length: {len(old_character.to_prompt())}")
        
    except Exception as e:
        logger.error(f"Old system error: {e}")


def demonstrate_new_enhanced_system():
    """Demonstrate the new enhanced character generation system."""
    logger.info("=== NEW ENHANCED SYSTEM DEMONSTRATION ===")
    
    try:
        # Create enhanced config loader
        config_loader = EnhancedConfigLoader()
        logger.info("✓ EnhancedConfigLoader instantiated successfully")
        
        # Get attribute set
        attribute_set = config_loader.get_attribute_set()
        logger.info("✓ Attribute set loaded successfully")
        
        # Show some statistics
        total_attributes = sum(
            len(collection.attributes) 
            for collection in attribute_set.collections.values()
        )
        logger.info(f"✓ Total attributes loaded: {total_attributes}")
        
        # Create enhanced generator
        enhanced_generator = EnhancedCharacterGenerator(config_loader)
        logger.info("✓ EnhancedCharacterGenerator instantiated successfully")
        
        # Create generation preferences
        preferences = GenerationPreferences(
            target_age=45,
            target_ethnicity="caucasian",
            target_style="goth",
            explicit_content_allowed=True,
            safety_level="strict",
            diversity_target=0.7
        )
        logger.info("✓ Generation preferences created")
        
        # Generate character with enhanced system
        enhanced_character = enhanced_generator.generate_character(preferences)
        logger.info("✓ Enhanced character generated successfully")
        logger.info(f"Enhanced character age: {enhanced_character.age}")
        logger.info(f"Enhanced character prompt length: {len(enhanced_character.to_prompt())}")
        
        # Generate batch
        batch_characters = enhanced_generator.generate_batch(3, preferences)
        logger.info(f"✓ Batch of {len(batch_characters)} characters generated")
        
        # Show validation
        validation_result = config_loader.validate_character_selection([])
        logger.info(f"✓ Character validation performed: {validation_result['is_valid']}")
        
    except Exception as e:
        logger.error(f"Enhanced system error: {e}")
        import traceback
        traceback.print_exc()


def compare_systems():
    """Compare old and new systems."""
    logger.info("=== SYSTEM COMPARISON ===")
    
    try:
        # Old system
        old_generator = CharacterGenerator()
        old_character = old_generator.generate()
        old_prompt = old_character.to_prompt()
        
        # New system
        config_loader = EnhancedConfigLoader()
        enhanced_generator = EnhancedCharacterGenerator(config_loader)
        preferences = GenerationPreferences()
        enhanced_character = enhanced_generator.generate_character(preferences)
        enhanced_prompt = enhanced_character.to_prompt()
        
        # Comparison
        logger.info("System Comparison:")
        logger.info(f"  Old system prompt length: {len(old_prompt)}")
        logger.info(f"  New system prompt length: {len(enhanced_prompt)}")
        logger.info(f"  Length difference: {len(enhanced_prompt) - len(old_prompt)}")
        
        # Show attribute counts
        old_attr_count = len([
            attr for attr in [
                old_character.age_keywords,
                old_character.skin_keywords,
                old_character.ethnicity_keywords,
                old_character.eye_keywords,
                old_character.hair_keywords,
                old_character.hair_texture_keywords,
                old_character.body_keywords,
                old_character.breast_keywords,
                old_character.clothing_keywords,
                # Add more as needed
            ] if attr
        ])
        
        logger.info(f"  Approximate attributes in old system: {old_attr_count}")
        logger.info("  Attributes in new system: Dynamic (class-based)")
        
    except Exception as e:
        logger.error(f"Comparison error: {e}")


def show_migration_path():
    """Show the migration path from old to new system."""
    logger.info("=== MIGRATION PATH ===")
    
    logger.info("""
    Migration Path from Old to New System:
    
    1. BACKWARD COMPATIBILITY MAINTAINED
       - Old CharacterGenerator still works
       - Existing code continues to function
       - No breaking changes to public API
    
    2. ENHANCEMENTS IN NEW SYSTEM
       - Class-based attributes instead of dictionaries
       - Improved compatibility checking
       - Enhanced safety features
       - Better validation and conflict resolution
       - More sophisticated selection logic
    
    3. PERFORMANCE IMPROVEMENTS
       - More efficient attribute storage
       - Better memory usage with class instances
       - Faster compatibility checking
       - Optimized selection algorithms
    
    4. SAFETY ENHANCEMENTS
       - Automatic blocking of inappropriate content
       - Advanced validation with detailed reporting
       - Context-aware selection
       - Age consistency checking
    
    5. MIGRATION RECOMMENDATIONS
       - Start using EnhancedCharacterGenerator for new features
       - Gradually migrate existing code when convenient
       - Take advantage of enhanced validation in new workflows
       - Benefit from improved safety without code changes
    
    6. DEPRECATION TIMELINE
       - Old system: Supported indefinitely for backward compatibility
       - New system: Active development and enhancements
       - Recommendation: Use new system for all new development
    """)


def main():
    """Main demonstration function."""
    logger.info("Starting migration demonstration...")
    
    # Demonstrate old system (should still work)
    demonstrate_old_system()
    
    # Demonstrate new enhanced system
    demonstrate_new_enhanced_system()
    
    # Compare systems
    compare_systems()
    
    # Show migration path
    show_migration_path()
    
    logger.info("Migration demonstration completed!")


if __name__ == "__main__":
    main()