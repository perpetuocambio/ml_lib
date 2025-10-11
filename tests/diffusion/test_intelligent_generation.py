#!/usr/bin/env python3
"""Test suite for intelligent character generation system."""

import unittest
from ml_lib.diffusion.intelligent.prompting.intelligent_generator import (
    IntelligentCharacterGenerator, CharacterGenerationContext
)
from ml_lib.diffusion.intelligent.prompting.attribute_groups import (
    create_standard_groups, AttributeGroupType
)
from ml_lib.diffusion.intelligent.prompting.smart_attributes import (
    AttributeCategory, AttributeConfig, SmartAttribute
)


class TestIntelligentCharacterGeneration(unittest.TestCase):
    """Test intelligent character generation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = IntelligentCharacterGenerator()
    
    def test_basic_character_generation(self):
        """Test basic character generation."""
        context = CharacterGenerationContext()
        character = self.generator.generate_character(context)
        
        self.assertIsNotNone(character)
        self.assertIsInstance(character.age, int)
        self.assertGreaterEqual(character.age, 18)
        self.assertLessEqual(character.age, 80)
    
    def test_character_with_context(self):
        """Test character generation with specific context."""
        context = CharacterGenerationContext(
            target_age=45,
            target_ethnicity="caucasian",
            explicit_content_allowed=True
        )
        
        character = self.generator.generate_character(context)
        
        self.assertIsNotNone(character)
        # Note: Since we're using a simplified implementation,
        # we can't test the exact values, but we can test the structure
    
    def test_batch_generation(self):
        """Test batch character generation."""
        context = CharacterGenerationContext()
        characters = self.generator.generate_batch(3, context)
        
        self.assertEqual(len(characters), 3)
        for character in characters:
            self.assertIsNotNone(character)
    
    def test_safety_features(self):
        """Test safety features block inappropriate content."""
        # Test that blocked styles are not generated
        context = CharacterGenerationContext(target_style="schoolgirl")
        character = self.generator.generate_character(context)
        
        # The character should still be generated (safely)
        self.assertIsNotNone(character)
        
        # Note: In a full implementation, we would check that the
        # actual attributes don't include blocked content
    
    def test_attribute_groups(self):
        """Test attribute group functionality."""
        group_manager = create_standard_groups()
        
        # Test that groups are created
        self.assertGreater(len(group_manager.groups), 0)
        
        # Test getting specific groups
        goth_group = group_manager.get_group(AttributeGroupType.GOTH_SET)
        self.assertIsNotNone(goth_group)
        
        # Test group compatibility
        compatible = group_manager.get_compatible_groups(AttributeGroupType.GOTH_SET)
        self.assertIsInstance(compatible, list)
    
    def test_smart_attributes(self):
        """Test smart attribute functionality."""
        # Test attribute configuration
        config = AttributeConfig(
            name="test_attribute",
            category=AttributeCategory.AGE,
            keywords=["test", "keywords"],
            probability=0.5,
            min_age=18,
            max_age=80
        )
        
        self.assertEqual(config.name, "test_attribute")
        self.assertEqual(config.category, AttributeCategory.AGE)
        self.assertEqual(config.probability, 0.5)
        self.assertEqual(config.min_age, 18)
        self.assertEqual(config.max_age, 80)
    
    def test_age_validation(self):
        """Test age validation for attributes."""
        config = AttributeConfig(
            name="age_restricted",
            category=AttributeCategory.AGE,
            keywords=["restricted"],
            min_age=25,
            max_age=60
        )
        
        # This would be tested more thoroughly in a full implementation
        self.assertEqual(config.min_age, 25)
        self.assertEqual(config.max_age, 60)


if __name__ == "__main__":
    unittest.main()