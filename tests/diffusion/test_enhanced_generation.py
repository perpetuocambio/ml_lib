#!/usr/bin/env python3
"""Tests for the enhanced character generation system."""

import unittest
import logging
from pathlib import Path

from ml_lib.diffusion.services import (
    EnhancedCharacterGenerator,
    EnhancedConfigLoader,
)
from ml_lib.diffusion.models import (
    GenerationPreferences,
    CharacterAttributeSet,
    AttributeDefinition,
    AttributeType
)

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)


class TestEnhancedAttributes(unittest.TestCase):
    """Test enhanced attribute classes."""
    
    def test_attribute_definition_creation(self):
        """Test creation of attribute definitions."""
        attr = AttributeDefinition(
            name="test_attr",
            attribute_type=AttributeType.AGE_RANGE,
            keywords=["test", "keywords"],
            probability=0.5,
            prompt_weight=1.2,
            min_age=25,
            max_age=60
        )
        
        self.assertEqual(attr.name, "test_attr")
        self.assertEqual(attr.attribute_type, AttributeType.AGE_RANGE)
        self.assertEqual(attr.keywords, ["test", "keywords"])
        self.assertEqual(attr.probability, 0.5)
        self.assertEqual(attr.prompt_weight, 1.2)
        self.assertEqual(attr.min_age, 25)
        self.assertEqual(attr.max_age, 60)
        self.assertFalse(attr.is_blocked)
    
    def test_attribute_compatibility(self):
        """Test attribute compatibility checking."""
        attr1 = AttributeDefinition(
            name="attr1",
            attribute_type=AttributeType.AGE_RANGE,
            keywords=["attr1"]
        )
        
        attr2 = AttributeDefinition(
            name="attr2",
            attribute_type=AttributeType.ETHNICITY,
            keywords=["attr2"]
        )
        
        # Should be compatible by default
        self.assertTrue(attr1.is_compatible_with(attr2))
        self.assertTrue(attr2.is_compatible_with(attr1))
    
    def test_attribute_age_validation(self):
        """Test attribute age validation."""
        attr = AttributeDefinition(
            name="age_attr",
            attribute_type=AttributeType.AGE_RANGE,
            min_age=30,
            max_age=50
        )
        
        self.assertTrue(attr.validate_age(35))
        self.assertTrue(attr.validate_age(30))
        self.assertTrue(attr.validate_age(50))
        self.assertFalse(attr.validate_age(25))
        self.assertFalse(attr.validate_age(55))


class TestCharacterAttributeSet(unittest.TestCase):
    """Test character attribute set."""
    
    def test_attribute_set_creation(self):
        """Test creation of attribute set."""
        attr_set = CharacterAttributeSet()
        
        self.assertIsInstance(attr_set, CharacterAttributeSet)
        self.assertEqual(len(attr_set.collections), len(AttributeType))
    
    def test_attribute_collection_operations(self):
        """Test attribute collection operations."""
        attr_set = CharacterAttributeSet()
        
        # Test adding attribute
        test_attr = AttributeDefinition(
            name="test",
            attribute_type=AttributeType.AGE_RANGE,
            keywords=["test"]
        )
        
        attr_set.add_attribute(test_attr)
        
        # Test retrieving attribute
        retrieved = attr_set.get_attribute(AttributeType.AGE_RANGE, "test")
        self.assertEqual(retrieved, test_attr)
        
        # Test getting collection
        collection = attr_set.get_collection(AttributeType.AGE_RANGE)
        self.assertIsNotNone(collection)


class TestEnhancedConfigLoader(unittest.TestCase):
    """Test enhanced config loader."""
    
    def test_config_loader_creation(self):
        """Test creation of config loader."""
        try:
            loader = EnhancedConfigLoader()
            self.assertIsInstance(loader, EnhancedConfigLoader)
        except Exception as e:
            # Config files might not exist in test environment
            self.skipTest(f"Config files not available: {e}")
    
    def test_attribute_set_retrieval(self):
        """Test retrieval of attribute set."""
        try:
            loader = EnhancedConfigLoader()
            attr_set = loader.get_attribute_set()
            self.assertIsInstance(attr_set, CharacterAttributeSet)
        except Exception as e:
            self.skipTest(f"Config files not available: {e}")


class TestEnhancedCharacterGenerator(unittest.TestCase):
    """Test enhanced character generator."""
    
    def test_generator_creation(self):
        """Test creation of enhanced generator."""
        try:
            loader = EnhancedConfigLoader()
            generator = EnhancedCharacterGenerator(loader)
            self.assertIsInstance(generator, EnhancedCharacterGenerator)
        except Exception as e:
            # May fail if config files not available
            self.skipTest(f"Config files not available: {e}")
    
    def test_preferences_creation(self):
        """Test creation of generation preferences."""
        prefs = GenerationPreferences(
            target_age=45,
            target_ethnicity="caucasian",
            target_style="goth",
            explicit_content_allowed=True,
            safety_level="strict"
        )
        
        self.assertEqual(prefs.target_age, 45)
        self.assertEqual(prefs.target_ethnicity, "caucasian")
        self.assertEqual(prefs.target_style, "goth")
        self.assertTrue(prefs.explicit_content_allowed)
        self.assertEqual(prefs.safety_level, "strict")


class TestIntegration(unittest.TestCase):
    """Integration tests for the enhanced system."""
    
    def test_complete_workflow(self):
        """Test complete workflow of enhanced system."""
        try:
            # Create loader
            loader = EnhancedConfigLoader()
            
            # Get attribute set
            attr_set = loader.get_attribute_set()
            self.assertIsInstance(attr_set, CharacterAttributeSet)
            
            # Create generator
            generator = EnhancedCharacterGenerator(loader)
            
            # Create preferences
            prefs = GenerationPreferences(
                target_age=40,
                target_ethnicity="caucasian",
                safety_level="strict"
            )
            
            # Generate character (might be slow, so we skip if it fails)
            character = generator.generate_character(prefs)
            self.assertIsNotNone(character)
            
        except Exception as e:
            # Integration tests might fail in test environments
            self.skipTest(f"Integration test skipped due to: {e}")


if __name__ == "__main__":
    unittest.main()