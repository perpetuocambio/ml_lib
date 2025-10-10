"""Concept map type for prompt analysis."""

from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass
class ConceptMap:
    """Map of detected concepts by category.

    Replaces dict[str, list[str]] with strongly typed class.
    """

    concepts_by_category: Dict[str, List[str]] = field(default_factory=dict)

    def add_concept(self, category: str, concept: str) -> None:
        """Add a concept to a category.

        Args:
            category: Category name (e.g., 'anatomy', 'clothing', 'activity')
            concept: Concept to add
        """
        if category not in self.concepts_by_category:
            self.concepts_by_category[category] = []
        if concept not in self.concepts_by_category[category]:
            self.concepts_by_category[category].append(concept)

    def add_concepts(self, category: str, concepts: List[str]) -> None:
        """Add multiple concepts to a category.

        Args:
            category: Category name
            concepts: List of concepts to add
        """
        if category not in self.concepts_by_category:
            self.concepts_by_category[category] = []

        for concept in concepts:
            if concept not in self.concepts_by_category[category]:
                self.concepts_by_category[category].append(concept)

    def get_concepts(self, category: str) -> List[str]:
        """Get concepts for a category.

        Args:
            category: Category name

        Returns:
            List of concepts (empty if category not found)
        """
        return self.concepts_by_category.get(category, [])

    def has_category(self, category: str) -> bool:
        """Check if a category exists.

        Args:
            category: Category name

        Returns:
            True if category exists
        """
        return category in self.concepts_by_category

    def get_concept_count(self, category: str) -> int:
        """Get number of concepts in a category.

        Args:
            category: Category name

        Returns:
            Number of concepts
        """
        return len(self.get_concepts(category))

    @property
    def total_concept_count(self) -> int:
        """Total number of concepts across all categories."""
        return sum(len(concepts) for concepts in self.concepts_by_category.values())

    @property
    def categories(self) -> List[str]:
        """List of all categories."""
        return list(self.concepts_by_category.keys())

    @property
    def all_concepts(self) -> Set[str]:
        """Set of all unique concepts."""
        concepts = set()
        for concept_list in self.concepts_by_category.values():
            concepts.update(concept_list)
        return concepts

    def to_dict(self) -> Dict[str, List[str]]:
        """Convert to dictionary for backward compatibility.

        Returns:
            Dictionary mapping categories to concept lists
        """
        return self.concepts_by_category.copy()
