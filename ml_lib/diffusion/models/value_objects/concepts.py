"""Concept and emphasis value objects.

This module provides type-safe concept mapping and emphasis classes WITHOUT using
dicts, tuples, or any.
"""

from dataclasses import dataclass, field
from typing import Iterator


@dataclass(frozen=True)
class Concept:
    """Represents a single concept with associated keywords.

    Attributes:
        name: Concept name (e.g., "character", "style").
        _keywords: Internal list of keywords associated with this concept.

    Example:
        >>> concept = Concept("character", ["woman", "person", "portrait"])
        >>> print(concept.name)
        character
        >>> print(concept.keyword_count)
        3
    """

    name: str
    _keywords: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate concept."""
        if not self.name:
            raise ValueError("Concept name cannot be empty")
        if not self._keywords:
            raise ValueError(f"Concept '{self.name}' must have at least one keyword")

    @property
    def keyword_count(self) -> int:
        """Number of keywords in this concept."""
        return len(self._keywords)

    def has_keyword(self, keyword: str) -> bool:
        """Check if keyword is in this concept.

        Args:
            keyword: Keyword to check.

        Returns:
            True if keyword is present, False otherwise.
        """
        return keyword.lower() in (k.lower() for k in self._keywords)

    def matches_any(self, text: str) -> bool:
        """Check if text contains any of the keywords.

        Args:
            text: Text to search.

        Returns:
            True if any keyword is found, False otherwise.
        """
        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in self._keywords)

    def get_keywords(self) -> list[str]:
        """Get all keywords.

        Returns:
            List of all keywords.
        """
        return self._keywords.copy()


@dataclass(frozen=True)
class ConceptMap:
    """Collection of concepts with their keywords.

    Attributes:
        _concepts: Internal list of Concept instances.

    Example:
        >>> concepts = [
        ...     Concept("character", ["woman", "person"]),
        ...     Concept("style", ["photorealistic", "anime"])
        ... ]
        >>> concept_map = ConceptMap(concepts)
        >>> keywords = concept_map.get_keywords("character")
        >>> print(keywords)
        ['woman', 'person']
    """

    _concepts: list[Concept] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate concept map."""
        if not self._concepts:
            raise ValueError("ConceptMap cannot be empty")

        # Check for duplicate concept names
        names = [c.name for c in self._concepts]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate concept names found")

    @property
    def concept_count(self) -> int:
        """Number of concepts."""
        return len(self._concepts)

    @property
    def concept_names(self) -> list[str]:
        """List of concept names."""
        return [c.name for c in self._concepts]

    def get_concept(self, name: str) -> Concept | None:
        """Get concept by name.

        Args:
            name: Concept name.

        Returns:
            Concept instance or None if not found.
        """
        for c in self._concepts:
            if c.name == name:
                return c
        return None

    def get_keywords(self, concept_name: str) -> list[str] | None:
        """Get keywords for a concept.

        Args:
            concept_name: Name of the concept.

        Returns:
            List of keywords or None if concept not found.
        """
        concept = self.get_concept(concept_name)
        return concept.get_keywords() if concept else None

    def has_concept(self, name: str) -> bool:
        """Check if concept exists.

        Args:
            name: Concept name.

        Returns:
            True if exists, False otherwise.
        """
        return any(c.name == name for c in self._concepts)

    def get_all(self) -> list[Concept]:
        """Get all concepts.

        Returns:
            List of all Concept instances.
        """
        return self._concepts.copy()

    def __iter__(self) -> Iterator[Concept]:
        """Iterate over concepts."""
        return iter(self._concepts)


@dataclass(frozen=True)
class Emphasis:
    """Represents emphasis for a specific keyword.

    Attributes:
        keyword: The keyword to emphasize.
        weight: Emphasis weight (typically 0.0 to 2.0).

    Example:
        >>> emphasis = Emphasis("photorealistic", 1.5)
        >>> print(emphasis.keyword, emphasis.weight)
        photorealistic 1.5
    """

    keyword: str
    weight: float

    def __post_init__(self) -> None:
        """Validate emphasis."""
        if not self.keyword:
            raise ValueError("Emphasis keyword cannot be empty")
        if self.weight < 0:
            raise ValueError(f"Weight cannot be negative, got {self.weight}")
        if self.weight > 3.0:
            raise ValueError(f"Weight too high (max 3.0), got {self.weight}")

    @property
    def is_deemphasis(self) -> bool:
        """Check if this is a de-emphasis (weight < 1.0)."""
        return self.weight < 1.0

    @property
    def is_neutral(self) -> bool:
        """Check if this is neutral (weight ≈ 1.0)."""
        return abs(self.weight - 1.0) < 0.01

    @property
    def is_emphasis(self) -> bool:
        """Check if this is an emphasis (weight > 1.0)."""
        return self.weight > 1.0


@dataclass(frozen=True)
class EmphasisMap:
    """Collection of keyword emphases.

    Attributes:
        _emphases: Internal list of Emphasis instances.

    Example:
        >>> emphases = [
        ...     Emphasis("photorealistic", 1.5),
        ...     Emphasis("detailed", 1.2)
        ... ]
        >>> emphasis_map = EmphasisMap(emphases)
        >>> print(emphasis_map.get_weight("photorealistic"))
        1.5
    """

    _emphases: list[Emphasis] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate emphasis map."""
        if not self._emphases:
            raise ValueError("EmphasisMap cannot be empty")

        # Check for duplicate keywords
        keywords = [e.keyword for e in self._emphases]
        if len(keywords) != len(set(keywords)):
            raise ValueError("Duplicate keywords found")

    @property
    def count(self) -> int:
        """Number of emphases."""
        return len(self._emphases)

    @property
    def keywords(self) -> list[str]:
        """List of emphasized keywords."""
        return [e.keyword for e in self._emphases]

    def get_weight(self, keyword: str) -> float | None:
        """Get weight for a keyword.

        Args:
            keyword: Keyword to look up.

        Returns:
            Weight value or None if not found.
        """
        for e in self._emphases:
            if e.keyword == keyword:
                return e.weight
        return None

    def has_keyword(self, keyword: str) -> bool:
        """Check if keyword is emphasized.

        Args:
            keyword: Keyword to check.

        Returns:
            True if emphasized, False otherwise.
        """
        return any(e.keyword == keyword for e in self._emphases)

    def get_emphasized(self) -> list[str]:
        """Get list of keywords with weight > 1.0.

        Returns:
            List of emphasized keywords.
        """
        return [e.keyword for e in self._emphases if e.is_emphasis]

    def get_deemphasized(self) -> list[str]:
        """Get list of keywords with weight < 1.0.

        Returns:
            List of de-emphasized keywords.
        """
        return [e.keyword for e in self._emphases if e.is_deemphasis]

    def get_all(self) -> list[Emphasis]:
        """Get all emphases.

        Returns:
            List of all Emphasis instances.
        """
        return self._emphases.copy()

    def __iter__(self) -> Iterator[Emphasis]:
        """Iterate over emphases."""
        return iter(self._emphases)


__all__ = ["Concept", "ConceptMap", "Emphasis", "EmphasisMap"]
