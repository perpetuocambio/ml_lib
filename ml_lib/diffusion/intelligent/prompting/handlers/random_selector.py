"""Random attribute selector with uniform probabilities.

IMPORTANT: This selector uses UNIFORM probabilities for all valid attributes.
It does NOT use the probability field from AttributeDefinition.

This is intentional because the content is fantasy-based and should not follow
real-world statistical distributions.
"""

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ml_lib.diffusion.intelligent.prompting.enhanced_attributes import (
        AttributeCollection,
        AttributeDefinition,
    )


class RandomAttributeSelector:
    """Selects attributes with uniform probability (no statistical bias).

    This class replaces the weighted selection that used probability values
    from YAML configuration files.

    **Key principle**: All valid (non-blocked) attributes have equal probability.
    Fantasy content should not follow real-world statistics.

    Example:
        If there are 5 hair colors available, each has exactly 20% (1/5) chance.
        The probability field in AttributeDefinition is IGNORED for selection.
    """

    @staticmethod
    def select_random(collection: "AttributeCollection") -> "AttributeDefinition | None":
        """Select a random attribute with uniform probability.

        Args:
            collection: Collection to select from

        Returns:
            Randomly selected attribute, or None if no valid attributes available

        Note:
            - Blocked attributes are excluded
            - All remaining attributes have equal probability (1/N)
            - Does NOT use the probability field from AttributeDefinition
        """
        # Get all non-blocked attributes
        available = [
            attr
            for attr in collection.all_attributes()
            if not attr.is_blocked
        ]

        if not available:
            return None

        # Uniform random selection - all have equal probability
        return random.choice(available)

    @staticmethod
    def select_compatible(
        collection: "AttributeCollection",
        existing_attributes: list["AttributeDefinition"],
    ) -> "AttributeDefinition | None":
        """Select a random compatible attribute with uniform probability.

        Args:
            collection: Collection to select from
            existing_attributes: Already selected attributes to check compatibility with

        Returns:
            Randomly selected compatible attribute, or None if no compatible options

        Note:
            - Filters out blocked attributes
            - Filters out incompatible attributes
            - All remaining compatible attributes have equal probability
        """
        # Get compatible attributes (this already filters blocked ones)
        compatible = collection.get_compatible_attributes(existing_attributes)

        if not compatible:
            return None

        # Uniform random selection from compatible options
        return random.choice(compatible)

    @staticmethod
    def select_multiple(
        collection: "AttributeCollection",
        count: int,
    ) -> list["AttributeDefinition"]:
        """Select multiple random attributes without replacement.

        Args:
            collection: Collection to select from
            count: Number of attributes to select

        Returns:
            List of randomly selected attributes (may be fewer than count if not enough available)

        Note:
            - Uses random.sample() for selection without replacement
            - All valid attributes have equal probability
            - Blocked attributes are excluded
        """
        # Get all non-blocked attributes
        available = [
            attr
            for attr in collection.all_attributes()
            if not attr.is_blocked
        ]

        if not available:
            return []

        # Select without replacement, up to available count
        k = min(count, len(available))
        return random.sample(available, k)
