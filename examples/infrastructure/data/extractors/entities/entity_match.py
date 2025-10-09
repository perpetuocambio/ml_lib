"""Infrastructure entity match DTO - no Domain dependencies."""

from dataclasses import dataclass


@dataclass
class InfraEntityMatch:
    """Infrastructure entity match DTO - uses string instead of enum."""

    entity_text: str
    entity_type: str  # Infrastructure uses string instead of enum
    start_position: int
    end_position: int

    def get_position_span(self) -> int:
        """Get the length of the entity text span."""
        return self.end_position - self.start_position

    def is_valid_position(self) -> bool:
        """Check if positions are valid."""
        return (
            self.start_position >= 0
            and self.end_position >= 0
            and self.end_position >= self.start_position
        )
