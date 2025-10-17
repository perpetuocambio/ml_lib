"""Value objects for parameter modifications."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ParameterModificationEntry:
    """Single parameter modification."""

    parameter_name: str
    old_value: str
    new_value: str


@dataclass(frozen=True)
class ParameterModifications:
    """Collection of parameter modifications."""

    modifications: tuple[ParameterModificationEntry, ...]

    @classmethod
    def from_dict(cls, modifications_dict: dict[str, str]) -> "ParameterModifications":
        """
        Create ParameterModifications from a dict.

        Args:
            modifications_dict: Dict mapping parameter names to new values

        Returns:
            ParameterModifications instance
        """
        entries = tuple(
            ParameterModificationEntry(
                parameter_name=key, old_value="", new_value=value
            )
            for key, value in modifications_dict.items()
        )
        return cls(modifications=entries)

    def to_dict(self) -> dict[str, str]:
        """Convert to dict for backwards compatibility."""
        return {mod.parameter_name: mod.new_value for mod in self.modifications}
