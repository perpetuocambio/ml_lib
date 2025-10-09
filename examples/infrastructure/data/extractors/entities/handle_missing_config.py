from typing import Literal, TypedDict


class HandleMissingConfig(TypedDict, total=False):
    """Configuration for the 'handle_missing' operation."""

    strategy: Literal["drop", "fill_mean"]
