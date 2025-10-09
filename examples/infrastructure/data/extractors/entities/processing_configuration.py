from typing import TypedDict

from infrastructure.data.extractors.entities.handle_missing_config import (
    HandleMissingConfig,
)


class ProcessingConfiguration(TypedDict, total=False):
    """
    Strongly-typed configuration for data processing operations.
    'total=False' makes all keys optional.
    """

    handle_missing: HandleMissingConfig
