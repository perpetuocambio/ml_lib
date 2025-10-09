"""Types of configuration values."""

from enum import Enum


class ConfigValueType(Enum):
    """Types of configuration values."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    NESTED = "nested"
    NULL = "null"
