"""Validation value type enumeration."""

from enum import Enum


class ValidationValueType(Enum):
    """Types of values that can be validated."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ENUM = "enum"
    URL = "url"
    EMAIL = "email"
    PATH = "path"
    NULL = "null"
