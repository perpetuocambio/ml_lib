"""Configuration validation utilities."""

import re
from urllib.parse import urlparse

from infrastructure.config.types.validation_value import ValidationValue


class ConfigValidator:
    """Utility class for common configuration validations."""

    @staticmethod
    def validate_url(url: str, field_name: str = "URL") -> list[str]:
        """Validate URL format.

        Args:
            url: URL to validate.
            field_name: Name of the field for error messages.

        Returns:
            List of validation errors.
        """
        errors = []
        if not url:
            errors.append(f"{field_name} cannot be empty")
            return errors

        try:
            parsed = urlparse(url)
            if not parsed.scheme:
                errors.append(f"{field_name} must include scheme (http/https)")
            if not parsed.netloc:
                errors.append(f"{field_name} must include domain")
        except Exception:
            errors.append(f"{field_name} has invalid format")

        return errors

    @staticmethod
    def validate_port(port: int, field_name: str = "Port") -> list[str]:
        """Validate port number.

        Args:
            port: Port number to validate.
            field_name: Name of the field for error messages.

        Returns:
            List of validation errors.
        """
        errors = []
        if port < 1 or port > 65535:
            errors.append(f"{field_name} must be between 1 and 65535")
        return errors

    @staticmethod
    def validate_timeout(timeout: int, field_name: str = "Timeout") -> list[str]:
        """Validate timeout value.

        Args:
            timeout: Timeout in seconds to validate.
            field_name: Name of the field for error messages.

        Returns:
            List of validation errors.
        """
        errors = []
        if timeout <= 0:
            errors.append(f"{field_name} must be greater than 0")
        if timeout > 3600:  # 1 hour max
            errors.append(f"{field_name} cannot exceed 3600 seconds")
        return errors

    @staticmethod
    def validate_api_key(api_key: str, field_name: str = "API Key") -> list[str]:
        """Validate API key format.

        Args:
            api_key: API key to validate.
            field_name: Name of the field for error messages.

        Returns:
            List of validation errors.
        """
        errors = []
        if not api_key:
            errors.append(f"{field_name} cannot be empty")
            return errors

        if len(api_key) < 8:
            errors.append(f"{field_name} must be at least 8 characters")

        return errors

    @staticmethod
    def validate_model_name(
        model_name: str, field_name: str = "Model name"
    ) -> list[str]:
        """Validate model name format.

        Args:
            model_name: Model name to validate.
            field_name: Name of the field for error messages.

        Returns:
            List of validation errors.
        """
        errors = []
        if not model_name:
            errors.append(f"{field_name} cannot be empty")
            return errors

        # Allow alphanumeric, hyphens, underscores, colons (for model versions)
        if not re.match(r"^[a-zA-Z0-9\-_:\.]+$", model_name):
            errors.append(f"{field_name} contains invalid characters")

        return errors

    @staticmethod
    def validate_required_string(value: str, field_name: str) -> list[str]:
        """Validate required string field.

        Args:
            value: String value to validate.
            field_name: Name of the field for error messages.

        Returns:
            List of validation errors.
        """
        errors = []
        if not value or not value.strip():
            errors.append(f"{field_name} is required and cannot be empty")
        return errors

    @staticmethod
    def validate_positive_int(value: int, field_name: str) -> list[str]:
        """Validate positive integer.

        Args:
            value: Integer value to validate.
            field_name: Name of the field for error messages.

        Returns:
            List of validation errors.
        """
        errors = []
        if value <= 0:
            errors.append(f"{field_name} must be a positive integer")
        return errors

    @staticmethod
    def validate_enum_value(
        value: ValidationValue, enum_class: type, field_name: str
    ) -> list[str]:
        """Validate enum value.

        Args:
            value: Value to validate.
            enum_class: Enum class to validate against.
            field_name: Name of the field for error messages.

        Returns:
            List of validation errors.
        """
        errors = []
        try:
            if value not in [e.value for e in enum_class]:
                valid_values = [e.value for e in enum_class]
                errors.append(f"{field_name} must be one of: {valid_values}")
        except Exception:
            errors.append(f"{field_name} has invalid enum value")
        return errors
