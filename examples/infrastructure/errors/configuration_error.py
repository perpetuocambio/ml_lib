from infrastructure.errors.error_context import ErrorContext
from infrastructure.errors.infrastructure_error import InfrastructureError


class ConfigurationError(InfrastructureError):
    """Raised for configuration-related errors."""

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        config_value: str | int | float | bool | None = None,
        config_file: str | None = None,
        error_code: str | None = None,
        original_exception: Exception | None = None,
    ):
        context = ErrorContext.empty()
        if config_key:
            context = context.add_text_entry("config_key", config_key)
        if config_file:
            context = context.add_text_entry("config_file", config_file)
        if config_value is not None:
            if isinstance(config_value, str):
                context = context.add_text_entry("config_value", config_value)
            elif isinstance(config_value, bool):
                context = context.add_boolean_entry("config_value", config_value)
            elif isinstance(config_value, int | float):
                context = context.add_numeric_entry("config_value", config_value)

        super().__init__(
            message=message,
            error_code=error_code or "CONFIG_ERROR",
            context=context,
            original_exception=original_exception,
        )

    def is_retryable(self) -> bool:
        """Configuration errors are usually not retryable."""
        return False

    def get_user_friendly_message(self) -> str:
        """Get user-friendly configuration error message."""
        config_key = self.get_text_context_value("config_key")
        config_file = self.get_text_context_value("config_file")

        if config_key and config_file:
            return f"Configuration error in '{config_file}' for key '{config_key}'"
        elif config_key:
            return f"Configuration error for key '{config_key}'"
        elif config_file:
            return f"Configuration error in '{config_file}'"
        else:
            return f"Configuration error: {self.message}"

    @classmethod
    def missing_key(
        cls, key: str, config_file: str | None = None
    ) -> "ConfigurationError":
        """Create error for missing configuration key."""
        message = f"Required configuration key '{key}' is missing"
        if config_file:
            message += f" from '{config_file}'"
        return cls(
            message=message,
            config_key=key,
            config_file=config_file,
            error_code="CONFIG_MISSING_KEY",
        )

    @classmethod
    def invalid_value(
        cls,
        key: str,
        value: str | int | float | bool,
        expected_type: str | None = None,
        config_file: str | None = None,
    ) -> "ConfigurationError":
        """Create error for invalid configuration value."""
        message = f"Invalid value for configuration key '{key}': {value}"
        if expected_type:
            message += f" (expected {expected_type})"
        return cls(
            message=message,
            config_key=key,
            config_value=value,
            config_file=config_file,
            error_code="CONFIG_INVALID_VALUE",
        )

    @classmethod
    def file_not_found(cls, config_file: str) -> "ConfigurationError":
        """Create error for missing configuration file."""
        return cls(
            message=f"Configuration file not found: {config_file}",
            config_file=config_file,
            error_code="CONFIG_FILE_NOT_FOUND",
        )
