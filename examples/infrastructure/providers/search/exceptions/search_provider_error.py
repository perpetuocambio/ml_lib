"""Search provider error exception."""


class SearchProviderError(Exception):
    """Exception raised when search provider fails."""

    def __init__(self, message: str, provider: str, original_error: Exception = None):
        self.provider = provider
        self.original_error = original_error
        super().__init__(f"Search provider '{provider}' error: {message}")
