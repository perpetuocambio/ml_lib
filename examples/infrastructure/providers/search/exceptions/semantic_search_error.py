"""Semantic search error exception."""


class SemanticSearchError(Exception):
    """Exception raised when semantic search fails."""

    def __init__(self, message: str, original_error: Exception = None):
        self.original_error = original_error
        super().__init__(f"Semantic search error: {message}")
