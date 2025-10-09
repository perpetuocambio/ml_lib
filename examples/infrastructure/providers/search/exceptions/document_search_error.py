"""Document search error exception."""


class DocumentSearchError(Exception):
    """Exception raised when document search fails."""

    def __init__(
        self, message: str, project_id: str = None, original_error: Exception = None
    ):
        self.project_id = project_id
        self.original_error = original_error
        super().__init__(f"Document search error: {message}")
