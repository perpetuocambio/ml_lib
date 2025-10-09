from infrastructure.errors.error_context import ErrorContext
from infrastructure.errors.infrastructure_error import InfrastructureError


class LLMProviderError(InfrastructureError):
    """Raised for errors related to LLM provider interactions."""

    def __init__(
        self,
        message: str,
        provider_name: str | None = None,
        model_name: str | None = None,
        api_endpoint: str | None = None,
        request_id: str | None = None,
        tokens_consumed: int | None = None,
        response_time: float | None = None,
        rate_limited: bool = False,
        error_code: str | None = None,
        original_exception: Exception | None = None,
    ):
        context = ErrorContext.empty()

        if provider_name:
            context = context.add_text_entry("provider_name", provider_name)
        if model_name:
            context = context.add_text_entry("model_name", model_name)
        if api_endpoint:
            context = context.add_text_entry("api_endpoint", api_endpoint)
        if request_id:
            context = context.add_text_entry("request_id", request_id)
        if tokens_consumed is not None:
            context = context.add_numeric_entry("tokens_consumed", tokens_consumed)
        if response_time is not None:
            context = context.add_numeric_entry("response_time", response_time)
        if rate_limited:
            context = context.add_boolean_entry("rate_limited", rate_limited)

        super().__init__(
            message=message,
            error_code=error_code or "LLM_PROVIDER_ERROR",
            context=context,
            original_exception=original_exception,
        )

    def is_retryable(self) -> bool:
        """Determine if LLM provider error is retryable."""
        # Check if rate limited
        if self.get_boolean_context_value("rate_limited"):
            return True

        error_code = self.error_code
        if error_code:
            # Retryable LLM provider errors
            retryable_codes = {
                "RATE_LIMITED",
                "SERVICE_UNAVAILABLE",
                "TIMEOUT",
                "TEMPORARY_FAILURE",
                "SERVER_OVERLOADED",
                "QUOTA_EXCEEDED",
            }
            return error_code in retryable_codes
        return False

    def get_user_friendly_message(self) -> str:
        """Get user-friendly LLM provider error message."""
        provider_name = self.get_text_context_value("provider_name")
        model_name = self.get_text_context_value("model_name")
        rate_limited = self.get_boolean_context_value("rate_limited")

        if rate_limited:
            return f"Rate limit exceeded for {provider_name or 'LLM provider'}"
        elif provider_name and model_name:
            return f"LLM error with {provider_name} model '{model_name}'"
        elif provider_name:
            return f"LLM error with provider '{provider_name}'"
        else:
            return f"LLM provider error: {self.message}"

    def get_usage_summary(self) -> str:
        """Get LLM usage summary for this request."""
        provider = self.get_text_context_value("provider_name") or "unknown"
        model = self.get_text_context_value("model_name") or "unknown"
        tokens = self.get_numeric_context_value("tokens_consumed") or 0
        response_time = self.get_numeric_context_value("response_time")

        summary = f"Provider: {provider}, Model: {model}, Tokens: {tokens}"
        if response_time:
            summary += f", Response time: {response_time:.2f}s"
        return summary

    def is_rate_limited(self) -> bool:
        """Check if this error was due to rate limiting."""
        return self.get_boolean_context_value("rate_limited") or False

    @classmethod
    def rate_limit_exceeded(
        cls,
        provider_name: str,
        model_name: str | None = None,
        request_id: str | None = None,
    ) -> "LLMProviderError":
        """Create error for rate limit exceeded."""
        message = f"Rate limit exceeded for {provider_name}"
        if model_name:
            message += f" model '{model_name}'"
        return cls(
            message=message,
            provider_name=provider_name,
            model_name=model_name,
            request_id=request_id,
            rate_limited=True,
            error_code="RATE_LIMITED",
        )

    @classmethod
    def model_unavailable(
        cls, provider_name: str, model_name: str
    ) -> "LLMProviderError":
        """Create error for model unavailability."""
        return cls(
            message=f"Model '{model_name}' is unavailable from provider '{provider_name}'",
            provider_name=provider_name,
            model_name=model_name,
            error_code="MODEL_UNAVAILABLE",
        )

    @classmethod
    def api_timeout(
        cls, provider_name: str, api_endpoint: str, response_time: float
    ) -> "LLMProviderError":
        """Create error for API timeout."""
        return cls(
            message=f"API timeout for {provider_name} after {response_time:.2f}s",
            provider_name=provider_name,
            api_endpoint=api_endpoint,
            response_time=response_time,
            error_code="TIMEOUT",
        )

    @classmethod
    def quota_exceeded(
        cls, provider_name: str, tokens_consumed: int
    ) -> "LLMProviderError":
        """Create error for quota exceeded."""
        return cls(
            message=f"Quota exceeded for {provider_name} (consumed: {tokens_consumed} tokens)",
            provider_name=provider_name,
            tokens_consumed=tokens_consumed,
            error_code="QUOTA_EXCEEDED",
        )

    @classmethod
    def invalid_response(
        cls,
        provider_name: str,
        model_name: str,
        request_id: str | None = None,
        original_exception: Exception | None = None,
    ) -> "LLMProviderError":
        """Create error for invalid response format."""
        return cls(
            message=f"Invalid response format from {provider_name} model '{model_name}'",
            provider_name=provider_name,
            model_name=model_name,
            request_id=request_id,
            error_code="INVALID_RESPONSE",
            original_exception=original_exception,
        )
