from infrastructure.errors.error_context import ErrorContext
from infrastructure.errors.infrastructure_error import InfrastructureError


class ProcessingError(InfrastructureError):
    """Raised for errors during data processing."""

    def __init__(
        self,
        message: str,
        processing_stage: str | None = None,
        input_file: str | None = None,
        output_file: str | None = None,
        processed_rows: int | None = None,
        failed_rows: int | None = None,
        processing_duration: float | None = None,
        error_code: str | None = None,
        original_exception: Exception | None = None,
    ):
        context = ErrorContext.empty()

        if processing_stage:
            context = context.add_text_entry("processing_stage", processing_stage)
        if input_file:
            context = context.add_text_entry("input_file", input_file)
        if output_file:
            context = context.add_text_entry("output_file", output_file)
        if processed_rows is not None:
            context = context.add_numeric_entry("processed_rows", processed_rows)
        if failed_rows is not None:
            context = context.add_numeric_entry("failed_rows", failed_rows)
        if processing_duration is not None:
            context = context.add_numeric_entry(
                "processing_duration", processing_duration
            )

        super().__init__(
            message=message,
            error_code=error_code or "PROCESSING_ERROR",
            context=context,
            original_exception=original_exception,
        )

    def is_retryable(self) -> bool:
        """Determine if processing error is retryable."""
        error_code = self.error_code
        if error_code:
            # Retryable processing errors
            retryable_codes = {
                "TEMPORARY_IO_ERROR",
                "NETWORK_TIMEOUT",
                "RESOURCE_BUSY",
                "MEMORY_PRESSURE",
                "CONCURRENT_ACCESS",
            }
            return error_code in retryable_codes
        return False

    def get_user_friendly_message(self) -> str:
        """Get user-friendly processing error message."""
        processing_stage = self.get_text_context_value("processing_stage")
        input_file = self.get_text_context_value("input_file")
        failed_rows = self.get_numeric_context_value("failed_rows")

        if processing_stage and input_file:
            return (
                f"Processing failed during '{processing_stage}' for file '{input_file}'"
            )
        elif processing_stage:
            return f"Processing failed during '{processing_stage}'"
        elif input_file:
            return f"Processing failed for file '{input_file}'"
        elif failed_rows:
            return f"Processing failed with {failed_rows} failed rows"
        else:
            return f"Data processing failed: {self.message}"

    def get_processing_summary(self) -> str:
        """Get comprehensive processing failure summary."""
        stage = self.get_text_context_value("processing_stage") or "unknown"
        processed = self.get_numeric_context_value("processed_rows") or 0
        failed = self.get_numeric_context_value("failed_rows") or 0
        duration = self.get_numeric_context_value("processing_duration")

        summary = f"Stage: {stage}, Processed: {processed}, Failed: {failed}"
        if duration:
            summary += f", Duration: {duration:.2f}s"
        return summary

    @classmethod
    def validation_failed(
        cls, input_file: str, failed_rows: int, processing_stage: str = "validation"
    ) -> "ProcessingError":
        """Create error for data validation failure."""
        return cls(
            message=f"Data validation failed with {failed_rows} invalid rows",
            processing_stage=processing_stage,
            input_file=input_file,
            failed_rows=failed_rows,
            error_code="VALIDATION_FAILED",
        )

    @classmethod
    def transformation_failed(
        cls,
        processing_stage: str,
        processed_rows: int,
        original_exception: Exception | None = None,
    ) -> "ProcessingError":
        """Create error for data transformation failure."""
        return cls(
            message=f"Data transformation failed at stage '{processing_stage}'",
            processing_stage=processing_stage,
            processed_rows=processed_rows,
            error_code="TRANSFORMATION_FAILED",
            original_exception=original_exception,
        )

    @classmethod
    def file_processing_failed(
        cls,
        input_file: str,
        output_file: str | None = None,
        original_exception: Exception | None = None,
    ) -> "ProcessingError":
        """Create error for file processing failure."""
        message = f"Failed to process file '{input_file}'"
        if output_file:
            message += f" to '{output_file}'"
        return cls(
            message=message,
            input_file=input_file,
            output_file=output_file,
            error_code="FILE_PROCESSING_FAILED",
            original_exception=original_exception,
        )

    @classmethod
    def resource_exhausted(
        cls, processing_stage: str, processed_rows: int, processing_duration: float
    ) -> "ProcessingError":
        """Create error for resource exhaustion during processing."""
        return cls(
            message=f"Processing resources exhausted during '{processing_stage}'",
            processing_stage=processing_stage,
            processed_rows=processed_rows,
            processing_duration=processing_duration,
            error_code="RESOURCE_EXHAUSTED",
        )
