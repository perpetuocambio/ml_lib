"""Synthesis metadata container instead of generic dict."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SynthesisMetadata:
    """Structured metadata for synthesis elements instead of Dict[str, Any]."""

    # Common metadata fields
    extraction_method: str = ""
    source_document: str = ""
    confidence_basis: str = ""

    # Analysis context
    technique_used: str = ""
    analysis_phase: str = ""

    # Additional structured data
    tags: list[str] | None = None
    properties: list[str] | None = None  # Key-value pairs as strings

    @classmethod
    def from_field_list(cls, field_data: list[str]) -> "SynthesisMetadata":
        """Create from field list with safe defaults."""
        extraction_method = ""
        source_document = ""
        confidence_basis = ""
        technique_used = ""
        analysis_phase = ""
        tags = None
        properties = []

        for field in field_data:
            if ":" in field:
                key, value = field.split(":", 1)
                if key == "extraction_method":
                    extraction_method = value
                elif key == "source_document":
                    source_document = value
                elif key == "confidence_basis":
                    confidence_basis = value
                elif key == "technique_used":
                    technique_used = value
                elif key == "analysis_phase":
                    analysis_phase = value
                elif key == "tags" and "," in value:
                    tags = value.split(",")
                else:
                    properties.append(field)

        return cls(
            extraction_method=extraction_method,
            source_document=source_document,
            confidence_basis=confidence_basis,
            technique_used=technique_used,
            analysis_phase=analysis_phase,
            tags=tags,
            properties=properties,
        )
