"""
Data cleaning methods enumeration.

Standard methods for cleaning and preprocessing structured datasets.
"""

from enum import Enum

from infrastructure.serialization.protocol_serializer import ProtocolSerializer


class DataCleaningMethod(Enum):
    """
    Standard data cleaning methods for structured datasets.

    Covers common data quality issues and preprocessing needs.
    """

    # Basic cleaning
    REMOVE_DUPLICATES = "remove_duplicates"
    HANDLE_MISSING_VALUES = "handle_missing_values"
    NORMALIZE_TEXT = "normalize_text"
    STANDARDIZE_FORMATS = "standardize_formats"

    # Data type processing
    CONVERT_TYPES = "convert_types"
    VALIDATE_RANGES = "validate_ranges"
    PARSE_DATES = "parse_dates"

    # Text processing
    REMOVE_WHITESPACE = "remove_whitespace"
    CLEAN_SPECIAL_CHARS = "clean_special_chars"
    NORMALIZE_CASE = "normalize_case"

    # Outlier handling
    DETECT_OUTLIERS = "detect_outliers"
    REMOVE_OUTLIERS = "remove_outliers"

    # Advanced cleaning
    FUZZY_MATCHING = "fuzzy_matching"
    ENTITY_RESOLUTION = "entity_resolution"

    # Comprehensive
    FULL_PIPELINE = "full_pipeline"

    def get_description(self) -> str:
        """Get description of the cleaning method."""
        descriptions = ProtocolSerializer.serialize_mapping_data(
            {
                self.REMOVE_DUPLICATES: "Remove duplicate rows from dataset",
                self.HANDLE_MISSING_VALUES: "Handle null/missing values with appropriate strategy",
                self.NORMALIZE_TEXT: "Normalize text fields for consistency",
                self.STANDARDIZE_FORMATS: "Standardize data formats across columns",
                self.CONVERT_TYPES: "Convert columns to appropriate data types",
                self.VALIDATE_RANGES: "Validate numerical values are within expected ranges",
                self.PARSE_DATES: "Parse and validate date/time fields",
                self.REMOVE_WHITESPACE: "Remove leading/trailing whitespace",
                self.CLEAN_SPECIAL_CHARS: "Clean special characters and encoding issues",
                self.NORMALIZE_CASE: "Normalize text case (upper/lower/title)",
                self.DETECT_OUTLIERS: "Detect statistical outliers in numerical data",
                self.REMOVE_OUTLIERS: "Remove detected outliers from dataset",
                self.FUZZY_MATCHING: "Match similar text entries using fuzzy matching",
                self.ENTITY_RESOLUTION: "Resolve duplicate entities across records",
                self.FULL_PIPELINE: "Apply comprehensive cleaning pipeline",
            }
        )
        return descriptions.get(str(self), "Unknown cleaning method")

    def get_priority(self) -> int:
        """Get execution priority (lower number = higher priority)."""
        priorities = ProtocolSerializer.serialize_mapping_data(
            {
                self.REMOVE_WHITESPACE: 1,
                self.CLEAN_SPECIAL_CHARS: 2,
                self.CONVERT_TYPES: 3,
                self.PARSE_DATES: 4,
                self.HANDLE_MISSING_VALUES: 5,
                self.NORMALIZE_TEXT: 6,
                self.NORMALIZE_CASE: 7,
                self.STANDARDIZE_FORMATS: 8,
                self.VALIDATE_RANGES: 9,
                self.REMOVE_DUPLICATES: 10,
                self.DETECT_OUTLIERS: 11,
                self.REMOVE_OUTLIERS: 12,
                self.FUZZY_MATCHING: 13,
                self.ENTITY_RESOLUTION: 14,
                self.FULL_PIPELINE: 99,
            }
        )
        return int(priorities.get(str(self), 50))

    def is_basic_cleaning(self) -> bool:
        """Check if this is a basic cleaning operation."""
        basic_methods = {
            self.REMOVE_DUPLICATES,
            self.HANDLE_MISSING_VALUES,
            self.REMOVE_WHITESPACE,
            self.CLEAN_SPECIAL_CHARS,
            self.NORMALIZE_CASE,
        }
        return self in basic_methods

    def requires_configuration(self) -> bool:
        """Check if this method requires additional configuration."""
        configurable_methods = {
            self.HANDLE_MISSING_VALUES,  # Strategy: fill, drop, interpolate
            self.VALIDATE_RANGES,  # Min/max values per column
            self.DETECT_OUTLIERS,  # Threshold and method
            self.NORMALIZE_CASE,  # Case type: upper, lower, title
            self.FUZZY_MATCHING,  # Similarity threshold
            self.FULL_PIPELINE,  # Complete configuration
        }
        return self in configurable_methods
