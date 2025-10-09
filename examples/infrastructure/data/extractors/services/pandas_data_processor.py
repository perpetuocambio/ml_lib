"""Pandas-based data processing implementation - Architecture compliant."""

import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
from infrastructure.data.extractors.entities.data_processing_result import (
    DataProcessingResult,
)
from infrastructure.data.extractors.entities.processing_configuration import (
    ProcessingConfiguration,
)
from infrastructure.data.extractors.interfaces.dataset_processor_interface import (
    IInfraDatasetProcessor,
)
from infrastructure.serialization.protocol_serializer import ProtocolSerializer


class PandasDataProcessor(IInfraDatasetProcessor):
    """Pandas-based data processing - Infrastructure layer implementation."""

    def clean_dataset(
        self,
        dataset_id: str,
        source_file_path: Path,
        output_directory: Path,
        operations: list[str],  # Simple string operations
        configuration: ProcessingConfiguration | None = None,
    ) -> DataProcessingResult:
        """Process dataset using pandas - Infrastructure implementation."""
        start_time = datetime.now()
        processing_id = f"proc_{dataset_id}_{start_time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        config = configuration or ProtocolSerializer.serialize_dict_data()
        warnings = []
        errors = []
        operations_summary = ProtocolSerializer.serialize_dict_data()

        try:
            # Load dataset
            df = self._load_dataset(source_file_path)
            original_row_count = len(df)
            original_column_count = len(df.columns)

            # Apply operations
            for operation in operations:
                try:
                    df = self._apply_operation(df, operation, config)
                    operations_summary[operation] = "completed"
                except Exception as e:
                    error_msg = f"Error in {operation}: {str(e)}"
                    errors.append(error_msg)
                    operations_summary[operation] = "failed"

            # Save processed dataset
            output_file_path = output_directory / f"{processing_id}_processed.csv"
            df.to_csv(output_file_path, index=False)

            execution_time = (datetime.now() - start_time).total_seconds()

            return DataProcessingResult(
                processing_id=processing_id,
                dataset_id=dataset_id,
                source_file_path=source_file_path,
                output_file_path=output_file_path,
                original_row_count=original_row_count,
                final_row_count=len(df),
                original_column_count=original_column_count,
                final_column_count=len(df.columns),
                execution_time_seconds=execution_time,
                warnings=warnings,
                errors=errors,
                operations_summary=operations_summary,
            )

        except Exception as e:
            error_msg = f"Critical error in data processing: {str(e)}"
            errors.append(error_msg)
            execution_time = (datetime.now() - start_time).total_seconds()

            return DataProcessingResult(
                processing_id=processing_id,
                dataset_id=dataset_id,
                source_file_path=source_file_path,
                output_file_path=None,
                original_row_count=0,
                final_row_count=0,
                original_column_count=0,
                final_column_count=0,
                execution_time_seconds=execution_time,
                warnings=warnings,
                errors=errors,
                operations_summary=ProtocolSerializer.serialize_dict_data(),
            )

    def _load_dataset(self, file_path: Path) -> pd.DataFrame:
        """Load dataset from file using pandas."""
        if file_path.suffix.lower() == ".csv":
            return pd.read_csv(file_path)
        elif file_path.suffix.lower() == ".json":
            return pd.read_json(file_path)
        elif file_path.suffix.lower() in [".xlsx", ".xls"]:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def _apply_operation(
        self, df: pd.DataFrame, operation: str, config: ProcessingConfiguration
    ) -> pd.DataFrame:
        """Apply data processing operation."""
        if operation == "remove_duplicates":
            return df.drop_duplicates()
        elif operation == "handle_missing":
            if handle_missing_config := config.get("handle_missing"):
                strategy = handle_missing_config.get("strategy", "drop")
            else:
                strategy = "drop"

            if strategy == "drop":
                return df.dropna()
            elif strategy == "fill_mean":
                numeric_cols = df.select_dtypes(include=["number"]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                return df
            else:
                return df
        elif operation == "standardize_text":
            text_cols = df.select_dtypes(include=["object"]).columns
            df[text_cols] = df[text_cols].astype(str).apply(lambda x: x.str.lower())
            return df
        elif operation == "remove_outliers":
            numeric_cols = df.select_dtypes(include=["number"]).columns
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            return df
        else:
            # Unknown operation - return unchanged
            return df
