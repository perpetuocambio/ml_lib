"""Pandas-based dataset import implementation."""

import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
from infrastructure.communication.http.client.http_client import HttpClient
from infrastructure.communication.http.headers.http_header import HttpHeader
from infrastructure.communication.http.headers.http_headers_collection import (
    HttpHeadersCollection,
)
from infrastructure.data.entities.dataset_import_configuration import (
    DatasetImportConfiguration,
)
from infrastructure.data.entities.imported_dataset import ImportedDataset
from infrastructure.data.enums.dataset_format import DatasetFormat
from infrastructure.serialization.protocol_serializer import ProtocolSerializer


class PandasDatasetImporter:
    """Pandas-based implementation for dataset import."""

    def __init__(self):
        self.http_client = HttpClient()

    def import_from_file(
        self, file_path: Path, config: DatasetImportConfiguration
    ) -> ImportedDataset:
        """Import dataset from file using pandas."""
        try:
            start_time = datetime.now()

            # Read data based on format
            df = self._read_dataframe(file_path, config)

            # Apply data cleaning if requested
            if config.clean_data:
                df = self._clean_dataframe(df)

            # Apply row limits
            if config.max_rows:
                df = df.head(config.max_rows)

            # Get sample data
            sample_data = self._get_sample_from_dataframe(df, sample_size=5)

            # Get file stats
            file_size = file_path.stat().st_size

            processing_time = (datetime.now() - start_time).total_seconds()

            return ImportedDataset(
                success=True,
                dataset_id="",  # Will be set by application service
                project_id="",  # Will be set by application service
                source_description="",  # Will be set by application service
                source_type=config.source_type,
                format=config.format,
                file_path=str(file_path),
                row_count=len(df),
                column_count=len(df.columns),
                column_names=list(df.columns),
                file_size_bytes=file_size,
                import_timestamp=datetime.now(),
                processing_time_seconds=processing_time,
                sample_data=sample_data,
            )

        except Exception as e:
            return ImportedDataset(
                success=False,
                dataset_id="",
                project_id="",
                source_description="",
                source_type=config.source_type,
                format=config.format,
                file_path=str(file_path) if file_path else "",
                row_count=0,
                column_count=0,
                column_names=[],
                file_size_bytes=0,
                import_timestamp=datetime.now(),
                processing_time_seconds=0,
                error_message=str(e),
            )

    def import_from_url(
        self, url: str, config: DatasetImportConfiguration
    ) -> ImportedDataset:
        """Import dataset from URL."""
        try:
            # Download file
            headers = HttpHeadersCollection()
            headers.add_header(
                HttpHeader.user_agent("PyIntelCivil-DatasetImporter/1.0")
            )

            response = self.http_client.get(url, headers)
            response.raise_for_status()

            # Save to temporary file
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f".{config.format.value.lower()}"
            ) as temp_file:
                temp_file.write(response.content)
                temp_path = Path(temp_file.name)

            # Import from the temporary file
            result = self.import_from_file(temp_path, config)

            # Update the file path in result to point to temp file for later moving
            if result.success:
                result.file_path = str(temp_path)
            else:
                # Clean up temp file on failure
                temp_path.unlink()

            return result

        except Exception as e:
            return ImportedDataset(
                success=False,
                dataset_id="",
                project_id="",
                source_description="",
                source_type=config.source_type,
                format=config.format,
                file_path="",
                row_count=0,
                column_count=0,
                column_names=[],
                file_size_bytes=0,
                import_timestamp=datetime.now(),
                processing_time_seconds=0,
                error_message=str(e),
            )

    def validate_format(self, file_path: Path, expected_format: str) -> bool:
        """Validate dataset format."""
        try:
            expected_fmt = DatasetFormat(expected_format.upper())
            config = DatasetImportConfiguration(
                source_type=None,  # Not needed for validation
                format=expected_fmt,
            )

            # Try to read a small sample
            df = self._read_dataframe(file_path, config, nrows=1)
            return len(df) >= 0  # If we can read it, format is valid

        except Exception:
            return False

    def get_sample_data(
        self, file_path: Path, config: DatasetImportConfiguration, sample_size: int = 5
    ) -> list[dict]:
        """Get sample data from dataset."""
        try:
            df = self._read_dataframe(file_path, config, nrows=sample_size)
            return self._get_sample_from_dataframe(df, sample_size)
        except Exception:
            return []

    def _read_dataframe(
        self,
        file_path: Path,
        config: DatasetImportConfiguration,
        nrows: int | None = None,
    ) -> pd.DataFrame:
        """Read dataframe based on format."""
        read_kwargs = {}

        if nrows:
            read_kwargs["nrows"] = nrows
        elif config.max_rows:
            read_kwargs["nrows"] = config.max_rows

        if config.skip_rows > 0:
            read_kwargs["skiprows"] = config.skip_rows

        if config.format == DatasetFormat.CSV:
            read_kwargs["delimiter"] = config.delimiter or ","
            read_kwargs["encoding"] = config.encoding
            if not config.has_header:
                read_kwargs["header"] = None
            return pd.read_csv(file_path, **read_kwargs)

        elif config.format == DatasetFormat.TSV:
            read_kwargs["delimiter"] = "\t"
            read_kwargs["encoding"] = config.encoding
            if not config.has_header:
                read_kwargs["header"] = None
            return pd.read_csv(file_path, **read_kwargs)

        elif config.format == DatasetFormat.JSON:
            return pd.read_json(file_path, encoding=config.encoding)

        elif config.format == DatasetFormat.EXCEL:
            if not config.has_header:
                read_kwargs["header"] = None
            return pd.read_excel(file_path, **read_kwargs)

        elif config.format == DatasetFormat.PARQUET:
            return pd.read_parquet(file_path)

        else:
            raise ValueError(f"Unsupported format: {config.format}")

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean dataframe by removing empty rows and standardizing column names."""
        # Remove completely empty rows
        df = df.dropna(how="all")

        # Clean column names
        df.columns = df.columns.astype(str)
        df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]

        return df

    def _get_sample_from_dataframe(
        self, df: pd.DataFrame, sample_size: int
    ) -> list[dict]:
        """Get sample data from dataframe."""
        sample_df = df.head(sample_size)

        # Convert to protocol-safe dict format using ProtocolSerializer
        sample_data = []
        for _, row in sample_df.iterrows():
            row_mapping = {}
            for col, value in row.items():
                if pd.isna(value):
                    row_mapping[col] = None
                elif isinstance(value, pd.Timestamp | datetime):
                    row_mapping[col] = value.isoformat()
                else:
                    row_mapping[col] = str(value)

            # Use ProtocolSerializer for boundary crossing
            row_dict = ProtocolSerializer.serialize_mapping_data(row_mapping)
            sample_data.append(row_dict)

        return sample_data
