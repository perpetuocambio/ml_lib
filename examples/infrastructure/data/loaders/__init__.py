"""
Data loaders infrastructure module.
Handles configuration loading and dataset import.
"""

from infrastructure.data.loaders.services.typed_yaml_loader import TypedYamlLoader

__all__ = [
    "TypedYamlLoader",
]

# Optional heavy imports - only import if needed
try:
    import pandas as pd  # noqa: F401
    from infrastructure.data.importers.pandas_dataset_importer import (
        PandasDatasetImporter,
    )

    __all__.append("PandasDatasetImporter")
except ImportError:
    # pandas not available, skip heavy dataset importer
    pass
