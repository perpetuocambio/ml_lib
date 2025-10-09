"""Infrastructure extractor entities - clean document type mappings."""

from infrastructure.data.extractors.entities.document_type_mappings import (
    DocumentTypeMappings,
)
from infrastructure.data.extractors.entities.extension_mapping import (
    ExtensionMapping,
)
from infrastructure.data.extractors.entities.extension_mapping_collection import (
    ExtensionMappingCollection,
)
from infrastructure.data.extractors.entities.mime_type_mapping import (
    MimeTypeMapping,
)
from infrastructure.data.extractors.entities.mime_type_mapping_collection import (
    MimeTypeMappingCollection,
)

__all__ = [
    "DocumentTypeMappings",
    "ExtensionMapping",
    "ExtensionMappingCollection",
    "MimeTypeMapping",
    "MimeTypeMappingCollection",
]
