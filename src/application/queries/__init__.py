"""Application Queries - CQRS Query Pattern implementation.

Queries represent read operations in CQRS architecture.
They retrieve data without modifying state.

CQRS Separation:
- Commands: Write operations (mutations) → application/commands/
- Queries: Read operations (no side effects) → application/queries/

Query Pattern benefits:
- Optimized read models separate from write models
- Cacheable queries
- Read-only operations are explicit
- Performance optimization for reads
- Query-specific projections
"""

from ml_lib.diffusion.application.queries.base import (
    IQuery,
    IQueryHandler,
    IQueryBus,
    QueryResult,
)
from ml_lib.diffusion.application.queries.bus import QueryBus
from ml_lib.diffusion.application.queries.lora_queries import (
    GetAllLoRAsQuery,
    GetLoRAsByBaseModelQuery,
    SearchLoRAsByPromptQuery,
    GetAllLoRAsHandler,
    GetLoRAsByBaseModelHandler,
    SearchLoRAsByPromptHandler,
)

__all__ = [
    # Base interfaces
    "IQuery",
    "IQueryHandler",
    "IQueryBus",
    "QueryResult",
    # Implementations
    "QueryBus",
    # LoRA queries
    "GetAllLoRAsQuery",
    "GetLoRAsByBaseModelQuery",
    "SearchLoRAsByPromptQuery",
    # LoRA handlers
    "GetAllLoRAsHandler",
    "GetLoRAsByBaseModelHandler",
    "SearchLoRAsByPromptHandler",
]
