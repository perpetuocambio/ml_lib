"""Search source enumeration for result tracking."""

from enum import Enum


class SearchSource(Enum):
    """Sources of search results."""

    DUCKDUCKGO = "duckduckgo"
    BING = "bing"
    LOCAL_DOCUMENTS = "local_documents"
    PROJECT_DOCUMENTS = "project_documents"
    RAG_VECTOR_STORE = "rag_vector_store"
    CACHED_RESULTS = "cached_results"
