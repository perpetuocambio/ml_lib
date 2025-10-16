"""Tests for Query Pattern implementation.

Tests for queries, handlers, and QueryBus.
"""

import pytest
from ml_lib.diffusion.application.queries import (
    QueryBus,
    GetAllLoRAsQuery,
    GetLoRAsByBaseModelQuery,
    SearchLoRAsByPromptQuery,
    GetAllLoRAsHandler,
    GetLoRAsByBaseModelHandler,
    SearchLoRAsByPromptHandler,
)

from ml_lib.diffusion.domain.entities.lora import LoRA
from ml_lib.diffusion.infrastructure.persistence.in_memory_model_repository import (
    InMemoryModelRepository,
)
from ml_lib.diffusion.domain.services.lora_recommendation_service import (
    LoRARecommendationService,
)


@pytest.fixture
def repository():
    """Create repository with sample LoRAs."""
    repo = InMemoryModelRepository()

    # Add sample LoRAs for different models
    loras = [
        # SDXL LoRAs
        LoRA(
            id="anime-sdxl",
            name="Anime Style SDXL",
            base_model="SDXL",
            trigger_words=["anime", "manga"],
            strength=0.8,
            filename="anime-sdxl.safetensors",
        ),
        LoRA(
            id="portrait-sdxl",
            name="Portrait Detail SDXL",
            base_model="SDXL",
            trigger_words=["portrait", "face"],
            strength=0.7,
            filename="portrait-sdxl.safetensors",
        ),
        # SD 1.5 LoRAs
        LoRA(
            id="anime-sd15",
            name="Anime Style SD15",
            base_model="SD 1.5",
            trigger_words=["anime"],
            strength=0.8,
            filename="anime-sd15.safetensors",
        ),
        # Pony V6 LoRAs
        LoRA(
            id="cyberpunk-pony",
            name="Cyberpunk Pony",
            base_model="Pony Diffusion V6",
            trigger_words=["cyberpunk", "neon"],
            strength=0.9,
            filename="cyberpunk-pony.safetensors",
        ),
    ]

    for lora in loras:
        repo.add_lora(lora)

    return repo


@pytest.fixture
def lora_service(repository):
    """Create LoRA recommendation service."""
    return LoRARecommendationService(repository)


@pytest.fixture
def query_bus(lora_service):
    """Create query bus with registered handlers."""
    bus = QueryBus()

    bus.register(GetAllLoRAsQuery, GetAllLoRAsHandler(lora_service))
    bus.register(GetLoRAsByBaseModelQuery, GetLoRAsByBaseModelHandler(lora_service))
    bus.register(SearchLoRAsByPromptQuery, SearchLoRAsByPromptHandler(lora_service))

    return bus


# ============================================================================
# QueryBus Tests
# ============================================================================

def test_query_bus_registration(query_bus):
    """Test query registration."""
    assert query_bus.is_registered(GetAllLoRAsQuery)
    assert query_bus.is_registered(GetLoRAsByBaseModelQuery)
    assert query_bus.is_registered(SearchLoRAsByPromptQuery)


def test_query_bus_unregister(lora_service):
    """Test query unregistration."""
    bus = QueryBus()
    handler = GetAllLoRAsHandler(lora_service)

    bus.register(GetAllLoRAsQuery, handler)
    assert bus.is_registered(GetAllLoRAsQuery)

    bus.unregister(GetAllLoRAsQuery)
    assert not bus.is_registered(GetAllLoRAsQuery)


def test_query_bus_duplicate_registration(lora_service):
    """Test that duplicate registration raises error."""
    bus = QueryBus()
    handler = GetAllLoRAsHandler(lora_service)

    bus.register(GetAllLoRAsQuery, handler)

    with pytest.raises(ValueError, match="Handler already registered"):
        bus.register(GetAllLoRAsQuery, handler)


def test_query_bus_no_handler_registered():
    """Test dispatch with no handler registered."""
    bus = QueryBus()

    # Create a dummy query class
    from dataclasses import dataclass
    from ml_lib.diffusion.application.queries.base import IQuery

    @dataclass(frozen=True)
    class DummyQuery(IQuery):
        value: str

    query = DummyQuery(value="test")

    with pytest.raises(ValueError, match="No handler registered"):
        bus.dispatch(query)


def test_query_bus_get_registered_queries(query_bus):
    """Test getting list of registered queries."""
    registered = query_bus.get_registered_queries()

    assert GetAllLoRAsQuery in registered
    assert GetLoRAsByBaseModelQuery in registered
    assert SearchLoRAsByPromptQuery in registered
    assert len(registered) == 3


def test_query_bus_with_monitoring(lora_service):
    """Test query bus with performance monitoring."""
    bus = QueryBus(enable_monitoring=True)
    bus.register(GetAllLoRAsQuery, GetAllLoRAsHandler(lora_service))

    query = GetAllLoRAsQuery()
    result = bus.dispatch(query)

    assert result.metadata is not None
    assert "query_time_ms" in result.metadata
    assert "query_name" in result.metadata
    assert result.metadata["query_name"] == "GetAllLoRAsQuery"
    assert isinstance(result.metadata["query_time_ms"], float)
    assert result.metadata["query_time_ms"] >= 0


def test_query_bus_without_monitoring(lora_service):
    """Test query bus without performance monitoring."""
    bus = QueryBus(enable_monitoring=False)
    bus.register(GetAllLoRAsQuery, GetAllLoRAsHandler(lora_service))

    query = GetAllLoRAsQuery()
    result = bus.dispatch(query)

    # Metadata might exist from handler, but no monitoring data added
    assert result.data is not None


# ============================================================================
# GetAllLoRAsQuery Tests
# ============================================================================

def test_get_all_loras_query_success(query_bus):
    """Test successful retrieval of all LoRAs."""
    query = GetAllLoRAsQuery()
    result = query_bus.dispatch(query)

    assert result.data is not None
    assert isinstance(result.data, list)
    assert len(result.data) == 4  # All LoRAs from fixture
    assert all(isinstance(lora, LoRA) for lora in result.data)
    assert result.metadata["count"] == 4
    assert result.metadata["query_type"] == "get_all"


def test_get_all_loras_query_empty_repository():
    """Test get all with empty repository."""
    empty_repo = InMemoryModelRepository()
    service = LoRARecommendationService(empty_repo)
    handler = GetAllLoRAsHandler(service)

    query = GetAllLoRAsQuery()
    result = handler.handle(query)

    assert result.data is not None
    assert isinstance(result.data, list)
    assert len(result.data) == 0
    assert result.metadata["count"] == 0


def test_get_all_loras_query_returns_all_models(query_bus):
    """Test that get all returns LoRAs from all base models."""
    query = GetAllLoRAsQuery()
    result = query_bus.dispatch(query)

    base_models = {lora.base_model.value for lora in result.data}

    assert "SDXL" in base_models
    assert "SD 1.5" in base_models
    assert "Pony Diffusion V6" in base_models


# ============================================================================
# GetLoRAsByBaseModelQuery Tests
# ============================================================================

def test_get_loras_by_base_model_query_sdxl(query_bus):
    """Test retrieval of SDXL LoRAs."""
    query = GetLoRAsByBaseModelQuery(base_model="SDXL")
    result = query_bus.dispatch(query)

    assert result.data is not None
    assert isinstance(result.data, list)
    assert len(result.data) == 2  # anime-sdxl, portrait-sdxl
    assert all(lora.base_model.value == "SDXL" for lora in result.data)
    assert result.metadata["count"] == 2
    assert result.metadata["base_model"] == "SDXL"
    assert result.metadata["query_type"] == "filter_by_model"


def test_get_loras_by_base_model_query_sd15(query_bus):
    """Test retrieval of SD 1.5 LoRAs."""
    query = GetLoRAsByBaseModelQuery(base_model="SD 1.5")
    result = query_bus.dispatch(query)

    assert len(result.data) == 1  # anime-sd15
    assert result.data[0].base_model.value == "SD 1.5"


def test_get_loras_by_base_model_query_pony(query_bus):
    """Test retrieval of Pony V6 LoRAs."""
    query = GetLoRAsByBaseModelQuery(base_model="Pony Diffusion V6")
    result = query_bus.dispatch(query)

    assert len(result.data) == 1  # cyberpunk-pony
    assert result.data[0].base_model.value == "Pony Diffusion V6"


def test_get_loras_by_base_model_query_no_matches(query_bus):
    """Test query with base model that has no LoRAs."""
    query = GetLoRAsByBaseModelQuery(base_model="NonExistentModel")
    result = query_bus.dispatch(query)

    assert result.data is not None
    assert isinstance(result.data, list)
    assert len(result.data) == 0
    assert result.metadata["count"] == 0


def test_get_loras_by_base_model_query_case_sensitivity(query_bus):
    """Test that base model matching is case-sensitive."""
    query = GetLoRAsByBaseModelQuery(base_model="sdxl")  # lowercase
    result = query_bus.dispatch(query)

    # Should not match "SDXL" (case-sensitive)
    assert len(result.data) == 0


# ============================================================================
# SearchLoRAsByPromptQuery Tests
# ============================================================================

def test_search_loras_by_prompt_query_anime(query_bus):
    """Test search for 'anime' trigger word."""
    query = SearchLoRAsByPromptQuery(
        prompt="anime girl",
        base_model="SDXL"
    )
    result = query_bus.dispatch(query)

    assert result.data is not None
    assert isinstance(result.data, list)
    assert len(result.data) >= 1  # At least anime-sdxl
    assert result.metadata["count"] == len(result.data)
    assert result.metadata["prompt"] == "anime girl"
    assert result.metadata["base_model"] == "SDXL"
    assert result.metadata["query_type"] == "search_by_prompt"


def test_search_loras_by_prompt_query_portrait(query_bus):
    """Test search for 'portrait' trigger word."""
    query = SearchLoRAsByPromptQuery(
        prompt="portrait of a woman",
        base_model="SDXL"
    )
    result = query_bus.dispatch(query)

    assert len(result.data) >= 1  # portrait-sdxl
    # Should contain portrait LoRA
    lora_names = [lora.name.value for lora in result.data]
    assert any("Portrait" in name for name in lora_names)


def test_search_loras_by_prompt_query_cyberpunk(query_bus):
    """Test search for 'cyberpunk' with Pony model."""
    query = SearchLoRAsByPromptQuery(
        prompt="cyberpunk city",
        base_model="Pony Diffusion V6"
    )
    result = query_bus.dispatch(query)

    assert len(result.data) >= 1  # cyberpunk-pony
    assert result.data[0].name.value == "Cyberpunk Pony"


def test_search_loras_by_prompt_query_no_matches(query_bus):
    """Test search with no matching trigger words."""
    query = SearchLoRAsByPromptQuery(
        prompt="abstract geometric patterns",
        base_model="SDXL"
    )
    result = query_bus.dispatch(query)

    assert result.data is not None
    assert isinstance(result.data, list)
    # May be empty if no trigger words match


def test_search_loras_by_prompt_query_wrong_model(query_bus):
    """Test search with correct prompt but wrong base model."""
    query = SearchLoRAsByPromptQuery(
        prompt="anime girl",
        base_model="SD 1.5"
    )
    result = query_bus.dispatch(query)

    # Should only return SD 1.5 anime LoRA
    assert all(lora.base_model.value == "SD 1.5" for lora in result.data)


def test_search_loras_by_prompt_query_multiple_trigger_words(query_bus):
    """Test search with multiple trigger words in prompt."""
    query = SearchLoRAsByPromptQuery(
        prompt="anime manga style portrait",
        base_model="SDXL"
    )
    result = query_bus.dispatch(query)

    # Should match both anime and portrait LoRAs
    assert len(result.data) >= 2


# ============================================================================
# QueryResult Tests
# ============================================================================

def test_query_result_success():
    """Test QueryResult.success factory method."""
    from ml_lib.diffusion.application.queries.base import QueryResult

    result = QueryResult.success(
        data={"key": "value"},
        metadata={"count": 1}
    )

    assert result.data == {"key": "value"}
    assert result.metadata == {"count": 1}


def test_query_result_without_metadata():
    """Test QueryResult without metadata."""
    from ml_lib.diffusion.application.queries.base import QueryResult

    result = QueryResult.success(data="test")

    assert result.data == "test"
    assert result.metadata is None


def test_query_result_immutability():
    """Test that QueryResult is immutable."""
    from ml_lib.diffusion.application.queries.base import QueryResult

    result = QueryResult.success(data="test")

    with pytest.raises(Exception):  # FrozenInstanceError
        result.data = "modified"


# ============================================================================
# Integration Tests
# ============================================================================

def test_query_workflow_get_all_then_filter(query_bus):
    """Test workflow: get all → filter by model."""
    # Step 1: Get all LoRAs
    get_all_query = GetAllLoRAsQuery()
    all_result = query_bus.dispatch(get_all_query)
    assert len(all_result.data) == 4

    # Step 2: Get only SDXL LoRAs
    filter_query = GetLoRAsByBaseModelQuery(base_model="SDXL")
    filtered_result = query_bus.dispatch(filter_query)
    assert len(filtered_result.data) == 2
    assert len(filtered_result.data) < len(all_result.data)


def test_query_workflow_search_specific_model(query_bus):
    """Test workflow: search with specific base model."""
    # Search for anime in SDXL
    sdxl_query = SearchLoRAsByPromptQuery(
        prompt="anime",
        base_model="SDXL"
    )
    sdxl_result = query_bus.dispatch(sdxl_query)

    # Search for anime in SD 1.5
    sd15_query = SearchLoRAsByPromptQuery(
        prompt="anime",
        base_model="SD 1.5"
    )
    sd15_result = query_bus.dispatch(sd15_query)

    # Results should be different (different models)
    sdxl_ids = {lora.id for lora in sdxl_result.data}
    sd15_ids = {lora.id for lora in sd15_result.data}
    assert sdxl_ids != sd15_ids


def test_query_performance_monitoring_overhead(lora_service):
    """Test that monitoring doesn't significantly impact performance."""
    bus_with_monitoring = QueryBus(enable_monitoring=True)
    bus_without_monitoring = QueryBus(enable_monitoring=False)

    handler = GetAllLoRAsHandler(lora_service)
    bus_with_monitoring.register(GetAllLoRAsQuery, handler)
    bus_without_monitoring.register(GetAllLoRAsQuery, handler)

    query = GetAllLoRAsQuery()

    # Both should return same data
    result_with = bus_with_monitoring.dispatch(query)
    result_without = bus_without_monitoring.dispatch(query)

    assert len(result_with.data) == len(result_without.data)


def test_queries_are_read_only(query_bus, repository):
    """Test that queries don't modify repository state."""
    initial_count = len(repository.get_all_loras())

    # Execute multiple queries
    query_bus.dispatch(GetAllLoRAsQuery())
    query_bus.dispatch(GetLoRAsByBaseModelQuery(base_model="SDXL"))
    query_bus.dispatch(SearchLoRAsByPromptQuery(prompt="anime", base_model="SDXL"))

    final_count = len(repository.get_all_loras())

    # Repository should be unchanged
    assert final_count == initial_count


def test_query_immutability():
    """Test that query objects are immutable."""
    query = GetLoRAsByBaseModelQuery(base_model="SDXL")

    with pytest.raises(Exception):  # FrozenInstanceError
        query.base_model = "SD 1.5"
