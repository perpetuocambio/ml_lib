"""LoRA-related queries.

Read-only queries for LoRA data retrieval.
Following CQRS: these are separate from Commands (write operations).
"""

from dataclasses import dataclass
from ml_lib.diffusion.application.queries.base import (
    IQuery,
    IQueryHandler,
    QueryResult,
)
from ml_lib.diffusion.domain.services.lora_recommendation_service import (
    LoRARecommendationService,
)


@dataclass(frozen=True)
class GetAllLoRAsQuery(IQuery):
    """
    Query to get all available LoRAs.

    Use case: User wants to browse all LoRAs.
    """

    pass  # No parameters needed


@dataclass(frozen=True)
class GetLoRAsByBaseModelQuery(IQuery):
    """
    Query to get LoRAs compatible with a base model.

    Use case: User wants to see LoRAs for a specific model architecture.
    """

    base_model: str


@dataclass(frozen=True)
class SearchLoRAsByPromptQuery(IQuery):
    """
    Query to search LoRAs by prompt keywords.

    Use case: User wants to find LoRAs matching their prompt.
    """

    prompt: str
    base_model: str


class GetAllLoRAsHandler(IQueryHandler[GetAllLoRAsQuery]):
    """
    Handler for GetAllLoRAsQuery.

    Returns all available LoRAs from repository.
    """

    def __init__(self, service: LoRARecommendationService):
        """
        Initialize handler.

        Args:
            service: LoRA recommendation service
        """
        self.service = service

    def handle(self, query: GetAllLoRAsQuery) -> QueryResult:
        """
        Handle get all LoRAs query.

        Args:
            query: GetAllLoRAsQuery

        Returns:
            QueryResult with list of all LoRAs
        """
        # Get all LoRAs from repository via service
        loras = self.service.repository.get_all_loras()

        return QueryResult.success(
            data=loras,
            metadata={
                "count": len(loras),
                "query_type": "get_all",
            },
        )


class GetLoRAsByBaseModelHandler(IQueryHandler[GetLoRAsByBaseModelQuery]):
    """
    Handler for GetLoRAsByBaseModelQuery.

    Returns LoRAs filtered by base model compatibility.
    """

    def __init__(self, service: LoRARecommendationService):
        """
        Initialize handler.

        Args:
            service: LoRA recommendation service
        """
        self.service = service

    def handle(self, query: GetLoRAsByBaseModelQuery) -> QueryResult:
        """
        Handle get LoRAs by base model query.

        Args:
            query: GetLoRAsByBaseModelQuery

        Returns:
            QueryResult with filtered LoRAs
        """
        # Get compatible LoRAs from repository
        loras = self.service.repository.get_loras_by_base_model(query.base_model)

        return QueryResult.success(
            data=loras,
            metadata={
                "count": len(loras),
                "base_model": query.base_model,
                "query_type": "filter_by_model",
            },
        )


class SearchLoRAsByPromptHandler(IQueryHandler[SearchLoRAsByPromptQuery]):
    """
    Handler for SearchLoRAsByPromptQuery.

    Searches LoRAs that match prompt trigger words.
    """

    def __init__(self, service: LoRARecommendationService):
        """
        Initialize handler.

        Args:
            service: LoRA recommendation service
        """
        self.service = service

    def handle(self, query: SearchLoRAsByPromptQuery) -> QueryResult:
        """
        Handle search LoRAs by prompt query.

        Args:
            query: SearchLoRAsByPromptQuery

        Returns:
            QueryResult with matching LoRA recommendations
        """
        # Use service method that finds LoRAs by trigger words
        recommendations = self.service.get_recommendations_by_trigger_words(
            prompt=query.prompt,
            base_model=query.base_model,
        )

        # Extract just the LoRAs from recommendations
        loras = [rec.lora for rec in recommendations]

        return QueryResult.success(
            data=loras,
            metadata={
                "count": len(loras),
                "prompt": query.prompt,
                "base_model": query.base_model,
                "query_type": "search_by_prompt",
            },
        )
