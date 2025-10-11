"""
LLM Infrastructure - Proveedores de servicios LLM agnósticos y gestión de prompts.

Este módulo provee una interfaz unificada para trabajar con diferentes proveedores
LLM (Ollama, OpenAI, Anthropic, etc.) de manera agnóstica.
"""

# Configuration
from ml_lib.llm.config.llm_provider_config import LLMProviderConfig, LLMProviderType

# Core Entities
from ml_lib.llm.entities.llm_prompt import LLMPrompt
from ml_lib.llm.entities.llm_response import LLMResponse
from ml_lib.llm.entities.prompt_type import PromptType
from ml_lib.llm.entities.prompt_category import PromptCategory
from ml_lib.llm.entities.document_prompt import DocumentPrompt
from ml_lib.llm.entities.proposal_template_fields import ProposalTemplateFields
from ml_lib.llm.entities.template_categories import TemplateCategoriesListing

# Interfaces
from ml_lib.llm.interfaces.llm_provider import LLMProvider
from ml_lib.llm.clients.llm_client import LLMClient

# Embeddings
from ml_lib.llm.embeddings.entities.embedding_vector import EmbeddingVector
from ml_lib.llm.embeddings.entities.health_check_result import EmbeddingHealthCheckResult
from ml_lib.llm.embeddings.entities.model_info import ModelInfo

# Messages
from ml_lib.llm.messages.base_message import BaseMessage
from ml_lib.llm.messages.ai_message import AIMessage

# Services
# Note: PromptManager requires PromptsConfig from infrastructure layer
# Only import if needed by client code
try:
    from ml_lib.llm.services.prompt_manager import PromptManager
    _prompt_manager_available = True
except ImportError:
    PromptManager = None
    _prompt_manager_available = False

__all__ = [
    # Configuration
    "LLMProviderConfig",
    "LLMProviderType",
    # Core
    "LLMProvider",
    "LLMClient",
    "LLMPrompt",
    "LLMResponse",
    "PromptType",
    "PromptCategory",
    "DocumentPrompt",
    "ProposalTemplateFields",
    "TemplateCategoriesListing",
    # Embeddings
    "EmbeddingVector",
    "EmbeddingHealthCheckResult",
    "ModelInfo",
    # Messages
    "BaseMessage",
    "AIMessage",
]

# Add PromptManager to __all__ only if available
if _prompt_manager_available:
    __all__.append("PromptManager")
