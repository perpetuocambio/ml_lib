"""
Proveedor LLM para Ollama (local).
"""

import base64
import json
import logging
from pathlib import Path

from infrastructure.communication.http.interfaces.http_client_interface import (
    HttpClientInterface,
)
from infrastructure.communication.http.services.services import RequestsHttpClient
from infrastructure.config.providers.llm_provider_config import LLMProviderConfig
from infrastructure.persistence.storage.interfaces.storage_interface import (
    StorageInterface,
)
from infrastructure.providers.llm.entities.document_prompt import DocumentPrompt
from infrastructure.providers.llm.entities.llm_prompt import LLMPrompt
from infrastructure.providers.llm.entities.llm_response import LLMResponse
from infrastructure.providers.llm.entities.ollama_request import (
    OllamaOptions,
    OllamaRequest,
)
from infrastructure.providers.llm.entities.ollama_response import (
    OllamaResponse,
)
from infrastructure.providers.llm.interfaces.llm_provider import LLMProvider
from infrastructure.providers.llm.services.ollama_response_parser import (
    OllamaResponseParser,
)

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """
    Proveedor LLM para Ollama.

    Soporta modelos de texto y visión disponibles localmente via Ollama.
    """

    def __init__(
        self,
        configuration: LLMProviderConfig,
        http_client: HttpClientInterface | None = None,
        storage_client: StorageInterface | None = None,
    ):
        """
        Inicializa el proveedor Ollama.

        Args:
            configuration: Configuración del proveedor
        """
        super().__init__(configuration)
        self.api_url = configuration.api_endpoint or "http://localhost:11434"
        # If no http_client provided, lazily import the default RequestsHttpClient
        if http_client is None:
            self.http_client = RequestsHttpClient()
        else:
            self.http_client = http_client

        # Optional storage client for reading image bytes (e.g. remote/local adapters)
        self.storage_client = storage_client

        # Parser for API responses
        self._parser = OllamaResponseParser()

    def generate_response(self, prompt: LLMPrompt | DocumentPrompt) -> LLMResponse:
        """
        Genera una respuesta usando Ollama.

        Args:
            prompt: Prompt estructurado para el LLM

        Returns:
            LLMResponse: Respuesta estructurada del LLM
        """
        try:
            # Preparar request para Ollama
            request_data = self._prepare_request(prompt)

            # Llamar API de Ollama usando RequestsHttpClient
            response_data = self._call_ollama_api(request_data)

            # Procesar respuesta
            return self._process_response(response_data)

        except json.JSONDecodeError as exc:
            logger.error("Error decoding JSON from Ollama: %s", exc)
            return LLMResponse(
                content=f"Error: {str(exc)}",
                usage_tokens=0,
                model_name=self.configuration.model_name,
                confidence_score=0.0,
            )
        except Exception as exc:
            # HTTP client or other runtime errors
            logger.error("HTTP error calling Ollama: %s", exc)
            return LLMResponse(
                content=f"HTTP error: {str(exc)}",
                usage_tokens=0,
                model_name=self.configuration.model_name,
                confidence_score=0.0,
            )

    def is_available(self) -> bool:
        """
        Verifica si Ollama está disponible.

        Returns:
            bool: True si Ollama puede ser utilizado
        """
        try:
            resp = self.http_client.get(f"{self.api_url}/api/tags")
            return resp.status_code == 200
        except Exception as exc:
            # Any HTTP client error -> not available
            logger.debug("Ollama availability check failed: %s", exc)
            return False

    def get_supported_models(self) -> list[str]:
        """
        Obtiene la lista de modelos disponibles en Ollama.

        Returns:
            list[str]: Lista de nombres de modelos disponibles
        """
        try:
            resp = self.http_client.get(f"{self.api_url}/api/tags")
            models_response = self._parser.parse_models_response(resp.content)
            return models_response.get_model_names()
        except json.JSONDecodeError as exc:
            logger.error("Error decoding JSON when fetching Ollama models: %s", exc)
            return []
        except Exception as exc:
            logger.error("HTTP error fetching Ollama models: %s", exc)
            return []

    def _prepare_request(self, prompt: LLMPrompt | DocumentPrompt) -> OllamaRequest:
        """
        Prepara el request para la API de Ollama.

        Args:
            prompt: Prompt estructurado

        Returns:
            OllamaRequest: Request data para Ollama
        """
        # Handle different prompt types
        if isinstance(prompt, DocumentPrompt):
            temperature = prompt.temperature
            context_size = prompt.context_window_size
        else:  # LLMPrompt
            temperature = prompt.temperature
            context_size = getattr(prompt, "max_tokens", 4000)

        options = OllamaOptions(temperature=temperature, num_ctx=context_size)
        request = OllamaRequest(
            model=self.configuration.model_name,
            prompt=prompt.content,
            stream=False,
            options=options,
        )

        # Add images if present (for vision models)
        if hasattr(prompt, "images") and prompt.images:
            request.images = []
            for image_path in prompt.images:
                if isinstance(image_path, str | Path):
                    p = Path(image_path)
                    img_bytes: bytes | None = None

                    # Prefer storage client if provided
                    if getattr(self, "storage_client", None) is not None:
                        try:
                            img_bytes = self.storage_client.read_file(str(p))
                        except (FileNotFoundError, OSError) as exc:
                            logger.debug("storage_client failed to read %s: %s", p, exc)
                            img_bytes = None

                    # Fallback to local filesystem access
                    if img_bytes is None and p.exists():
                        with open(p, "rb") as img_file:
                            img_bytes = img_file.read()

                    if img_bytes:
                        img_data = base64.b64encode(img_bytes).decode("utf-8")
                        if request.images is None:
                            request.images = []
                        request.images.append(img_data)

        return request

    def _call_ollama_api(self, request_data: OllamaRequest) -> OllamaResponse:
        """
        Llama a la API de Ollama.

        Args:
            request_data: Datos del request

        Returns:
            OllamaResponse: Respuesta de Ollama
        """
        # Convert to request data and serialize directly
        typed_data = request_data.to_request_data()
        json_payload = json.dumps(
            {
                "model": typed_data.model,
                "prompt": typed_data.prompt,
                "stream": typed_data.stream,
                "options": {
                    "temperature": typed_data.temperature,
                    "num_ctx": typed_data.num_ctx,
                },
                "images": typed_data.images,
            }
        )

        resp = self.http_client.post(
            f"{self.api_url}/api/generate",
            data=json_payload,
            headers={"Content-Type": "application/json"},
        )
        return self._parser.parse_generate_response(resp.content)

    def _process_response(self, response_data: OllamaResponse) -> LLMResponse:
        """
        Procesa la respuesta de Ollama.

        Args:
            response_data: Respuesta estructurada de Ollama

        Returns:
            LLMResponse: Respuesta estructurada
        """
        content = response_data.response
        eval_count = response_data.eval_count

        # Estimate confidence based on response completeness
        confidence = 1.0 if response_data.done else 0.5

        return LLMResponse(
            content=content,
            usage_tokens=eval_count,
            model_name=self.configuration.model_name,
            confidence_score=confidence,
        )
