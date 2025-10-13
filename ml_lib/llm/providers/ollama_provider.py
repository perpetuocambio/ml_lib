"""
Proveedor LLM para Ollama (local).
"""

import base64
import json
import logging
import subprocess
import time
import atexit
from pathlib import Path
from typing import Protocol, Optional

from ml_lib.llm.config.llm_provider_config import LLMProviderConfig
from ml_lib.llm.entities.document_prompt import DocumentPrompt
from ml_lib.llm.entities.llm_prompt import LLMPrompt
from ml_lib.llm.entities.llm_response import LLMResponse
from ml_lib.llm.entities.ollama_request import (
    OllamaOptions,
    OllamaRequest,
)
from ml_lib.llm.entities.ollama_response import (
    OllamaResponse,
)
from ml_lib.llm.interfaces.llm_provider import LLMProvider
from ml_lib.llm.services.ollama_response_parser import (
    OllamaResponseParser,
)


# Protocol para HTTP client (duck typing) - el cliente puede inyectar su implementación
class HttpClientInterface(Protocol):
    """Interface para clientes HTTP."""

    def get(self, url: str, **kwargs):
        """GET request."""
        ...

    def post(self, url: str, data: str = None, headers: dict = None, **kwargs):
        """POST request."""
        ...


# Protocol para storage (duck typing) - el cliente puede inyectar su implementación
class StorageInterface(Protocol):
    """Interface para almacenamiento."""

    def read_file(self, path: str) -> bytes:
        """Lee un archivo."""
        ...

logger = logging.getLogger(__name__)


class OllamaServerContext:
    """
    Context manager for Ollama server lifecycle.

    Automatically starts server on enter and stops on exit for memory optimization.

    Example:
        >>> with OllamaServerContext() as provider:
        ...     response = provider.generate_response(prompt)
        ... # Server stopped automatically, freeing memory
    """

    def __init__(
        self,
        ollama_model: str = "dolphin3",
        ollama_url: str = "http://localhost:11434",
        auto_stop: bool = True,
    ):
        """
        Initialize context.

        Args:
            ollama_model: Model name to use
            ollama_url: Ollama API URL
            auto_stop: Stop server on exit
        """
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.auto_stop = auto_stop
        self.provider: Optional[OllamaProvider] = None

    def __enter__(self) -> "OllamaProvider":
        """Start server and return provider."""
        from ml_lib.llm.config.llm_provider_config import LLMProviderConfig

        config = LLMProviderConfig(
            model_name=self.ollama_model,
            api_endpoint=self.ollama_url,
            temperature=0.7,
        )

        self.provider = OllamaProvider(
            configuration=config,
            auto_start_server=True,
            auto_stop_on_exit=False,  # We'll handle it in __exit__
        )

        # Ensure server is running
        if not self.provider.ensure_server_running():
            raise RuntimeError("Failed to start Ollama server")

        return self.provider

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop server if configured."""
        if self.auto_stop and self.provider:
            self.provider.stop_server()

        return False  # Don't suppress exceptions


class OllamaProvider(LLMProvider):
    """
    Proveedor LLM para Ollama.

    Soporta modelos de texto y visión disponibles localmente via Ollama.
    Incluye gestión automática del servidor Ollama para optimizar memoria.
    """

    def __init__(
        self,
        configuration: LLMProviderConfig,
        http_client: HttpClientInterface | None = None,
        storage_client: StorageInterface | None = None,
        auto_start_server: bool = True,
        auto_stop_on_exit: bool = False,
    ):
        """
        Inicializa el proveedor Ollama.

        Args:
            configuration: Configuración del proveedor
            http_client: Cliente HTTP personalizado (opcional)
            storage_client: Cliente de almacenamiento personalizado (opcional)
            auto_start_server: Auto-start servidor si no está corriendo
            auto_stop_on_exit: Auto-stop servidor al salir del programa
        """
        super().__init__(configuration)
        self.api_url = configuration.api_endpoint or "http://localhost:11434"
        self.auto_start_server = auto_start_server
        self.auto_stop_on_exit = auto_stop_on_exit

        # Server process management
        self._server_process: Optional[subprocess.Popen] = None
        self._server_started_by_us = False

        # If no http_client provided, use default requests implementation
        if http_client is None:
            import requests

            class DefaultHttpClient:
                """Default HTTP client using requests library."""

                def get(self, url: str, **kwargs):
                    return requests.get(url, **kwargs)

                def post(self, url: str, data: str = None, headers: dict = None, **kwargs):
                    return requests.post(url, data=data, headers=headers, **kwargs)

            self.http_client = DefaultHttpClient()
        else:
            self.http_client = http_client

        # Optional storage client for reading image bytes (e.g. remote/local adapters)
        self.storage_client = storage_client

        # Parser for API responses
        self._parser = OllamaResponseParser()

        # Register cleanup on exit if requested
        if self.auto_stop_on_exit:
            atexit.register(self.stop_server)

    def start_server(self, wait_time: float = 3.0, max_retries: int = 10) -> bool:
        """
        Inicia el servidor Ollama si no está corriendo.

        Args:
            wait_time: Tiempo de espera inicial en segundos
            max_retries: Máximo número de reintentos para verificar

        Returns:
            True si el servidor está disponible, False en caso contrario
        """
        # Check if already running
        if self.is_available():
            logger.info("Ollama server is already running")
            return True

        logger.info("Starting Ollama server...")

        try:
            # Start ollama serve in background
            self._server_process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            self._server_started_by_us = True

            # Wait for server to be ready
            time.sleep(wait_time)

            # Verify server is responding
            for attempt in range(max_retries):
                if self.is_available():
                    logger.info("✅ Ollama server started successfully")
                    return True
                logger.debug(f"Waiting for Ollama server... attempt {attempt + 1}/{max_retries}")
                time.sleep(1.0)

            logger.error("Ollama server failed to start after multiple attempts")
            return False

        except FileNotFoundError:
            logger.error("'ollama' command not found. Please install Ollama first.")
            return False
        except Exception as e:
            logger.error(f"Failed to start Ollama server: {e}")
            return False

    def stop_server(self, force: bool = False) -> bool:
        """
        Detiene el servidor Ollama si fue iniciado por nosotros.

        Args:
            force: Forzar stop incluso si no lo iniciamos nosotros

        Returns:
            True si se detuvo exitosamente, False en caso contrario
        """
        if not force and not self._server_started_by_us:
            logger.debug("Server was not started by us, not stopping")
            return False

        if self._server_process is None:
            logger.debug("No server process to stop")
            return False

        try:
            logger.info("Stopping Ollama server...")
            self._server_process.terminate()

            # Wait for graceful shutdown
            try:
                self._server_process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                logger.warning("Server did not terminate gracefully, killing...")
                self._server_process.kill()
                self._server_process.wait(timeout=2.0)

            self._server_process = None
            self._server_started_by_us = False

            logger.info("✅ Ollama server stopped")
            return True

        except Exception as e:
            logger.error(f"Failed to stop Ollama server: {e}")
            return False

    def ensure_server_running(self) -> bool:
        """
        Asegura que el servidor esté corriendo, iniciándolo si es necesario.

        Returns:
            True si el servidor está disponible, False en caso contrario
        """
        if self.is_available():
            return True

        if self.auto_start_server:
            return self.start_server()

        return False

    def generate_response(self, prompt: LLMPrompt | DocumentPrompt) -> LLMResponse:
        """
        Genera una respuesta usando Ollama.
        Auto-inicia el servidor si está configurado y no está corriendo.

        Args:
            prompt: Prompt estructurado para el LLM

        Returns:
            LLMResponse: Respuesta estructurada del LLM
        """
        # Ensure server is running
        if not self.ensure_server_running():
            return LLMResponse(
                content="Error: Ollama server is not available and could not be started",
                usage_tokens=0,
                model_name=self.configuration.model_name,
                confidence_score=0.0,
            )

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
