"""
Adaptador para integrar nuestro módulo LLM con MarkItDown.
"""

import logging

from infrastructure.data.extractors.entities.llm_message import LLMMessage
from infrastructure.providers.llm.entities.document_prompt import DocumentPrompt
from infrastructure.providers.llm.interfaces.llm_provider import LLMProvider

logger = logging.getLogger(__name__)


class LLMMarkItDownAdapter:
    """
    Adaptador que permite usar nuestro módulo LLM con MarkItDown.

    MarkItDown espera un cliente LLM compatible con OpenAI,
    pero nosotros podemos adaptar nuestro LLMProvider (incluyendo Ollama).
    """

    def __init__(self, llm_provider: LLMProvider, model_name: str = "llava"):
        """
        Inicializa el adaptador.

        Args:
            llm_provider: Nuestro proveedor LLM (Ollama, Azure OpenAI, etc.)
            model_name: Nombre del modelo a usar (ej. "llava" para Ollama)
        """
        self.llm_provider = llm_provider
        self.model_name = model_name

    def chat_completions_create(
        self, model: str, messages: list[LLMMessage], **kwargs
    ) -> str:
        """
        Método que simula la interfaz de OpenAI para compatibilidad con MarkItDown.

        Args:
            model: Nombre del modelo (se ignora, usamos el configurado)
            messages: Lista de mensajes estructurados
            **kwargs: Parámetros adicionales

        Returns:
            Respuesta del LLM como string
        """
        try:
            # Convertir mensajes de OpenAI a nuestro formato DocumentPrompt
            prompt_text = self._convert_messages_to_prompt(messages)

            # Crear nuestro DocumentPrompt (más simple y directo)
            document_prompt = DocumentPrompt(
                content=prompt_text,
                temperature=kwargs.get("temperature", 0.3),  # Low for document tasks
                context_window_size=kwargs.get("max_tokens", 4000),
                # Note: images would be handled separately in a real implementation
            )

            # Usar nuestro provider
            response = self.llm_provider.generate_response(document_prompt)

            # Convertir nuestra respuesta al formato esperado por MarkItDown
            return self._create_openai_compatible_response(response.content)

        except Exception as exc:
            logger.error("Error in LLM adapter: %s", exc)
            # Devolver respuesta vacía en caso de error
            return self._create_openai_compatible_response("No description available")

    def _convert_messages_to_prompt(self, messages: list[LLMMessage]) -> str:
        """
        Convierte mensajes de OpenAI a un prompt de texto simple.

        Args:
            messages: Mensajes estructurados

        Returns:
            Prompt de texto simple
        """
        prompt_parts = []

        for message in messages:
            role = message.role
            content = message.content

            if isinstance(content, str):
                prompt_parts.append(f"{role}: {content}")
            elif isinstance(content, list):
                # Manejar contenido multimodal (texto + imágenes)
                text_parts = []
                for item in content:
                    if item.content_type == "text":
                        text_parts.append(item.text or "")
                    elif item.content_type == "image_url":
                        text_parts.append("[IMAGE_DESCRIPTION_REQUEST]")

                prompt_parts.append(f"{role}: {' '.join(text_parts)}")

        return "\n".join(prompt_parts)

    def _create_openai_compatible_response(self, content: str) -> str:
        """
        Devuelve la respuesta directamente sin mock objects basura.

        Args:
            content: Contenido de la respuesta

        Returns:
            El contenido tal como está
        """
        return content
