"""
Prompt estructurado para integración con LLMs.
"""

from dataclasses import dataclass

from ml_lib.llm.entities.prompt_category import PromptCategory
from ml_lib.llm.entities.prompt_type import PromptType


@dataclass(frozen=True)
class LLMPrompt:
    """
    Prompt estructurado para integración con LLMs.

    Encapsula todo lo necesario para una interacción con LLM.
    """

    prompt_id: str
    prompt_type: PromptType
    category: PromptCategory
    content: str
    context_description: str
    expected_output_format: str
    temperature: float = 0.7  # 0.0 a 1.0
    max_tokens: int = 1000

    def __post_init__(self) -> None:
        """Valida los valores después de la inicialización."""
        if not (0.0 <= self.temperature <= 1.0):
            raise ValueError(
                f"Temperature debe estar entre 0.0 y 1.0, recibido: {self.temperature}"
            )
        if self.max_tokens <= 0:
            raise ValueError(
                f"Max tokens debe ser positivo, recibido: {self.max_tokens}"
            )

    def is_system_prompt(self) -> bool:
        """Verifica si es un prompt de sistema."""
        return self.category == PromptCategory.SYSTEM

    def is_for_preprocessing(self) -> bool:
        """Verifica si es para preprocesamiento de datos."""
        return self.prompt_type == PromptType.DATA_PREPROCESSING

    def is_for_analysis(self) -> bool:
        """Verifica si es para análisis principal."""
        return self.prompt_type in (
            PromptType.PATTERN_RECOGNITION,
            PromptType.HYPOTHESIS_GENERATION,
            PromptType.EVIDENCE_ANALYSIS,
        )

    def is_for_synthesis(self) -> bool:
        """Verifica si es para síntesis de resultados."""
        return self.prompt_type in (
            PromptType.RESULT_SYNTHESIS,
            PromptType.RECOMMENDATION_GENERATION,
        )

    def requires_low_creativity(self) -> bool:
        """Verifica si requiere baja creatividad (temperature baja)."""
        return self.temperature < 0.3

    def requires_high_creativity(self) -> bool:
        """Verifica si requiere alta creatividad (temperature alta)."""
        return self.temperature > 0.7
