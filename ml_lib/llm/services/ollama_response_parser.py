"""Parser for Ollama API responses."""

import json
from datetime import datetime

from ml_lib.llm.entities.ollama_model_details import (
    OllamaModelDetails,
)
from ml_lib.llm.entities.ollama_model_info import OllamaModelInfo
from ml_lib.llm.entities.ollama_models_response import (
    OllamaModelsResponse,
)
from ml_lib.llm.entities.ollama_raw_model_data import (
    OllamaRawModelData,
)
from ml_lib.llm.entities.ollama_response import OllamaResponse


class OllamaResponseParser:
    """Parser for Ollama API responses without dict violations."""

    def parse_models_response(self, raw_json: str) -> OllamaModelsResponse:
        """Parse models response from JSON string."""

        data = json.loads(raw_json)

        # Parse into typed structure
        raw_models = []
        if "models" in data and isinstance(data["models"], list):
            for model_item in data["models"]:
                # Extract details
                details_data = model_item.get("details", {})
                details = OllamaModelDetails(
                    family=details_data.get("family", ""),
                    format=details_data.get("format", ""),
                    parameter_size=details_data.get("parameter_size", ""),
                    quantization_level=details_data.get("quantization_level", ""),
                )

                # Create raw model data
                raw_model = OllamaRawModelData(
                    name=model_item.get("name", ""),
                    size=model_item.get("size", 0),
                    digest=model_item.get("digest", ""),
                    modified_at=model_item.get("modified_at", "1970-01-01T00:00:00Z"),
                    details=details,
                )
                raw_models.append(raw_model)

        # Convert to domain models
        domain_models = []
        for raw_model in raw_models:
            try:
                modified_time = datetime.fromisoformat(
                    raw_model.modified_at.replace("Z", "+00:00")
                )
            except ValueError:
                modified_time = datetime.fromtimestamp(0)

            domain_model = OllamaModelInfo(
                name=raw_model.name,
                size_bytes=raw_model.size,
                digest=raw_model.digest,
                modified_at=modified_time,
                family=raw_model.details.family,
                format=raw_model.details.format,
                parameter_size=raw_model.details.parameter_size,
                quantization_level=raw_model.details.quantization_level,
            )
            domain_models.append(domain_model)

        return OllamaModelsResponse(models=domain_models)

    def parse_generate_response(self, raw_json: str) -> OllamaResponse:
        """Parse generate response from JSON string."""

        data = json.loads(raw_json)

        return OllamaResponse(
            response=data.get("response", ""),
            done=data.get("done", False),
            eval_count=data.get("eval_count", 0),
        )
