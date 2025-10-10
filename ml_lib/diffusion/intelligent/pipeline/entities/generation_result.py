"""Generation result entities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from PIL import Image
import json


@dataclass
class LoRAInfo:
    """Information about a LoRA used in generation."""

    name: str
    """LoRA name/ID."""

    alpha: float
    """LoRA weight/alpha value."""

    source: str = "huggingface"
    """Source: 'huggingface', 'civitai', 'local'."""


@dataclass
class GenerationMetadata:
    """Complete metadata for a generation."""

    # Prompt
    prompt: str
    negative_prompt: str
    seed: int

    # Parameters
    steps: int
    cfg_scale: float
    width: int
    height: int
    sampler: str

    # LoRAs
    loras_used: list[LoRAInfo]

    # Resources
    generation_time_seconds: float
    peak_vram_gb: float

    # Model
    base_model_id: str
    pipeline_type: str = "intelligent"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "seed": self.seed,
            "steps": self.steps,
            "cfg_scale": self.cfg_scale,
            "width": self.width,
            "height": self.height,
            "sampler": self.sampler,
            "loras_used": [
                {"name": lora.name, "alpha": lora.alpha, "source": lora.source}
                for lora in self.loras_used
            ],
            "generation_time_seconds": self.generation_time_seconds,
            "peak_vram_gb": self.peak_vram_gb,
            "base_model_id": self.base_model_id,
            "pipeline_type": self.pipeline_type,
        }


@dataclass
class GenerationResult:
    """Result of an image generation."""

    id: str
    """Unique generation ID."""

    image: Image.Image
    """Generated image."""

    metadata: GenerationMetadata
    """Generation metadata."""

    explanation: "GenerationExplanation"
    """Explanation of decisions made."""

    def save(
        self,
        path: Path | str,
        save_metadata: bool = True,
        save_explanation: bool = True,
    ):
        """
        Save image with optional metadata and explanation.

        Args:
            path: Path to save image (PNG format)
            save_metadata: Whether to embed metadata in PNG
            save_explanation: Whether to save explanation as JSON
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save image
        if save_metadata:
            # Embed metadata in PNG
            pnginfo = Image.PngImagePlugin.PngInfo()
            pnginfo.add_text("metadata", json.dumps(self.metadata.to_dict()))
            self.image.save(path, pnginfo=pnginfo)
        else:
            self.image.save(path)

        # Save explanation as separate JSON
        if save_explanation:
            explanation_path = path.with_suffix(".explanation.json")
            with open(explanation_path, "w") as f:
                json.dump(
                    {
                        "summary": self.explanation.summary,
                        "lora_reasoning": self.explanation.lora_reasoning,
                        "parameter_reasoning": self.explanation.parameter_reasoning,
                        "performance_notes": self.explanation.performance_notes,
                    },
                    f,
                    indent=2,
                )

    def save_metadata_json(self, path: Path | str):
        """
        Save metadata as standalone JSON file.

        Args:
            path: Path to save JSON
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.metadata.to_dict(), f, indent=2)
