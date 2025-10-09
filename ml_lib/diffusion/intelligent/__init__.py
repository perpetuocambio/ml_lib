"""
Intelligent Image Generation Module.

This module provides intelligent automation for image generation using
diffusion models from HuggingFace Hub and CivitAI.

Features:
- Automatic model and LoRA discovery
- Semantic prompt analysis with Ollama
- Intelligent parameter optimization
- Efficient memory management
- End-to-end generation pipeline
"""

from ml_lib.diffusion.intelligent.hub_integration import (
    HuggingFaceHubService,
    CivitAIService,
    ModelRegistry,
)

__all__ = [
    "HuggingFaceHubService",
    "CivitAIService",
    "ModelRegistry",
]
