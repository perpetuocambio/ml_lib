"""
Ollama-powered intelligent model selection.

Uses Ollama LLM to analyze prompts and select optimal models.

Analyzes:
- Prompt style (realistic, anime, artistic, etc.)
- Content (portraits, landscapes, characters, etc.)
- Desired quality level
- Technical requirements

Recommends:
- Best base model
- Compatible LoRAs
- Optimal generation parameters
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PromptAnalysis:
    """Analysis of user prompt."""

    # Style detection
    style: str  # "realistic", "anime", "artistic", "3d", "photo", etc.
    style_confidence: float  # 0-1

    # Content type
    content_type: str  # "portrait", "landscape", "character", "scene", etc.
    content_confidence: float

    # Quality indicators
    suggested_quality: str  # "fast", "balanced", "high", "ultra"

    # Keywords for model matching
    key_concepts: list[str]  # Important concepts from prompt
    trigger_words: list[str]  # Potential trigger words

    # Technical recommendations
    suggested_base_model: str  # "SDXL", "SD15", "Flux", etc.
    suggested_steps: int
    suggested_cfg: float

    # LoRA recommendations
    recommended_lora_tags: list[str]  # Tags to match LoRAs


class OllamaModelSelector:
    """
    Intelligent model selector using Ollama.

    Analyzes prompt semantically to select optimal models.
    Includes automatic server management for memory optimization.
    """

    def __init__(
        self,
        ollama_model: str = "llama3.2",
        ollama_url: str = "http://localhost:11434",
        auto_manage_server: bool = True,
    ):
        """
        Initialize Ollama selector.

        Args:
            ollama_model: Ollama model to use
            ollama_url: Ollama API endpoint
            auto_manage_server: Auto-start server if not running
        """
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.auto_manage_server = auto_manage_server
        self._ollama_provider = None  # Lazy init
        self._check_ollama()

    def _get_or_create_provider(self) -> "OllamaProvider":  # type: ignore
        """Get or create Ollama provider with auto-start."""
        if self._ollama_provider is None:
            from ml_lib.llm.providers.ollama_provider import OllamaProvider
            from ml_lib.llm.config.llm_provider_config import LLMProviderConfig, LLMProviderType

            config = LLMProviderConfig(
                provider_type=LLMProviderType.OLLAMA,
                model_name=self.ollama_model,
                api_endpoint=self.ollama_url,
                default_temperature=0.7,
            )

            self._ollama_provider = OllamaProvider(
                configuration=config,
                auto_start_server=self.auto_manage_server,
                auto_stop_on_exit=False,  # Manual control for memory optimization
            )

        return self._ollama_provider

    def stop_server(self, force: bool = False) -> bool:
        """
        Stop Ollama server to free memory.

        Call this after analysis is complete to optimize memory usage.

        Args:
            force: Force stop even if not started by us (uses pkill)

        Returns:
            True if stopped successfully
        """
        if self._ollama_provider:
            return self._ollama_provider.stop_server(force=force)
        return False

    def _check_ollama(self) -> bool:
        """Check if Ollama is available, starting if necessary."""
        try:
            import requests

            try:
                response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
                if response.status_code == 200:
                    logger.info("Ollama connection successful")
                    return True
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                # Server not running
                logger.debug("Ollama server not responding")

                # Try to start if auto_manage enabled
                if self.auto_manage_server:
                    logger.info("Attempting to start Ollama server...")
                    provider = self._get_or_create_provider()
                    return provider.ensure_server_running()

                return False

        except Exception as e:
            logger.warning(f"Error checking Ollama: {e}")
            return False

        return False

    def analyze_prompt(self, prompt: str) -> Optional[PromptAnalysis]:
        """
        Analyze prompt using Ollama.

        Args:
            prompt: User's generation prompt

        Returns:
            Prompt analysis or None if Ollama unavailable
        """
        try:
            import requests

            # Construct analysis prompt
            analysis_prompt = self._build_analysis_prompt(prompt)

            # Call Ollama
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": self.ollama_model, "prompt": analysis_prompt, "stream": False},
                timeout=30,
            )

            if response.status_code != 200:
                logger.warning(f"Ollama API error: {response.status_code}")
                return None

            result = response.json()
            analysis_text = result.get("response", "")

            # Parse response
            return self._parse_analysis(analysis_text, prompt)

        except Exception as e:
            logger.warning(f"Failed to analyze prompt with Ollama: {e}")
            return None

    def _build_analysis_prompt(self, user_prompt: str) -> str:
        """Build analysis prompt for Ollama."""
        return f"""Analyze this image generation prompt and provide structured recommendations.

User Prompt: "{user_prompt}"

Provide your analysis in JSON format with these fields:

{{
  "style": "realistic/anime/artistic/3d/photo/painting/etc",
  "style_confidence": 0.0-1.0,
  "content_type": "portrait/landscape/character/scene/object/etc",
  "content_confidence": 0.0-1.0,
  "suggested_quality": "fast/balanced/high/ultra",
  "key_concepts": ["list", "of", "main", "concepts"],
  "trigger_words": ["potential", "trigger", "words"],
  "suggested_base_model": "SDXL/SD15/Flux/SD3",
  "suggested_steps": 20-50,
  "suggested_cfg": 3.0-12.0,
  "recommended_lora_tags": ["tags", "to", "match", "loras"],
  "nsfw_acts": ["explicit", "sexual", "acts", "if", "present"]
}}

Guidelines:
- For photorealistic prompts: suggest "realistic" style, SDXL or Flux
- For anime/manga: suggest "anime" style, SD15 or SDXL with anime models
- For artistic/painting: suggest "artistic" style
- Quality: "fast" for testing, "balanced" for general, "high" for quality, "ultra" for best
- Extract key concepts that would match model tags
- Suggest tags that would find relevant LoRAs (character types, styles, effects)

IMPORTANT: For explicit/NSFW prompts, extract specific acts to nsfw_acts field:
- fellatio, blowjob, oral → "oral"
- anal, anal sex → "anal"
- vaginal, sex, fucking → "sex"
- cum, cumshot, facial, bukkake → "cum"
- bondage, bdsm, tied, restrained → "bondage"
- nude, naked, breasts, nipples → "nude"
- masturbation, fingering → "masturbation"

These NSFW tags are critical for finding relevant LoRAs!

Respond ONLY with valid JSON, no other text."""

    def _parse_analysis(self, analysis_text: str, original_prompt: str) -> PromptAnalysis:
        """Parse Ollama response into PromptAnalysis."""
        try:
            # Try to extract JSON from response
            # Sometimes Ollama wraps response in markdown
            if "```json" in analysis_text:
                analysis_text = analysis_text.split("```json")[1].split("```")[0]
            elif "```" in analysis_text:
                analysis_text = analysis_text.split("```")[1].split("```")[0]

            data = json.loads(analysis_text.strip())

            analysis = PromptAnalysis(
                style=data.get("style", "realistic"),
                style_confidence=float(data.get("style_confidence", 0.7)),
                content_type=data.get("content_type", "scene"),
                content_confidence=float(data.get("content_confidence", 0.7)),
                suggested_quality=data.get("suggested_quality", "balanced"),
                key_concepts=data.get("key_concepts", []),
                trigger_words=data.get("trigger_words", []),
                suggested_base_model=data.get("suggested_base_model", "SDXL"),
                suggested_steps=int(data.get("suggested_steps", 30)),
                suggested_cfg=float(data.get("suggested_cfg", 7.0)),
                recommended_lora_tags=data.get("recommended_lora_tags", []),
            )

            # Add nsfw_acts as attribute if present
            if "nsfw_acts" in data:
                analysis.nsfw_acts = data["nsfw_acts"]
            else:
                analysis.nsfw_acts = []

            return analysis

        except Exception as e:
            logger.warning(f"Failed to parse Ollama response: {e}")
            # Return fallback analysis
            return self._fallback_analysis(original_prompt)

    def _fallback_analysis(self, prompt: str) -> PromptAnalysis:
        """Fallback analysis when Ollama unavailable or parsing fails."""
        # Simple keyword-based analysis
        prompt_lower = prompt.lower()

        # Detect style
        style = "realistic"
        if any(word in prompt_lower for word in ["anime", "manga", "cartoon"]):
            style = "anime"
        elif any(word in prompt_lower for word in ["painting", "artistic", "art"]):
            style = "artistic"
        elif any(word in prompt_lower for word in ["photo", "photograph", "realistic"]):
            style = "realistic"

        # Detect content
        content_type = "scene"
        if any(word in prompt_lower for word in ["portrait", "face", "person"]):
            content_type = "portrait"
        elif any(word in prompt_lower for word in ["landscape", "mountain", "nature"]):
            content_type = "landscape"
        elif any(word in prompt_lower for word in ["character", "girl", "boy"]):
            content_type = "character"

        # Extract NSFW keywords for better LoRA matching
        nsfw_keywords = {
            "oral", "fellatio", "blowjob", "deepthroat",
            "anal", "vaginal", "sex", "fucking",
            "cum", "cumshot", "facial", "bukkake", "creampie",
            "penetration", "pussy", "cock", "penis", "dick",
            "nude", "naked", "breasts", "nipples", "tits", "ass",
            "threesome", "gangbang", "group",
            "bdsm", "bondage", "tied", "restrained",
            "masturbation", "handjob", "footjob", "titjob", "paizuri",
            "missionary", "doggystyle", "cowgirl", "spitroast",
            "ahegao", "orgasm", "aroused"
        }

        # Find NSFW acts in prompt
        found_nsfw = [kw for kw in nsfw_keywords if kw in prompt_lower]

        # Extract key concepts (prioritize NSFW + other important words)
        words = prompt_lower.split()
        key_concepts = []

        # Add found NSFW keywords first (highest priority)
        key_concepts.extend(found_nsfw[:5])

        # Add other meaningful words
        other_concepts = [w for w in words if len(w) > 4 and w.isalpha() and w not in found_nsfw]
        key_concepts.extend(other_concepts[:5])

        # Build recommended LoRA tags (NSFW acts + style + character types)
        recommended_lora_tags = []
        recommended_lora_tags.extend(found_nsfw[:10])  # All NSFW acts found

        # Add character/style tags
        character_tags = []
        for word in ["girl", "woman", "anime", "realistic", "photo"]:
            if word in prompt_lower:
                character_tags.append(word)
        recommended_lora_tags.extend(character_tags)

        # Add explicit/nsfw tag if any NSFW content found
        if found_nsfw:
            recommended_lora_tags.extend(["nsfw", "explicit", "sex"])

        return PromptAnalysis(
            style=style,
            style_confidence=0.6,
            content_type=content_type,
            content_confidence=0.6,
            suggested_quality="balanced",
            key_concepts=key_concepts[:10],  # Keep top 10
            trigger_words=[],
            suggested_base_model="SDXL",
            suggested_steps=30,
            suggested_cfg=7.0,
            recommended_lora_tags=list(set(recommended_lora_tags)),  # Remove duplicates
        )


class ModelMatcher:
    """
    Match prompt analysis to available models.

    Uses semantic matching to find best models for prompt.

    Note: This class does NOT actually use Ollama for matching - it uses
    algorithmic scoring based on tags, popularity, etc. The ollama_selector
    parameter is kept for backward compatibility but is not used.
    """

    def __init__(self, ollama_selector: Optional[OllamaModelSelector] = None):
        """
        Initialize matcher.

        Args:
            ollama_selector: Ollama selector (optional, not used for scoring)
        """
        # Don't create Ollama instance if not provided - we don't actually need it
        self.ollama_selector = ollama_selector

    def match_base_model(
        self, analysis: PromptAnalysis, available_models: list
    ) -> Optional[any]:
        """
        Match best base model for analysis.

        Args:
            analysis: Prompt analysis
            available_models: List of available base models with metadata

        Returns:
            Best matching model or None
        """
        if not available_models:
            return None

        # Score each model
        scored_models = []

        for model in available_models:
            score = self._score_base_model(model, analysis)
            scored_models.append((score, model))

        # Sort by score (highest first)
        scored_models.sort(key=lambda x: x[0], reverse=True)

        best_score, best_model = scored_models[0]

        logger.info(f"Best base model: {best_model.model_name} (score: {best_score:.2f})")

        return best_model

    def match_loras(
        self,
        analysis: PromptAnalysis,
        available_loras: list,
        base_model_architecture: str,
        max_loras: int = 3,
    ) -> list:
        """
        Match best LoRAs for analysis.

        Args:
            analysis: Prompt analysis
            available_loras: List of available LoRAs with metadata
            base_model_architecture: Base model type (must match)
            max_loras: Maximum number of LoRAs to select

        Returns:
            List of (lora, weight) tuples
        """
        if not available_loras:
            return []

        # Filter by compatible architecture
        compatible_loras = [
            lora
            for lora in available_loras
            if self._is_compatible_architecture(lora.base_model, base_model_architecture)
        ]

        if not compatible_loras:
            logger.warning(f"No LoRAs compatible with {base_model_architecture}")
            return []

        # Score each LoRA
        scored_loras = []

        for lora in compatible_loras:
            score = self._score_lora(lora, analysis)
            weight = lora.recommended_lora_weight
            scored_loras.append((score, lora, weight))

        # Sort by score
        scored_loras.sort(key=lambda x: x[0], reverse=True)

        # Take top N
        selected = scored_loras[:max_loras]

        logger.info(
            f"Selected {len(selected)} LoRAs: "
            f"{', '.join([lora.model_name for _, lora, _ in selected])}"
        )

        return [(lora, weight) for _, lora, weight in selected]

    def _score_base_model(self, model, analysis: PromptAnalysis) -> float:
        """Score base model match for analysis."""
        score = 0.0

        # Architecture match
        if model.get_base_model_enum().value.upper() == analysis.suggested_base_model.upper():
            score += 30.0

        # Popularity (normalized to 0-20)
        score += min(model.popularity_score / 5, 20.0)

        # Tag matching
        model_tags = [tag.lower() for tag in model.tags]
        key_concepts = [kw.lower() for kw in analysis.key_concepts]

        matching_tags = len(set(model_tags) & set(key_concepts))
        score += matching_tags * 10.0

        # Style matching
        style_keywords = {
            "realistic": ["realistic", "photo", "photography", "photorealistic"],
            "anime": ["anime", "manga", "cartoon", "animated"],
            "artistic": ["artistic", "art", "painting", "illustration"],
        }

        style_terms = style_keywords.get(analysis.style, [])
        if any(term in " ".join(model_tags) for term in style_terms):
            score += 20.0

        return score

    def _score_lora(self, lora, analysis: PromptAnalysis) -> float:
        """Score LoRA match for analysis."""
        score = 0.0

        # Extract NSFW keywords from analysis
        nsfw_keywords = {
            "oral", "fellatio", "blowjob", "deepthroat",
            "anal", "vaginal", "sex", "fucking",
            "cum", "cumshot", "facial", "bukkake", "creampie",
            "penetration", "pussy", "cock", "penis", "dick",
            "nude", "naked", "breasts", "nipples", "tits", "ass",
            "threesome", "gangbang", "group",
            "bdsm", "bondage", "tied", "restrained",
            "nsfw", "explicit", "xxx", "pornographic"
        }

        lora_tags = [tag.lower() for tag in lora.tags]
        lora_name_lower = lora.model_name.lower()
        recommended_tags = [tag.lower() for tag in analysis.recommended_lora_tags]
        key_concepts = [kw.lower() for kw in analysis.key_concepts]

        # Check if prompt has NSFW content
        has_nsfw_content = any(kw in " ".join(recommended_tags + key_concepts) for kw in nsfw_keywords)

        # BONUS 1: NSFW LoRA matching (HIGH PRIORITY)
        if has_nsfw_content:
            # Check if LoRA has NSFW tags
            lora_nsfw_tags = [tag for tag in lora_tags if any(nsfw in tag for nsfw in nsfw_keywords)]
            nsfw_in_name = any(nsfw in lora_name_lower for nsfw in ["nsfw", "sex", "explicit", "xxx"])

            if lora_nsfw_tags or nsfw_in_name:
                score += 25.0  # BIG bonus for NSFW LoRAs matching NSFW prompts
                logger.debug(f"NSFW bonus for {lora.model_name}: +25.0")

            # Extra bonus for specific act matching
            prompt_nsfw_acts = [kw for kw in key_concepts if kw in nsfw_keywords]
            matching_acts = len(set(lora_tags) & set(prompt_nsfw_acts))
            if matching_acts > 0:
                score += matching_acts * 20.0
                logger.debug(f"NSFW act matching for {lora.model_name}: +{matching_acts * 20.0}")

        # Popularity (scaled down to make NSFW matching more important)
        score += min(lora.popularity_score / 5, 10.0)

        # Tag matching (general)
        all_search_terms = set(recommended_tags + key_concepts)
        matching_tags = len(set(lora_tags) & all_search_terms)
        score += matching_tags * 12.0

        # Trigger word matching
        trigger_words = [tw.lower() for tw in lora.trigger_words]
        prompt_words = set(key_concepts)

        if any(tw in prompt_words for tw in trigger_words):
            score += 8.0

        # Model name matching (sometimes model names are descriptive)
        if any(concept in lora_name_lower for concept in key_concepts):
            score += 8.0

        return score

    def _is_compatible_architecture(
        self, lora_base_model: str, target_architecture: str
    ) -> bool:
        """Check if LoRA is compatible with target architecture."""
        lora_lower = lora_base_model.lower()
        target_lower = target_architecture.lower()

        # Direct match
        if target_lower in lora_lower or lora_lower in target_lower:
            return True

        # Architecture compatibility
        compatibility_map = {
            "sdxl": ["sdxl", "pony", "xl"],
            "pony": ["sdxl", "pony", "xl"],
            "sd15": ["sd 1.5", "sd15", "sd1.5"],
            "sd3": ["sd 3", "sd3"],
            "flux": ["flux"],
        }

        target_compat = compatibility_map.get(target_lower, [])
        return any(compat in lora_lower for compat in target_compat)
