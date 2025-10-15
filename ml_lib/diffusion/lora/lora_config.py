from dataclasses import dataclass, field

from ml_lib.diffusion.generation.types import TagList, WeightDict


@dataclass(frozen=True)
class LoRAConfig:
    """Configuration for LoRA recommendations."""

    blocked_tags: TagList = field(
        default_factory=lambda: ["anime", "cartoon", "child", "minor", "teen"]
    )
    priority_tags: TagList = field(
        default_factory=lambda: ["photorealistic", "nsfw", "mature", "realistic"]
    )
    anatomy_tags: TagList = field(
        default_factory=lambda: ["anatomy", "detailed", "breasts", "body"]
    )
    scoring_weights: WeightDict = field(
        default_factory=lambda: {
            "priority_score_weight": 0.25,
            "anatomy_score_weight": 0.20,
            "keyword_score_weight": 0.25,
            "tag_score_weight": 0.20,
            "popularity_score_weight": 0.10,
        }
    )
    limits: dict[str, int | float] = field(
        default_factory=lambda: {
            "max_loras": 3,
            "min_confidence": 0.5,
            "min_individual_weight": 0.3,
            "max_individual_weight": 1.2,
            "max_total_weight": 3.0,
        }
    )
