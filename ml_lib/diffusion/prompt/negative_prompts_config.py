from dataclasses import dataclass, field


@dataclass(frozen=True)
class NegativePromptsConfig:
    """Configuration for negative prompts."""

    general: list[str] = field(
        default_factory=lambda: ["low quality", "blurry", "deformed", "bad anatomy"]
    )
    photorealistic: list[str] = field(
        default_factory=lambda: ["cartoon", "anime", "unrealistic"]
    )
    age_inappropriate: list[str] = field(
        default_factory=lambda: ["child", "minor", "teen", "underage"]
    )
    explicit: list[str] = field(default_factory=lambda: ["nsfw"])
