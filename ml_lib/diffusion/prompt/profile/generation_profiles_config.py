from ml_lib.diffusion.prompt.age.age_profile import AgeProfile


@dataclass(frozen=True)
class GenerationProfilesConfig:
    """Generation profiles configuration."""

    age_profiles: dict[str, AgeProfile]
    group_profiles: dict[str, GroupProfile]
    activity_profiles: dict[str, ActivityProfile]
    detail_presets: dict[str, DetailPreset]
    default_ranges: DefaultRanges
    vram_presets: dict[str, VramPreset]
