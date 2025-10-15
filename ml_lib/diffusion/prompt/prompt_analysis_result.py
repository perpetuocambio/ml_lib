@dataclass(frozen=True)
class PromptAnalysisResult:
    """Result of prompt analysis.

    Attributes:
        concepts: Detected concepts in the prompt.
        emphases: Emphasis map for keywords.
        reasoning: Reasoning for analysis decisions.

    Example:
        >>> result = PromptAnalysisResult(
        ...     concepts=concept_map,
        ...     emphases=emphasis_map,
        ...     reasoning=reasoning_map
        ... )
    """

    concepts: ConceptMap
    emphases: EmphasisMap
    reasoning: ReasoningMap

    @property
    def concept_count(self) -> int:
        """Number of detected concepts."""
        return self.concepts.concept_count

    @property
    def emphasis_count(self) -> int:
        """Number of emphasized keywords."""
        return self.emphases.count
