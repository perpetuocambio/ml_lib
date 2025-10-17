"""Feedback collector for gathering user feedback on generations."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Protocol
from dataclasses import dataclass, field
import json

from ml_lib.diffusion.domain.value_objects_models.value_objects import (
    GenerationParameters,
    ParameterModification,
    ParameterModifications,
    FeedbackStatistics,
    TagCount,
)

logger = logging.getLogger(__name__)


class RecommendationsProtocol(Protocol):
    """Protocol for recommendations object."""
    suggested_loras: list  # list of LoRA recommendations
    suggested_params: "SuggestedParamsProtocol"  # parameter recommendations


class SuggestedParamsProtocol(Protocol):
    """Protocol for suggested parameters."""
    num_steps: int
    guidance_scale: float
    width: int
    height: int
    sampler_name: str


@dataclass(frozen=True)
class GenerationSession:
    """Tracking data for a generation session."""

    generation_id: str
    timestamp: str
    prompt: str
    negative_prompt: str

    # What was recommended
    recommended_loras: list[str]
    recommended_params: GenerationParameters

    # What was actually used (may differ if user modified)
    actual_loras: list[str]
    actual_params: GenerationParameters

    # User modifications
    user_modified: bool = False
    modifications: ParameterModifications = field(default_factory=lambda: ParameterModifications())


@dataclass
class UserFeedback:
    """User feedback on a generation."""

    generation_id: str
    timestamp: str

    # Rating
    rating: int  # 1-5 stars
    liked: Optional[bool] = None  # Simple like/dislike

    # Detailed feedback
    quality_rating: Optional[int] = None  # 1-5
    accuracy_rating: Optional[int] = None  # How well it matched prompt
    aesthetic_rating: Optional[int] = None  # Artistic quality

    # Textual feedback
    comments: str = ""
    tags: list[str] = field(default_factory=list)

    # Actions
    saved: bool = False
    shared: bool = False
    regenerated: bool = False

    def __post_init__(self):
        """Validate feedback."""
        if not 1 <= self.rating <= 5:
            raise ValueError("Rating must be between 1 and 5")
        if self.quality_rating is not None and not 1 <= self.quality_rating <= 5:
            raise ValueError("Quality rating must be between 1 and 5")
        if self.accuracy_rating is not None and not 1 <= self.accuracy_rating <= 5:
            raise ValueError("Accuracy rating must be between 1 and 5")
        if self.aesthetic_rating is not None and not 1 <= self.aesthetic_rating <= 5:
            raise ValueError("Aesthetic rating must be between 1 and 5")


class LearningEngineProtocol(Protocol):
    """Protocol for learning engine."""
    def record_feedback(self, feedback: "GenerationFeedback") -> None:  # type: ignore
        """Record feedback for learning."""
        ...


class FeedbackCollector:
    """
    Collects user feedback on generations for learning and improvement.

    Tracks:
    - Generation sessions (what was generated, what was recommended)
    - User feedback (ratings, comments, tags)
    - User modifications (what did they change?)
    - User actions (save, share, regenerate)

    Integrates with LearningEngine to improve future recommendations.

    Example:
        >>> collector = FeedbackCollector(learning_engine)
        >>> # Record a generation
        >>> collector.start_session(
        ...     generation_id="gen_001",
        ...     prompt="anime girl",
        ...     recommendations=recommendations
        ... )
        >>> # User provides feedback
        >>> feedback = UserFeedback(
        ...     generation_id="gen_001",
        ...     timestamp=datetime.now().isoformat(),
        ...     rating=5,
        ...     comments="Perfect!"
        ... )
        >>> collector.collect_feedback(feedback)
    """

    def __init__(
        self,
        learning_engine: Optional[LearningEngineProtocol] = None,
        session_log_path: Optional[Path] = None,
    ):
        """
        Initialize feedback collector.

        Args:
            learning_engine: LearningEngine instance (optional)
            session_log_path: Path to save session logs (optional)
        """
        self.learning_engine = learning_engine
        self.session_log_path = session_log_path

        # In-memory session tracking
        self._active_sessions: list[GenerationSession] = []

        # Feedback history
        self._feedback_history: list[UserFeedback] = []

        logger.info("FeedbackCollector initialized")

    def _find_session(self, generation_id: str) -> Optional[GenerationSession]:
        """Find session by ID."""
        for session in self._active_sessions:
            if session.generation_id == generation_id:
                return session
        return None

    def start_session(
        self,
        generation_id: str,
        prompt: str,
        negative_prompt: str,
        recommendations: RecommendationsProtocol,
        actual_params: Optional[GenerationParameters] = None,
    ) -> GenerationSession:
        """
        Start tracking a generation session.

        Args:
            generation_id: Unique generation ID
            prompt: Text prompt
            negative_prompt: Negative prompt
            recommendations: Recommendations that were made
            actual_params: Actual params used (None = same as recommended)

        Returns:
            GenerationSession
        """
        # Extract recommended LoRAs and params
        recommended_loras = [
            lora.lora_name for lora in recommendations.suggested_loras
        ]
        recommended_params = GenerationParameters(
            num_steps=recommendations.suggested_params.num_steps,
            guidance_scale=recommendations.suggested_params.guidance_scale,
            width=recommendations.suggested_params.width,
            height=recommendations.suggested_params.height,
            sampler=recommendations.suggested_params.sampler_name,
            seed=None,  # Not tracked in recommendations
        )

        # Check if user modified
        actual_params_final = actual_params if actual_params is not None else recommended_params
        user_modified = actual_params_final != recommended_params

        # Build modifications
        modifications_dict = {}
        if user_modified:
            if actual_params_final.num_steps != recommended_params.num_steps:
                modifications_dict["num_steps"] = ParameterModification(
                    recommended_value=recommended_params.num_steps,
                    actual_value=actual_params_final.num_steps
                )
            if actual_params_final.guidance_scale != recommended_params.guidance_scale:
                modifications_dict["guidance_scale"] = ParameterModification(
                    recommended_value=recommended_params.guidance_scale,
                    actual_value=actual_params_final.guidance_scale
                )
            if actual_params_final.width != recommended_params.width:
                modifications_dict["width"] = ParameterModification(
                    recommended_value=recommended_params.width,
                    actual_value=actual_params_final.width
                )
            if actual_params_final.height != recommended_params.height:
                modifications_dict["height"] = ParameterModification(
                    recommended_value=recommended_params.height,
                    actual_value=actual_params_final.height
                )
            if actual_params_final.sampler != recommended_params.sampler:
                modifications_dict["sampler"] = ParameterModification(
                    recommended_value=recommended_params.sampler,
                    actual_value=actual_params_final.sampler
                )
            if actual_params_final.seed != recommended_params.seed:
                modifications_dict["seed"] = ParameterModification(
                    recommended_value=recommended_params.seed if recommended_params.seed is not None else "None",
                    actual_value=actual_params_final.seed if actual_params_final.seed is not None else "None"
                )

        modifications = ParameterModifications(
            num_steps=modifications_dict.get("num_steps"),
            guidance_scale=modifications_dict.get("guidance_scale"),
            width=modifications_dict.get("width"),
            height=modifications_dict.get("height"),
            sampler=modifications_dict.get("sampler"),
            seed=modifications_dict.get("seed"),
        )

        session = GenerationSession(
            generation_id=generation_id,
            timestamp=datetime.now().isoformat(),
            prompt=prompt,
            negative_prompt=negative_prompt,
            recommended_loras=recommended_loras,
            recommended_params=recommended_params,
            actual_loras=recommended_loras,  # For now, assume same
            actual_params=actual_params_final,
            user_modified=user_modified,
            modifications=modifications,
        )

        self._active_sessions.append(session)

        logger.debug(
            f"Started session {generation_id[:8]}... "
            f"(modified: {user_modified})"
        )

        return session

    def collect_feedback(
        self,
        feedback: UserFeedback,
        session: Optional[GenerationSession] = None,
    ) -> None:
        """
        Collect user feedback for a generation.

        Args:
            feedback: User feedback
            session: Associated session (None = lookup by ID)
        """
        # Get session if not provided
        if session is None:
            session = self._find_session(feedback.generation_id)

        if session is None:
            logger.warning(
                f"No session found for generation {feedback.generation_id[:8]}..."
            )
            return

        # Store feedback
        self._feedback_history.append(feedback)

        # Send to learning engine
        if self.learning_engine:
            self._send_to_learning_engine(feedback, session)

        # Log session if path provided
        if self.session_log_path:
            self._log_session(session, feedback)

        logger.info(
            f"Collected feedback for {feedback.generation_id[:8]}... "
            f"(rating: {feedback.rating}/5)"
        )

    def _send_to_learning_engine(
        self,
        feedback: UserFeedback,
        session: GenerationSession,
    ) -> None:
        """
        Send feedback to learning engine for training.

        Args:
            feedback: User feedback
            session: Generation session
        """
        # Import here to avoid circular dependency
        from ml_lib.diffusion.domain.services.learning_engine import GenerationFeedback

        # Track LoRA modifications
        user_modified_loras = None
        if session.actual_loras != session.recommended_loras:
            user_modified_loras = {
                "recommended": session.recommended_loras,
                "actual": session.actual_loras,
                "added": [lora for lora in session.actual_loras if lora not in session.recommended_loras],
                "removed": [lora for lora in session.recommended_loras if lora not in session.actual_loras],
            }

        # Convert to learning engine format
        learning_feedback = GenerationFeedback(
            feedback_id=feedback.generation_id,
            timestamp=feedback.timestamp,
            original_prompt=session.prompt,
            recommended_loras=session.recommended_loras,
            recommended_params=session.recommended_params,
            rating=feedback.rating,
            user_modified_loras=user_modified_loras,
            user_modified_params=(
                session.actual_params if session.user_modified else None
            ),
            tags=feedback.tags,
            notes=feedback.comments,
        )

        try:
            if self.learning_engine:
                self.learning_engine.record_feedback(learning_feedback)
                logger.debug("Feedback sent to learning engine")
        except Exception as e:
            logger.error(f"Failed to send feedback to learning engine: {e}")

    def _log_session(
        self,
        session: GenerationSession,
        feedback: UserFeedback,
    ) -> None:
        """
        Log session and feedback to file.

        Args:
            session: Generation session
            feedback: User feedback
        """
        if not self.session_log_path:
            return

        log_path = Path(self.session_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Create log entry - allowed dict only for JSON serialization
        log_entry = {
            "session": {
                "generation_id": session.generation_id,
                "timestamp": session.timestamp,
                "prompt": session.prompt,
                "recommended_loras": session.recommended_loras,
                "recommended_params": {
                    "num_steps": session.recommended_params.num_steps,
                    "guidance_scale": session.recommended_params.guidance_scale,
                    "width": session.recommended_params.width,
                    "height": session.recommended_params.height,
                    "sampler": session.recommended_params.sampler,
                },
                "user_modified": session.user_modified,
                "modifications": session.modifications.count(),
            },
            "feedback": {
                "rating": feedback.rating,
                "liked": feedback.liked,
                "comments": feedback.comments,
                "tags": feedback.tags,
                "saved": feedback.saved,
                "shared": feedback.shared,
            },
        }

        try:
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            logger.debug(f"Session logged to {log_path}")
        except Exception as e:
            logger.error(f"Failed to log session: {e}")

    def get_feedback_stats(self) -> FeedbackStatistics:
        """
        Get statistics about collected feedback.

        Returns:
            FeedbackStatistics object with feedback statistics
        """
        if not self._feedback_history:
            return FeedbackStatistics(
                total_feedback=0,
                average_rating=0.0,
                like_rate=0.0,
                saved_count=0,
                shared_count=0,
                rating_1_count=0,
                rating_2_count=0,
                rating_3_count=0,
                rating_4_count=0,
                rating_5_count=0,
            )

        total = len(self._feedback_history)
        avg_rating = sum(f.rating for f in self._feedback_history) / total

        liked_count = sum(
            1 for f in self._feedback_history if f.liked is True
        )
        like_rate = (liked_count / total * 100) if total > 0 else 0.0

        saved_count = sum(1 for f in self._feedback_history if f.saved)
        shared_count = sum(1 for f in self._feedback_history if f.shared)

        # Count ratings
        rating_counts = [0, 0, 0, 0, 0]  # Indices 0-4 for ratings 1-5
        for feedback in self._feedback_history:
            rating_counts[feedback.rating - 1] += 1

        return FeedbackStatistics(
            total_feedback=total,
            average_rating=round(avg_rating, 2),
            like_rate=round(like_rate, 1),
            saved_count=saved_count,
            shared_count=shared_count,
            rating_1_count=rating_counts[0],
            rating_2_count=rating_counts[1],
            rating_3_count=rating_counts[2],
            rating_4_count=rating_counts[3],
            rating_5_count=rating_counts[4],
        )

    def get_common_tags(self, limit: int = 10) -> list[TagCount]:
        """
        Get most common feedback tags.

        Args:
            limit: Maximum tags to return

        Returns:
            List of TagCount objects sorted by count descending
        """
        # Count tags
        tag_count_map: list[TagCount] = []
        tag_names_seen: list[str] = []

        for feedback in self._feedback_history:
            for tag in feedback.tags:
                if tag in tag_names_seen:
                    # Update existing count
                    for i, tc in enumerate(tag_count_map):
                        if tc.tag == tag:
                            tag_count_map[i] = TagCount(tag=tag, count=tc.count + 1)
                            break
                else:
                    # Add new tag
                    tag_names_seen.append(tag)
                    tag_count_map.append(TagCount(tag=tag, count=1))

        # Sort by count descending
        sorted_tags = sorted(
            tag_count_map,
            key=lambda x: x.count,
            reverse=True,
        )

        return sorted_tags[:limit]

    def export_feedback_data(self, output_path: Path) -> None:
        """
        Export all feedback data to JSON file.

        Args:
            output_path: Path to save JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stats = self.get_feedback_stats()
        common_tags = self.get_common_tags()

        # Convert value objects to dicts for JSON serialization
        data = {
            "stats": {
                "total_feedback": stats.total_feedback,
                "average_rating": stats.average_rating,
                "like_rate": stats.like_rate,
                "saved_count": stats.saved_count,
                "shared_count": stats.shared_count,
                "rating_distribution": {
                    "1": stats.rating_1_count,
                    "2": stats.rating_2_count,
                    "3": stats.rating_3_count,
                    "4": stats.rating_4_count,
                    "5": stats.rating_5_count,
                },
            },
            "common_tags": [
                {"tag": tc.tag, "count": tc.count}
                for tc in common_tags
            ],
            "feedback_history": [
                {
                    "generation_id": f.generation_id,
                    "timestamp": f.timestamp,
                    "rating": f.rating,
                    "liked": f.liked,
                    "comments": f.comments,
                    "tags": f.tags,
                    "saved": f.saved,
                    "shared": f.shared,
                }
                for f in self._feedback_history
            ],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Feedback data exported to {output_path}")

    def clear_history(self, keep_recent: int = 0) -> None:
        """
        Clear feedback history.

        Args:
            keep_recent: Number of recent entries to keep (0 = clear all)
        """
        if keep_recent > 0:
            self._feedback_history = self._feedback_history[-keep_recent:]
        else:
            self._feedback_history = []

        logger.info(f"Feedback history cleared (kept {keep_recent} recent entries)")
