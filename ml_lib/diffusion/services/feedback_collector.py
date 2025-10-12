"""Feedback collector for gathering user feedback on generations."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field
import json

from ml_lib.diffusion.services.learning_engine import GenerationFeedback

logger = logging.getLogger(__name__)


@dataclass
class GenerationSession:
    """Tracking data for a generation session."""

    generation_id: str
    timestamp: str
    prompt: str
    negative_prompt: str

    # What was recommended
    recommended_loras: list[str]
    recommended_params: dict[str, Any]

    # What was actually used (may differ if user modified)
    actual_loras: list[str]
    actual_params: dict[str, Any]

    # User modifications
    user_modified: bool = False
    modifications: dict[str, Any] = field(default_factory=dict)


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
        learning_engine: Optional[Any] = None,
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
        self.active_sessions: dict[str, GenerationSession] = {}

        # Feedback history
        self.feedback_history: list[UserFeedback] = []

        logger.info("FeedbackCollector initialized")

    def start_session(
        self,
        generation_id: str,
        prompt: str,
        negative_prompt: str,
        recommendations: Any,
        actual_params: Optional[dict[str, Any]] = None,
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
        recommended_params = {
            "num_steps": recommendations.suggested_params.num_steps,
            "guidance_scale": recommendations.suggested_params.guidance_scale,
            "width": recommendations.suggested_params.width,
            "height": recommendations.suggested_params.height,
            "sampler": recommendations.suggested_params.sampler_name,
        }

        # Check if user modified
        actual_params = actual_params or recommended_params
        user_modified = actual_params != recommended_params

        modifications = {}
        if user_modified:
            for key, recommended_value in recommended_params.items():
                actual_value = actual_params.get(key)
                if actual_value != recommended_value:
                    modifications[key] = {
                        "recommended": recommended_value,
                        "actual": actual_value,
                    }

        session = GenerationSession(
            generation_id=generation_id,
            timestamp=datetime.now().isoformat(),
            prompt=prompt,
            negative_prompt=negative_prompt,
            recommended_loras=recommended_loras,
            recommended_params=recommended_params,
            actual_loras=recommended_loras,  # For now, assume same
            actual_params=actual_params,
            user_modified=user_modified,
            modifications=modifications,
        )

        self.active_sessions[generation_id] = session

        logger.debug(
            f"Started session {generation_id[:8]}... "
            f"(modified: {user_modified})"
        )

        return session

    def collect_feedback(
        self,
        feedback: UserFeedback,
        session: Optional[GenerationSession] = None,
    ):
        """
        Collect user feedback for a generation.

        Args:
            feedback: User feedback
            session: Associated session (None = lookup by ID)
        """
        # Get session if not provided
        if session is None:
            session = self.active_sessions.get(feedback.generation_id)

        if session is None:
            logger.warning(
                f"No session found for generation {feedback.generation_id[:8]}..."
            )
            return

        # Store feedback
        self.feedback_history.append(feedback)

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
    ):
        """
        Send feedback to learning engine for training.

        Args:
            feedback: User feedback
            session: Generation session
        """
        # Convert to learning engine format
        learning_feedback = GenerationFeedback(
            feedback_id=feedback.generation_id,
            timestamp=feedback.timestamp,
            original_prompt=session.prompt,
            recommended_loras=session.recommended_loras,
            recommended_params=session.recommended_params,
            rating=feedback.rating,
            user_modified_loras=None,  # TODO: track LoRA modifications
            user_modified_params=(
                session.actual_params if session.user_modified else None
            ),
            tags=feedback.tags,
            notes=feedback.comments,
        )

        try:
            self.learning_engine.record_feedback(learning_feedback)
            logger.debug("Feedback sent to learning engine")
        except Exception as e:
            logger.error(f"Failed to send feedback to learning engine: {e}")

    def _log_session(
        self,
        session: GenerationSession,
        feedback: UserFeedback,
    ):
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

        # Append to log file
        log_entry = {
            "session": {
                "generation_id": session.generation_id,
                "timestamp": session.timestamp,
                "prompt": session.prompt,
                "recommended_loras": session.recommended_loras,
                "recommended_params": session.recommended_params,
                "user_modified": session.user_modified,
                "modifications": session.modifications,
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

    def get_feedback_stats(self) -> dict[str, Any]:
        """
        Get statistics about collected feedback.

        Returns:
            Dictionary with feedback statistics
        """
        if not self.feedback_history:
            return {
                "total_feedback": 0,
                "average_rating": 0.0,
                "like_rate": 0.0,
            }

        total = len(self.feedback_history)
        avg_rating = sum(f.rating for f in self.feedback_history) / total

        liked_count = sum(
            1 for f in self.feedback_history if f.liked is True
        )
        like_rate = liked_count / total if total > 0 else 0.0

        saved_count = sum(1 for f in self.feedback_history if f.saved)
        shared_count = sum(1 for f in self.feedback_history if f.shared)

        return {
            "total_feedback": total,
            "average_rating": round(avg_rating, 2),
            "like_rate": round(like_rate * 100, 1),
            "saved_count": saved_count,
            "shared_count": shared_count,
            "rating_distribution": self._get_rating_distribution(),
        }

    def _get_rating_distribution(self) -> dict[int, int]:
        """Get distribution of ratings (1-5)."""
        distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

        for feedback in self.feedback_history:
            distribution[feedback.rating] += 1

        return distribution

    def get_common_tags(self, limit: int = 10) -> list[tuple[str, int]]:
        """
        Get most common feedback tags.

        Args:
            limit: Maximum tags to return

        Returns:
            List of (tag, count) tuples
        """
        tag_counts: dict[str, int] = {}

        for feedback in self.feedback_history:
            for tag in feedback.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        # Sort by count descending
        sorted_tags = sorted(
            tag_counts.items(),
            key=lambda x: x[1],
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

        data = {
            "stats": self.get_feedback_stats(),
            "common_tags": self.get_common_tags(),
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
                for f in self.feedback_history
            ],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Feedback data exported to {output_path}")

    def clear_history(self, keep_recent: int = 0):
        """
        Clear feedback history.

        Args:
            keep_recent: Number of recent entries to keep (0 = clear all)
        """
        if keep_recent > 0:
            self.feedback_history = self.feedback_history[-keep_recent:]
        else:
            self.feedback_history = []

        logger.info(f"Feedback history cleared (kept {keep_recent} recent entries)")
