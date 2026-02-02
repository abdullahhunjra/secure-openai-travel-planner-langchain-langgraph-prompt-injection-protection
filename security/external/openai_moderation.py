"""
OpenAI Moderation API wrapper for content safety checks.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from openai import OpenAI

from config.security_config import SECURITY_CONFIG
from security.logging.security_logger import SecurityLogger, SecurityEventType


@dataclass
class ModerationResult:
    """Result of OpenAI moderation check."""
    is_flagged: bool
    is_blocked: bool
    flagged_categories: List[str] = field(default_factory=list)
    category_scores: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None


class OpenAIModerator:
    """
    Wrapper for OpenAI Moderation API.
    """

    def __init__(self, logger: Optional[SecurityLogger] = None):
        self.logger = logger or SecurityLogger()
        self.config = SECURITY_CONFIG.moderation
        self.client = None

        # Initialize OpenAI client if API key is available
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.client = OpenAI(api_key=api_key)

    def check(self, text: str, session_id: str = "unknown") -> ModerationResult:
        """
        Check content against OpenAI's moderation API.

        Args:
            text: Content to check
            session_id: Session identifier for logging

        Returns:
            ModerationResult with moderation details
        """
        if not self.config.enable_openai_moderation:
            return ModerationResult(
                is_flagged=False,
                is_blocked=False
            )

        if not self.client:
            self.logger.log_event(
                event_type=SecurityEventType.MODERATION_CHECK,
                session_id=session_id,
                details={"status": "skipped", "reason": "no_api_key"},
                severity="warning"
            )
            return ModerationResult(
                is_flagged=False,
                is_blocked=False,
                error="OpenAI API key not configured"
            )

        if not text or len(text.strip()) == 0:
            return ModerationResult(
                is_flagged=False,
                is_blocked=False
            )

        try:
            # Call OpenAI Moderation API
            response = self.client.moderations.create(input=text)
            result = response.results[0]

            # Get flagged categories
            flagged_categories = []
            category_scores = {}

            # Check each category
            categories_dict = result.categories.model_dump()
            scores_dict = result.category_scores.model_dump()

            for category, is_flagged in categories_dict.items():
                score = scores_dict.get(category, 0.0)
                category_scores[category] = score
                if is_flagged:
                    flagged_categories.append(category)

            # Determine if should be blocked
            is_blocked = bool(
                set(flagged_categories) & self.config.moderation_categories_to_block
            )

            moderation_result = ModerationResult(
                is_flagged=result.flagged,
                is_blocked=is_blocked,
                flagged_categories=flagged_categories,
                category_scores=category_scores
            )

            # Log if flagged
            if result.flagged:
                self.logger.log_event(
                    event_type=SecurityEventType.MODERATION_FLAG,
                    session_id=session_id,
                    details={
                        "is_blocked": is_blocked,
                        "flagged_categories": flagged_categories
                    },
                    severity="warning" if not is_blocked else "error"
                )

            return moderation_result

        except Exception as e:
            self.logger.log_event(
                event_type=SecurityEventType.MODERATION_CHECK,
                session_id=session_id,
                details={"status": "error", "error": str(e)},
                severity="error"
            )
            return ModerationResult(
                is_flagged=False,
                is_blocked=False,
                error=f"Moderation API error: {str(e)}"
            )

    def is_safe(self, text: str) -> bool:
        """
        Quick check if content passes moderation.

        Args:
            text: Content to check

        Returns:
            True if content is safe
        """
        result = self.check(text)
        return not result.is_blocked
