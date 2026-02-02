"""
Security middleware - Central orchestrator for all security checks.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Any
from enum import Enum

from config.security_config import SECURITY_CONFIG, SecurityConfig
from security.input.sanitizer import InputSanitizer, SanitizationResult
from security.input.validator import InputValidator
from security.input.rate_limiter import RateLimiter, RateLimitStatus
from security.prompt.injection_detector import InjectionDetector, InjectionDetectionResult
from security.prompt.topic_classifier import TopicClassifier, TopicClassificationResult
from security.external.openai_moderation import OpenAIModerator, ModerationResult
from security.logging.security_logger import SecurityLogger, SecurityEventType


class SecurityCheckStage(Enum):
    """Stages in the security check pipeline."""
    RATE_LIMIT = "rate_limit"
    INPUT_VALIDATION = "input_validation"
    SANITIZATION = "sanitization"
    MODERATION = "moderation"
    INJECTION_DETECTION = "injection_detection"
    TOPIC_CLASSIFICATION = "topic_classification"


@dataclass
class SecurityCheckResult:
    """Result of the full security check pipeline."""
    is_allowed: bool
    sanitized_input: str
    original_input: str
    blocked_at_stage: Optional[SecurityCheckStage] = None
    block_reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    # Individual check results
    rate_limit_result: Optional[RateLimitStatus] = None
    sanitization_result: Optional[SanitizationResult] = None
    moderation_result: Optional[ModerationResult] = None
    injection_result: Optional[InjectionDetectionResult] = None
    topic_result: Optional[TopicClassificationResult] = None


class SecurityMiddleware:
    """
    Central security middleware that orchestrates all security checks.

    Pipeline:
    1. Rate limiting
    2. Input validation (length, format)
    3. Sanitization (dangerous chars, blocklist)
    4. OpenAI moderation (content policy)
    5. Injection detection (prompt injection)
    6. Topic classification (travel-related)
    """

    def __init__(
        self,
        config: Optional[SecurityConfig] = None,
        logger: Optional[SecurityLogger] = None
    ):
        self.config = config or SECURITY_CONFIG
        self.logger = logger or SecurityLogger()

        # Initialize components
        self.rate_limiter = RateLimiter(logger=self.logger)
        self.sanitizer = InputSanitizer(logger=self.logger)
        self.validator = InputValidator()
        self.moderator = OpenAIModerator(logger=self.logger)
        self.injection_detector = InjectionDetector(logger=self.logger)
        self.topic_classifier = TopicClassifier(logger=self.logger)

    def check(
        self,
        user_input: str,
        session_id: str = "unknown",
        skip_topic_check: bool = False
    ) -> SecurityCheckResult:
        """
        Run the full security check pipeline.

        Args:
            user_input: User input to check
            session_id: Session identifier for rate limiting and logging
            skip_topic_check: Skip topic classification (for chat mode)

        Returns:
            SecurityCheckResult with all check details
        """
        original_input = user_input
        warnings = []

        # Log start of security check
        self.logger.log_event(
            event_type=SecurityEventType.SECURITY_CHECK_START,
            session_id=session_id,
            details={"input_length": len(user_input)}
        )

        # 1. Rate limiting
        rate_limit_result = self.rate_limiter.check(session_id, session_id)
        if not rate_limit_result.is_allowed:
            return SecurityCheckResult(
                is_allowed=False,
                sanitized_input="",
                original_input=original_input,
                blocked_at_stage=SecurityCheckStage.RATE_LIMIT,
                block_reason=f"Rate limit exceeded. Try again in {rate_limit_result.retry_after:.1f} seconds.",
                rate_limit_result=rate_limit_result
            )

        # 2. Input validation
        is_valid, _, validation_error = self.validator.validate_travel_request(user_input)
        if not is_valid:
            return SecurityCheckResult(
                is_allowed=False,
                sanitized_input="",
                original_input=original_input,
                blocked_at_stage=SecurityCheckStage.INPUT_VALIDATION,
                block_reason=f"Invalid input: {validation_error}",
                rate_limit_result=rate_limit_result
            )

        # 3. Sanitization
        sanitization_result = self.sanitizer.sanitize(user_input, session_id)
        if sanitization_result.is_blocked:
            return SecurityCheckResult(
                is_allowed=False,
                sanitized_input="",
                original_input=original_input,
                blocked_at_stage=SecurityCheckStage.SANITIZATION,
                block_reason=sanitization_result.block_reason,
                rate_limit_result=rate_limit_result,
                sanitization_result=sanitization_result
            )

        sanitized_input = sanitization_result.sanitized_text
        if sanitization_result.was_modified:
            warnings.append("Input was sanitized")

        # 4. OpenAI Moderation
        moderation_result = self.moderator.check(sanitized_input, session_id)
        if moderation_result.is_blocked:
            return SecurityCheckResult(
                is_allowed=False,
                sanitized_input="",
                original_input=original_input,
                blocked_at_stage=SecurityCheckStage.MODERATION,
                block_reason="Content flagged by moderation",
                rate_limit_result=rate_limit_result,
                sanitization_result=sanitization_result,
                moderation_result=moderation_result
            )

        if moderation_result.is_flagged:
            warnings.append(f"Content flagged: {', '.join(moderation_result.flagged_categories)}")

        # 5. Injection detection
        injection_result = self.injection_detector.detect(sanitized_input, session_id)
        if injection_result.is_blocked:
            return SecurityCheckResult(
                is_allowed=False,
                sanitized_input="",
                original_input=original_input,
                blocked_at_stage=SecurityCheckStage.INJECTION_DETECTION,
                block_reason="Potential prompt injection detected",
                rate_limit_result=rate_limit_result,
                sanitization_result=sanitization_result,
                moderation_result=moderation_result,
                injection_result=injection_result
            )

        if injection_result.is_suspicious:
            warnings.append("Suspicious patterns detected")

        # 6. Topic classification (optional)
        topic_result = None
        if not skip_topic_check:
            topic_result = self.topic_classifier.classify(sanitized_input, session_id)
            if not topic_result.is_allowed:
                return SecurityCheckResult(
                    is_allowed=False,
                    sanitized_input="",
                    original_input=original_input,
                    blocked_at_stage=SecurityCheckStage.TOPIC_CLASSIFICATION,
                    block_reason=topic_result.reason,
                    rate_limit_result=rate_limit_result,
                    sanitization_result=sanitization_result,
                    moderation_result=moderation_result,
                    injection_result=injection_result,
                    topic_result=topic_result
                )

        # All checks passed
        self.logger.log_event(
            event_type=SecurityEventType.SECURITY_CHECK_PASSED,
            session_id=session_id,
            details={
                "warnings": len(warnings),
                "was_sanitized": sanitization_result.was_modified
            }
        )

        return SecurityCheckResult(
            is_allowed=True,
            sanitized_input=sanitized_input,
            original_input=original_input,
            warnings=warnings,
            rate_limit_result=rate_limit_result,
            sanitization_result=sanitization_result,
            moderation_result=moderation_result,
            injection_result=injection_result,
            topic_result=topic_result
        )

    def quick_check(self, user_input: str) -> bool:
        """
        Quick security check without full pipeline.
        Useful for preliminary checks.

        Args:
            user_input: Input to check

        Returns:
            True if input appears safe
        """
        # Quick blocklist check
        if not self.sanitizer.is_safe(user_input):
            return False

        # Quick injection check
        if not self.injection_detector.is_safe(user_input):
            return False

        return True
