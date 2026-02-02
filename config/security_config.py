"""
Security configuration settings for the AI Travel Planner.
Contains thresholds, limits, and feature flags for security components.
"""

from dataclasses import dataclass, field
from typing import List, Set
import os


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    requests_per_minute: int = 20
    requests_per_hour: int = 100
    burst_limit: int = 5
    window_seconds: int = 60


@dataclass
class InputValidationConfig:
    """Input validation settings."""
    max_input_length: int = 5000
    max_city_length: int = 100
    max_preferences_length: int = 500
    min_input_length: int = 3
    allowed_date_formats: List[str] = field(default_factory=lambda: [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"
    ])


@dataclass
class InjectionDetectionConfig:
    """Prompt injection detection thresholds."""
    max_special_char_ratio: float = 0.3
    max_repeated_chars: int = 5
    suspicious_pattern_threshold: int = 2
    enable_strict_mode: bool = True


@dataclass
class ModerationConfig:
    """Content moderation settings."""
    enable_openai_moderation: bool = True
    enable_pii_detection: bool = True
    pii_score_threshold: float = 0.7
    moderation_categories_to_block: Set[str] = field(default_factory=lambda: {
        "hate", "hate/threatening", "self-harm", "sexual/minors",
        "violence", "violence/graphic"
    })


@dataclass
class TopicClassificationConfig:
    """Topic classification settings."""
    enable_topic_filter: bool = True
    allowed_topics: Set[str] = field(default_factory=lambda: {
        "travel", "vacation", "trip", "flight", "hotel", "destination",
        "tourism", "booking", "itinerary", "accommodation"
    })
    confidence_threshold: float = 0.6


@dataclass
class LoggingConfig:
    """Security logging configuration."""
    log_level: str = "INFO"
    log_file: str = "logs/security.log"
    enable_console_logging: bool = True
    hash_sensitive_data: bool = True
    max_log_size_mb: int = 100
    backup_count: int = 5


@dataclass
class SecurityConfig:
    """Main security configuration container."""
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    input_validation: InputValidationConfig = field(default_factory=InputValidationConfig)
    injection_detection: InjectionDetectionConfig = field(default_factory=InjectionDetectionConfig)
    moderation: ModerationConfig = field(default_factory=ModerationConfig)
    topic_classification: TopicClassificationConfig = field(default_factory=TopicClassificationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Feature flags
    enable_security_middleware: bool = True
    enable_output_validation: bool = True
    fail_open: bool = False  # If True, allow requests when security checks fail


def get_security_config() -> SecurityConfig:
    """
    Get security configuration, optionally overriding from environment variables.
    """
    config = SecurityConfig()

    # Override from environment variables if present
    if os.getenv("SECURITY_RATE_LIMIT_PER_MIN"):
        config.rate_limit.requests_per_minute = int(os.getenv("SECURITY_RATE_LIMIT_PER_MIN"))

    if os.getenv("SECURITY_MAX_INPUT_LENGTH"):
        config.input_validation.max_input_length = int(os.getenv("SECURITY_MAX_INPUT_LENGTH"))

    if os.getenv("SECURITY_STRICT_MODE"):
        config.injection_detection.enable_strict_mode = os.getenv("SECURITY_STRICT_MODE").lower() == "true"

    if os.getenv("SECURITY_ENABLE_MODERATION"):
        config.moderation.enable_openai_moderation = os.getenv("SECURITY_ENABLE_MODERATION").lower() == "true"

    if os.getenv("SECURITY_ENABLE_PII"):
        config.moderation.enable_pii_detection = os.getenv("SECURITY_ENABLE_PII").lower() == "true"

    if os.getenv("SECURITY_FAIL_OPEN"):
        config.fail_open = os.getenv("SECURITY_FAIL_OPEN").lower() == "true"

    return config


# Global configuration instance
SECURITY_CONFIG = get_security_config()
