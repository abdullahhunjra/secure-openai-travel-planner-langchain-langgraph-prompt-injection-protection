"""
Structured security event logging for monitoring and SIEM integration.
"""

import os
import json
import hashlib
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, Dict, Any
from pathlib import Path

from config.security_config import SECURITY_CONFIG


class SecurityEventType(Enum):
    """Types of security events."""
    # Input events
    INPUT_SANITIZED = "input_sanitized"
    INPUT_BLOCKED = "input_blocked"

    # Rate limiting
    RATE_LIMIT = "rate_limit"

    # Injection detection
    INJECTION_ATTEMPT = "injection_attempt"

    # Moderation
    MODERATION_CHECK = "moderation_check"
    MODERATION_FLAG = "moderation_flag"

    # Topic classification
    TOPIC_BLOCKED = "topic_blocked"

    # Output validation
    OUTPUT_VALIDATION = "output_validation"

    # PII detection
    PII_DETECTED = "pii_detected"

    # Pipeline events
    SECURITY_CHECK_START = "security_check_start"
    SECURITY_CHECK_PASSED = "security_check_passed"
    SECURITY_CHECK_FAILED = "security_check_failed"

    # LangGraph events
    SECURITY_GATE_PASSED = "security_gate_passed"
    SECURITY_GATE_BLOCKED = "security_gate_blocked"


@dataclass
class SecurityEvent:
    """Structured security event for logging."""
    timestamp: str
    event_type: str
    session_id: str
    severity: str
    details: Dict[str, Any]
    input_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class SecurityLogger:
    """
    Structured security logger for monitoring and SIEM integration.
    """

    def __init__(self):
        self.config = SECURITY_CONFIG.logging
        self._logger = None
        self._setup_logger()

    def _setup_logger(self):
        """Set up the Python logger."""
        self._logger = logging.getLogger("security")
        self._logger.setLevel(getattr(logging, self.config.log_level.upper()))

        # Avoid duplicate handlers
        if self._logger.handlers:
            return

        # Create formatter for JSON output
        formatter = logging.Formatter('%(message)s')

        # Console handler
        if self.config.enable_console_logging:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)

        # File handler
        try:
            log_dir = Path(self.config.log_file).parent
            log_dir.mkdir(parents=True, exist_ok=True)

            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                self.config.log_file,
                maxBytes=self.config.max_log_size_mb * 1024 * 1024,
                backupCount=self.config.backup_count
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)
        except Exception:
            # If file logging fails, continue with console only
            pass

    def log_event(
        self,
        event_type: SecurityEventType,
        session_id: str,
        details: Optional[Dict[str, Any]] = None,
        severity: str = "info",
        input_text: Optional[str] = None
    ):
        """
        Log a security event.

        Args:
            event_type: Type of security event
            session_id: Session identifier
            details: Additional event details
            severity: Event severity (info, warning, error)
            input_text: Optional input text (will be hashed)
        """
        # Hash input text if provided and configured
        input_hash = None
        if input_text and self.config.hash_sensitive_data:
            input_hash = self._hash_input(input_text)

        event = SecurityEvent(
            timestamp=datetime.utcnow().isoformat() + "Z",
            event_type=event_type.value,
            session_id=session_id,
            severity=severity,
            details=details or {},
            input_hash=input_hash
        )

        # Log based on severity
        log_method = getattr(self._logger, severity.lower(), self._logger.info)
        log_method(event.to_json())

    def _hash_input(self, text: str) -> str:
        """
        Create a hash of input text for logging without exposing content.

        Args:
            text: Text to hash

        Returns:
            SHA-256 hash prefix
        """
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def log_injection_attempt(
        self,
        session_id: str,
        injection_types: list,
        confidence_score: float,
        was_blocked: bool
    ):
        """Log a prompt injection attempt."""
        self.log_event(
            event_type=SecurityEventType.INJECTION_ATTEMPT,
            session_id=session_id,
            details={
                "injection_types": injection_types,
                "confidence_score": confidence_score,
                "was_blocked": was_blocked
            },
            severity="warning" if not was_blocked else "error"
        )

    def log_rate_limit(
        self,
        session_id: str,
        identifier: str,
        retry_after: float
    ):
        """Log a rate limit event."""
        self.log_event(
            event_type=SecurityEventType.RATE_LIMIT,
            session_id=session_id,
            details={
                "identifier_hash": self._hash_input(identifier),
                "retry_after": retry_after
            },
            severity="warning"
        )

    def log_moderation_flag(
        self,
        session_id: str,
        flagged_categories: list,
        was_blocked: bool
    ):
        """Log a content moderation flag."""
        self.log_event(
            event_type=SecurityEventType.MODERATION_FLAG,
            session_id=session_id,
            details={
                "flagged_categories": flagged_categories,
                "was_blocked": was_blocked
            },
            severity="warning" if not was_blocked else "error"
        )

    def log_pii_detection(
        self,
        session_id: str,
        entity_types: list,
        count: int
    ):
        """Log PII detection event."""
        self.log_event(
            event_type=SecurityEventType.PII_DETECTED,
            session_id=session_id,
            details={
                "entity_types": entity_types,
                "count": count
            },
            severity="warning"
        )

    def log_security_gate(
        self,
        session_id: str,
        passed: bool,
        block_reason: Optional[str] = None
    ):
        """Log LangGraph security gate result."""
        event_type = (
            SecurityEventType.SECURITY_GATE_PASSED
            if passed
            else SecurityEventType.SECURITY_GATE_BLOCKED
        )
        self.log_event(
            event_type=event_type,
            session_id=session_id,
            details={
                "passed": passed,
                "block_reason": block_reason
            },
            severity="info" if passed else "warning"
        )
