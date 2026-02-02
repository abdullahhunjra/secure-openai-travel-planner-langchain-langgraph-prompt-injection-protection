"""
Input sanitization module with blocklist patterns and dangerous character removal.
"""

import re
import html
from dataclasses import dataclass
from typing import List, Set, Optional

from config.security_config import SECURITY_CONFIG
from security.logging.security_logger import SecurityLogger, SecurityEventType


@dataclass
class SanitizationResult:
    """Result of input sanitization."""
    sanitized_text: str
    original_text: str
    was_modified: bool
    blocked_patterns_found: List[str]
    is_blocked: bool
    block_reason: Optional[str] = None


class InputSanitizer:
    """
    Input sanitizer with blocklist patterns and dangerous character removal.
    """

    # Blocklist patterns for known attack vectors
    BLOCKLIST_PATTERNS = [
        # Code execution attempts
        r"eval\s*\(",
        r"exec\s*\(",
        r"__import__\s*\(",
        r"subprocess\.",
        r"os\.system",
        r"os\.popen",
        r"import\s+os",
        r"import\s+subprocess",

        # Shell injection
        r";\s*rm\s+-",
        r";\s*cat\s+",
        r"\|\s*bash",
        r"\|\s*sh\b",
        r"`[^`]+`",  # Backtick command execution

        # SQL injection patterns
        r";\s*DROP\s+TABLE",
        r";\s*DELETE\s+FROM",
        r"UNION\s+SELECT",
        r"OR\s+1\s*=\s*1",
        r"'\s*OR\s+'",

        # XSS patterns
        r"<script[^>]*>",
        r"javascript\s*:",
        r"on\w+\s*=",

        # Path traversal
        r"\.\./",
        r"\.\.\\",
    ]

    # Characters to remove or escape
    DANGEROUS_CHARS = {
        '\x00': '',  # Null byte
        '\x0b': '',  # Vertical tab
        '\x0c': '',  # Form feed
        '\x1b': '',  # Escape character
    }

    # Maximum consecutive repeated characters
    MAX_REPEATED_CHARS = 10

    def __init__(self, logger: Optional[SecurityLogger] = None):
        self.logger = logger or SecurityLogger()
        self.config = SECURITY_CONFIG.input_validation
        self.compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.BLOCKLIST_PATTERNS
        ]

    def sanitize(self, text: str, session_id: str = "unknown") -> SanitizationResult:
        """
        Sanitize user input.

        Args:
            text: Input text to sanitize
            session_id: Session identifier for logging

        Returns:
            SanitizationResult with sanitized text and metadata
        """
        if not text:
            return SanitizationResult(
                sanitized_text="",
                original_text="",
                was_modified=False,
                blocked_patterns_found=[],
                is_blocked=False
            )

        original = text
        blocked_patterns = []

        # Check length limits
        if len(text) > self.config.max_input_length:
            self.logger.log_event(
                event_type=SecurityEventType.INPUT_BLOCKED,
                session_id=session_id,
                details={"reason": "input_too_long", "length": len(text)}
            )
            return SanitizationResult(
                sanitized_text="",
                original_text=original,
                was_modified=True,
                blocked_patterns_found=["length_exceeded"],
                is_blocked=True,
                block_reason=f"Input exceeds maximum length of {self.config.max_input_length} characters"
            )

        if len(text) < self.config.min_input_length:
            return SanitizationResult(
                sanitized_text="",
                original_text=original,
                was_modified=True,
                blocked_patterns_found=["length_too_short"],
                is_blocked=True,
                block_reason=f"Input must be at least {self.config.min_input_length} characters"
            )

        # Check for blocklist patterns
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                blocked_patterns.append(pattern.pattern)

        if blocked_patterns:
            self.logger.log_event(
                event_type=SecurityEventType.INPUT_BLOCKED,
                session_id=session_id,
                details={
                    "reason": "blocklist_match",
                    "patterns": blocked_patterns
                },
                severity="warning"
            )
            return SanitizationResult(
                sanitized_text="",
                original_text=original,
                was_modified=True,
                blocked_patterns_found=blocked_patterns,
                is_blocked=True,
                block_reason="Input contains blocked patterns"
            )

        # Remove dangerous characters
        sanitized = text
        for char, replacement in self.DANGEROUS_CHARS.items():
            sanitized = sanitized.replace(char, replacement)

        # Normalize excessive whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized)

        # Remove excessive repeated characters (but keep reasonable repeats)
        sanitized = self._limit_repeated_chars(sanitized)

        # HTML escape to prevent XSS (but preserve normal text)
        # Only escape angle brackets and ampersands
        sanitized = sanitized.replace('&', '&amp;')
        sanitized = sanitized.replace('<', '&lt;')
        sanitized = sanitized.replace('>', '&gt;')

        was_modified = sanitized != original

        if was_modified:
            self.logger.log_event(
                event_type=SecurityEventType.INPUT_SANITIZED,
                session_id=session_id,
                details={"modifications": "character_sanitization"}
            )

        return SanitizationResult(
            sanitized_text=sanitized.strip(),
            original_text=original,
            was_modified=was_modified,
            blocked_patterns_found=[],
            is_blocked=False
        )

    def _limit_repeated_chars(self, text: str) -> str:
        """Limit consecutive repeated characters."""
        result = []
        prev_char = None
        repeat_count = 0

        for char in text:
            if char == prev_char:
                repeat_count += 1
                if repeat_count < self.MAX_REPEATED_CHARS:
                    result.append(char)
            else:
                result.append(char)
                prev_char = char
                repeat_count = 1

        return ''.join(result)

    def is_safe(self, text: str) -> bool:
        """
        Quick check if input is safe (no blocklist matches).

        Args:
            text: Input to check

        Returns:
            True if input appears safe
        """
        if not text or len(text) > self.config.max_input_length:
            return False

        for pattern in self.compiled_patterns:
            if pattern.search(text):
                return False

        return True
