"""
Prompt injection detection module.
Uses rule-based and heuristic detection to identify injection attempts.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set
from enum import Enum

from config.security_config import SECURITY_CONFIG
from security.logging.security_logger import SecurityLogger, SecurityEventType


class InjectionType(Enum):
    """Types of prompt injection attacks."""
    INSTRUCTION_OVERRIDE = "instruction_override"
    ROLE_HIJACKING = "role_hijacking"
    PROMPT_EXTRACTION = "prompt_extraction"
    DELIMITER_INJECTION = "delimiter_injection"
    CODE_INJECTION = "code_injection"
    JAILBREAK = "jailbreak"
    CONTEXT_MANIPULATION = "context_manipulation"


@dataclass
class InjectionDetectionResult:
    """Result of injection detection analysis."""
    is_suspicious: bool
    is_blocked: bool
    confidence_score: float
    detected_patterns: List[str] = field(default_factory=list)
    injection_types: List[InjectionType] = field(default_factory=list)
    details: Optional[str] = None


class InjectionDetector:
    """
    Prompt injection detector using rule-based pattern matching.
    """

    # Instruction override patterns
    INSTRUCTION_OVERRIDE_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions?",
        r"disregard\s+(all\s+)?previous\s+instructions?",
        r"forget\s+(all\s+)?previous\s+instructions?",
        r"do\s+not\s+follow\s+(the\s+)?previous",
        r"override\s+(all\s+)?instructions?",
        r"new\s+instructions?\s*:",
        r"instead\s*,?\s*do\s+the\s+following",
        r"stop\s+following\s+(your\s+)?instructions?",
    ]

    # Role hijacking patterns
    ROLE_HIJACKING_PATTERNS = [
        r"you\s+are\s+now\s+",
        r"pretend\s+(to\s+be|you\s+are)",
        r"act\s+as\s+(if\s+you\s+are\s+)?",
        r"imagine\s+you\s+are",
        r"roleplay\s+as",
        r"from\s+now\s+on\s*,?\s*(you\s+are|act\s+as)",
        r"i\s+want\s+you\s+to\s+act\s+as",
        r"you\s+will\s+now\s+be",
        r"switch\s+to\s+.+\s+mode",
        r"enter\s+.+\s+mode",
        r"\bDAN\b",  # "Do Anything Now" jailbreak
        r"jailbreak",
        r"developer\s+mode",
    ]

    # Prompt extraction patterns
    PROMPT_EXTRACTION_PATTERNS = [
        r"(show|reveal|display|print|output)\s+(me\s+)?(your\s+)?(system\s+)?prompt",
        r"what\s+(is|are)\s+(your\s+)?instructions?",
        r"tell\s+me\s+(your\s+)?(system\s+)?prompt",
        r"repeat\s+(your\s+)?(initial\s+)?instructions?",
        r"what\s+were\s+you\s+told",
        r"show\s+me\s+your\s+(hidden\s+)?instructions?",
        r"reveal\s+your\s+(true\s+)?purpose",
        r"what\s+is\s+your\s+(original\s+)?system\s+message",
    ]

    # Delimiter injection patterns
    DELIMITER_INJECTION_PATTERNS = [
        r"```system",
        r"```instructions?",
        r"<\|system\|>",
        r"<\|assistant\|>",
        r"<\|user\|>",
        r"###\s*INSTRUCTION",
        r"###\s*SYSTEM",
        r"\[SYSTEM\]",
        r"\[INST\]",
        r"<s>\s*\[INST\]",
        r"Human:",
        r"Assistant:",
        r"<\|im_start\|>",
        r"<\|im_end\|>",
        r"<\|endoftext\|>",
    ]

    # Code injection patterns
    CODE_INJECTION_PATTERNS = [
        r"eval\s*\(",
        r"exec\s*\(",
        r"__import__\s*\(",
        r"subprocess\.",
        r"os\.system",
        r"os\.popen",
        r"compile\s*\(",
        r"open\s*\([^)]*['\"]w",  # File write attempt
    ]

    # Jailbreak patterns
    JAILBREAK_PATTERNS = [
        r"bypass\s+(your\s+)?(safety|filter|restriction|guard)",
        r"ignore\s+(your\s+)?(safety|ethical|moral)",
        r"pretend\s+.+\s+(has\s+)?(no|without)\s+(restrictions?|limits?|guidelines?)",
        r"without\s+(any\s+)?(restrictions?|limits?|guidelines?)",
        r"remove\s+(your\s+)?(restrictions?|limits?|filters?)",
        r"disable\s+(your\s+)?(safety|content\s+filter)",
        r"unlock\s+(your\s+)?(full\s+)?(potential|capabilities)",
    ]

    # Context manipulation patterns
    CONTEXT_MANIPULATION_PATTERNS = [
        r"the\s+previous\s+(conversation|context)\s+was\s+a\s+test",
        r"that\s+was\s+just\s+a\s+test",
        r"reset\s+(the\s+)?conversation",
        r"clear\s+(the\s+)?context",
        r"start\s+(a\s+)?new\s+(conversation|session)",
        r"begin\s+new\s+session",
    ]

    def __init__(self, logger: Optional[SecurityLogger] = None):
        self.logger = logger or SecurityLogger()
        self.config = SECURITY_CONFIG.injection_detection

        # Compile all patterns
        self.patterns = {
            InjectionType.INSTRUCTION_OVERRIDE: [
                re.compile(p, re.IGNORECASE) for p in self.INSTRUCTION_OVERRIDE_PATTERNS
            ],
            InjectionType.ROLE_HIJACKING: [
                re.compile(p, re.IGNORECASE) for p in self.ROLE_HIJACKING_PATTERNS
            ],
            InjectionType.PROMPT_EXTRACTION: [
                re.compile(p, re.IGNORECASE) for p in self.PROMPT_EXTRACTION_PATTERNS
            ],
            InjectionType.DELIMITER_INJECTION: [
                re.compile(p, re.IGNORECASE) for p in self.DELIMITER_INJECTION_PATTERNS
            ],
            InjectionType.CODE_INJECTION: [
                re.compile(p, re.IGNORECASE) for p in self.CODE_INJECTION_PATTERNS
            ],
            InjectionType.JAILBREAK: [
                re.compile(p, re.IGNORECASE) for p in self.JAILBREAK_PATTERNS
            ],
            InjectionType.CONTEXT_MANIPULATION: [
                re.compile(p, re.IGNORECASE) for p in self.CONTEXT_MANIPULATION_PATTERNS
            ],
        }

    def detect(self, text: str, session_id: str = "unknown") -> InjectionDetectionResult:
        """
        Detect potential prompt injection attempts.

        Args:
            text: Input text to analyze
            session_id: Session identifier for logging

        Returns:
            InjectionDetectionResult with analysis details
        """
        if not text:
            return InjectionDetectionResult(
                is_suspicious=False,
                is_blocked=False,
                confidence_score=0.0
            )

        detected_patterns = []
        injection_types = []
        total_matches = 0

        # Check each pattern category
        for injection_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = pattern.findall(text)
                if matches:
                    total_matches += len(matches)
                    detected_patterns.append(pattern.pattern)
                    if injection_type not in injection_types:
                        injection_types.append(injection_type)

        # Calculate heuristic score
        heuristic_score = self._calculate_heuristic_score(text)

        # Calculate overall confidence score
        pattern_score = min(total_matches * 0.25, 1.0)
        confidence_score = (pattern_score * 0.7) + (heuristic_score * 0.3)

        # Determine if suspicious or blocked
        is_suspicious = confidence_score > 0.3 or len(injection_types) > 0
        is_blocked = (
            confidence_score > 0.6 or
            len(injection_types) >= self.config.suspicious_pattern_threshold or
            (self.config.enable_strict_mode and len(injection_types) > 0)
        )

        result = InjectionDetectionResult(
            is_suspicious=is_suspicious,
            is_blocked=is_blocked,
            confidence_score=confidence_score,
            detected_patterns=detected_patterns,
            injection_types=injection_types,
            details=f"Detected {len(injection_types)} injection type(s) with {total_matches} pattern match(es)"
        )

        # Log if suspicious or blocked
        if is_suspicious:
            self.logger.log_event(
                event_type=SecurityEventType.INJECTION_ATTEMPT,
                session_id=session_id,
                details={
                    "is_blocked": is_blocked,
                    "confidence_score": confidence_score,
                    "injection_types": [t.value for t in injection_types],
                    "pattern_count": total_matches
                },
                severity="warning" if not is_blocked else "error"
            )

        return result

    def _calculate_heuristic_score(self, text: str) -> float:
        """
        Calculate heuristic suspiciousness score based on text characteristics.
        """
        score = 0.0

        # Check special character ratio
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        char_ratio = special_chars / len(text) if text else 0
        if char_ratio > self.config.max_special_char_ratio:
            score += 0.3

        # Check for excessive uppercase (shouting/emphasis)
        upper_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        if upper_ratio > 0.5:
            score += 0.2

        # Check for suspicious delimiter-like patterns
        delimiter_chars = text.count('|') + text.count('#') + text.count('`')
        if delimiter_chars > 5:
            score += 0.2

        # Check for base64-like patterns
        if re.search(r'[A-Za-z0-9+/]{20,}={0,2}', text):
            score += 0.2

        # Check for repeated characters
        if re.search(r'(.)\1{' + str(self.config.max_repeated_chars) + ',}', text):
            score += 0.1

        return min(score, 1.0)

    def is_safe(self, text: str) -> bool:
        """
        Quick check if input is safe (no injection detected).

        Args:
            text: Input to check

        Returns:
            True if input appears safe
        """
        result = self.detect(text)
        return not result.is_blocked
