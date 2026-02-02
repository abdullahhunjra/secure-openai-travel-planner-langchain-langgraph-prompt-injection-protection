"""
PII (Personally Identifiable Information) detection module using Presidio.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from config.security_config import SECURITY_CONFIG
from security.logging.security_logger import SecurityLogger, SecurityEventType


@dataclass
class PIIEntity:
    """Detected PII entity."""
    entity_type: str
    text: str
    start: int
    end: int
    score: float


@dataclass
class PIIDetectionResult:
    """Result of PII detection."""
    contains_pii: bool
    entities: List[PIIEntity] = field(default_factory=list)
    anonymized_text: Optional[str] = None
    error: Optional[str] = None


class PIIDetector:
    """
    PII detector using Microsoft Presidio.
    Falls back to regex-based detection if Presidio is not available.
    """

    # Regex patterns for fallback PII detection
    PII_PATTERNS = {
        "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "PHONE": r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        "SSN": r'\b\d{3}-\d{2}-\d{4}\b',
        "CREDIT_CARD": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        "IP_ADDRESS": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        "PASSPORT": r'\b[A-Z]{1,2}\d{6,9}\b',
    }

    def __init__(self, logger: Optional[SecurityLogger] = None):
        self.logger = logger or SecurityLogger()
        self.config = SECURITY_CONFIG.moderation
        self.analyzer = None
        self.anonymizer = None
        self._presidio_available = False

        # Try to initialize Presidio
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine

            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
            self._presidio_available = True
        except ImportError:
            # Presidio not installed, use regex fallback
            import re
            self._compiled_patterns = {
                name: re.compile(pattern)
                for name, pattern in self.PII_PATTERNS.items()
            }

    def detect(self, text: str, session_id: str = "unknown") -> PIIDetectionResult:
        """
        Detect PII in text.

        Args:
            text: Text to analyze
            session_id: Session identifier for logging

        Returns:
            PIIDetectionResult with detection details
        """
        if not self.config.enable_pii_detection:
            return PIIDetectionResult(contains_pii=False)

        if not text:
            return PIIDetectionResult(contains_pii=False)

        if self._presidio_available:
            return self._detect_with_presidio(text, session_id)
        else:
            return self._detect_with_regex(text, session_id)

    def _detect_with_presidio(self, text: str, session_id: str) -> PIIDetectionResult:
        """Detect PII using Presidio."""
        try:
            # Analyze text for PII
            results = self.analyzer.analyze(
                text=text,
                language='en',
                entities=[
                    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
                    "CREDIT_CARD", "US_SSN", "US_PASSPORT",
                    "IP_ADDRESS", "LOCATION"
                ]
            )

            # Filter by score threshold
            filtered_results = [
                r for r in results
                if r.score >= self.config.pii_score_threshold
            ]

            entities = [
                PIIEntity(
                    entity_type=r.entity_type,
                    text=text[r.start:r.end],
                    start=r.start,
                    end=r.end,
                    score=r.score
                )
                for r in filtered_results
            ]

            # Anonymize if PII found
            anonymized_text = None
            if entities:
                anonymized_result = self.anonymizer.anonymize(
                    text=text,
                    analyzer_results=filtered_results
                )
                anonymized_text = anonymized_result.text

                self.logger.log_event(
                    event_type=SecurityEventType.PII_DETECTED,
                    session_id=session_id,
                    details={
                        "entity_types": list(set(e.entity_type for e in entities)),
                        "count": len(entities)
                    },
                    severity="warning"
                )

            return PIIDetectionResult(
                contains_pii=len(entities) > 0,
                entities=entities,
                anonymized_text=anonymized_text
            )

        except Exception as e:
            self.logger.log_event(
                event_type=SecurityEventType.PII_DETECTED,
                session_id=session_id,
                details={"status": "error", "error": str(e)},
                severity="error"
            )
            return PIIDetectionResult(
                contains_pii=False,
                error=f"PII detection error: {str(e)}"
            )

    def _detect_with_regex(self, text: str, session_id: str) -> PIIDetectionResult:
        """Fallback PII detection using regex patterns."""
        import re

        entities = []

        for entity_type, pattern in self._compiled_patterns.items():
            for match in pattern.finditer(text):
                entities.append(PIIEntity(
                    entity_type=entity_type,
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    score=0.8  # Default confidence for regex matches
                ))

        # Simple anonymization by replacing matches
        anonymized_text = None
        if entities:
            anonymized_text = text
            # Sort by position (reverse) to avoid offset issues
            for entity in sorted(entities, key=lambda x: x.start, reverse=True):
                replacement = f"<{entity.entity_type}>"
                anonymized_text = (
                    anonymized_text[:entity.start] +
                    replacement +
                    anonymized_text[entity.end:]
                )

            self.logger.log_event(
                event_type=SecurityEventType.PII_DETECTED,
                session_id=session_id,
                details={
                    "entity_types": list(set(e.entity_type for e in entities)),
                    "count": len(entities),
                    "method": "regex_fallback"
                },
                severity="warning"
            )

        return PIIDetectionResult(
            contains_pii=len(entities) > 0,
            entities=entities,
            anonymized_text=anonymized_text
        )

    def anonymize(self, text: str) -> str:
        """
        Anonymize PII in text.

        Args:
            text: Text to anonymize

        Returns:
            Text with PII redacted
        """
        result = self.detect(text)
        return result.anonymized_text if result.anonymized_text else text

    def contains_pii(self, text: str) -> bool:
        """
        Quick check if text contains PII.

        Args:
            text: Text to check

        Returns:
            True if PII detected
        """
        result = self.detect(text)
        return result.contains_pii
