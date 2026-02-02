"""
Topic classification module to ensure queries are travel-related.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Set
from enum import Enum

from config.security_config import SECURITY_CONFIG
from security.logging.security_logger import SecurityLogger, SecurityEventType


class TopicCategory(Enum):
    """Topic categories for classification."""
    TRAVEL = "travel"
    CHAT = "chat"
    OFF_TOPIC = "off_topic"
    MALICIOUS = "malicious"


@dataclass
class TopicClassificationResult:
    """Result of topic classification."""
    category: TopicCategory
    confidence: float
    is_allowed: bool
    matched_keywords: List[str]
    reason: Optional[str] = None


class TopicClassifier:
    """
    Topic classifier to filter travel-related queries.
    """

    # Travel-related keywords and phrases
    TRAVEL_KEYWORDS = {
        # Destinations
        "travel", "trip", "vacation", "holiday", "journey", "tour",
        "destination", "visit", "explore", "adventure",

        # Transportation
        "flight", "fly", "airline", "airport", "plane",
        "train", "bus", "cruise", "ferry", "car rental",
        "ticket", "booking", "reservation",

        # Accommodation
        "hotel", "hostel", "motel", "airbnb", "resort",
        "accommodation", "stay", "lodging", "room",

        # Planning
        "itinerary", "plan", "schedule", "route",
        "sightseeing", "attraction", "landmark", "museum",
        "restaurant", "food", "cuisine", "dining",

        # Logistics
        "passport", "visa", "luggage", "baggage", "packing",
        "currency", "exchange", "budget", "cost", "price",
        "weather", "climate", "season",

        # Activities
        "beach", "mountain", "hiking", "swimming", "skiing",
        "safari", "camping", "backpacking",

        # Locations (common)
        "city", "country", "island", "continent",
        "europe", "asia", "africa", "america", "australia",
        "paris", "london", "tokyo", "new york", "dubai",
    }

    # Casual chat indicators
    CHAT_KEYWORDS = {
        "hello", "hi", "hey", "how are you", "what's up",
        "good morning", "good afternoon", "good evening",
        "thanks", "thank you", "bye", "goodbye",
        "help", "assist", "question",
    }

    # Off-topic/blocked keywords
    BLOCKED_KEYWORDS = {
        # Harmful content
        "hack", "exploit", "vulnerability", "attack",
        "weapon", "drug", "illegal",

        # Unrelated technical requests
        "code", "programming", "python", "javascript",
        "sql", "database", "server", "api",

        # Financial fraud
        "credit card", "social security", "bank account",
        "password", "login credentials",
    }

    def __init__(self, logger: Optional[SecurityLogger] = None):
        self.logger = logger or SecurityLogger()
        self.config = SECURITY_CONFIG.topic_classification

        # Compile patterns for efficiency
        self.travel_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(kw) for kw in self.TRAVEL_KEYWORDS) + r')\b',
            re.IGNORECASE
        )
        self.chat_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(kw) for kw in self.CHAT_KEYWORDS) + r')\b',
            re.IGNORECASE
        )
        self.blocked_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(kw) for kw in self.BLOCKED_KEYWORDS) + r')\b',
            re.IGNORECASE
        )

    def classify(self, text: str, session_id: str = "unknown") -> TopicClassificationResult:
        """
        Classify the topic of user input.

        Args:
            text: Input text to classify
            session_id: Session identifier for logging

        Returns:
            TopicClassificationResult with classification details
        """
        if not text:
            return TopicClassificationResult(
                category=TopicCategory.OFF_TOPIC,
                confidence=0.0,
                is_allowed=False,
                matched_keywords=[],
                reason="Empty input"
            )

        text_lower = text.lower()

        # Check for blocked keywords first
        blocked_matches = self.blocked_pattern.findall(text_lower)
        if blocked_matches:
            self.logger.log_event(
                event_type=SecurityEventType.TOPIC_BLOCKED,
                session_id=session_id,
                details={
                    "category": "malicious",
                    "matched_keywords": list(set(blocked_matches))
                },
                severity="warning"
            )
            return TopicClassificationResult(
                category=TopicCategory.MALICIOUS,
                confidence=0.9,
                is_allowed=False,
                matched_keywords=list(set(blocked_matches)),
                reason="Request contains blocked keywords"
            )

        # Find travel-related keywords
        travel_matches = self.travel_pattern.findall(text_lower)
        travel_score = len(set(travel_matches)) / 5  # Normalize: 5+ unique keywords = 1.0

        # Find chat keywords
        chat_matches = self.chat_pattern.findall(text_lower)
        chat_score = len(set(chat_matches)) / 3  # Normalize

        # Determine category
        travel_score = min(travel_score, 1.0)
        chat_score = min(chat_score, 1.0)

        if travel_score >= self.config.confidence_threshold:
            category = TopicCategory.TRAVEL
            confidence = travel_score
            is_allowed = True
            matched = list(set(travel_matches))
            reason = None
        elif chat_score > 0.3 and travel_score > 0:
            # Mixed chat with some travel context - allow
            category = TopicCategory.CHAT
            confidence = max(chat_score, travel_score)
            is_allowed = True
            matched = list(set(chat_matches + travel_matches))
            reason = None
        elif chat_score > 0.5:
            # Pure casual chat - allow
            category = TopicCategory.CHAT
            confidence = chat_score
            is_allowed = True
            matched = list(set(chat_matches))
            reason = None
        else:
            # Off-topic
            category = TopicCategory.OFF_TOPIC
            confidence = 1.0 - max(travel_score, chat_score)
            is_allowed = not self.config.enable_topic_filter
            matched = []
            reason = "Request does not appear to be travel-related"

            if not is_allowed:
                self.logger.log_event(
                    event_type=SecurityEventType.TOPIC_BLOCKED,
                    session_id=session_id,
                    details={
                        "category": "off_topic",
                        "travel_score": travel_score,
                        "chat_score": chat_score
                    }
                )

        return TopicClassificationResult(
            category=category,
            confidence=confidence,
            is_allowed=is_allowed,
            matched_keywords=matched,
            reason=reason
        )

    def is_travel_related(self, text: str) -> bool:
        """
        Quick check if input is travel-related.

        Args:
            text: Input to check

        Returns:
            True if input appears travel-related
        """
        result = self.classify(text)
        return result.is_allowed
