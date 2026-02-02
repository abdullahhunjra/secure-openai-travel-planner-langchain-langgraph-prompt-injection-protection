"""
Token bucket rate limiter for API request throttling.
"""

import time
from dataclasses import dataclass
from typing import Dict, Optional
from threading import Lock

from config.security_config import SECURITY_CONFIG
from security.logging.security_logger import SecurityLogger, SecurityEventType


@dataclass
class RateLimitStatus:
    """Status of rate limit check."""
    is_allowed: bool
    remaining_requests: int
    reset_time: float
    retry_after: Optional[float] = None


class TokenBucket:
    """Token bucket implementation for rate limiting."""

    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.

        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = Lock()

    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from the bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed, False if insufficient
        """
        with self.lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate

        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now

    @property
    def available_tokens(self) -> float:
        """Get current available tokens."""
        with self.lock:
            self._refill()
            return self.tokens

    def time_until_available(self, tokens: int = 1) -> float:
        """
        Calculate time until requested tokens are available.

        Args:
            tokens: Number of tokens needed

        Returns:
            Seconds until tokens available (0 if available now)
        """
        with self.lock:
            self._refill()
            if self.tokens >= tokens:
                return 0.0

            needed = tokens - self.tokens
            return needed / self.refill_rate


class RateLimiter:
    """
    Rate limiter for controlling request frequency per session/IP.
    """

    def __init__(self, logger: Optional[SecurityLogger] = None):
        self.logger = logger or SecurityLogger()
        self.config = SECURITY_CONFIG.rate_limit
        self.buckets: Dict[str, TokenBucket] = {}
        self.lock = Lock()

        # Calculate refill rate (tokens per second)
        self.refill_rate = self.config.requests_per_minute / 60.0

    def check(self, identifier: str, session_id: str = "unknown") -> RateLimitStatus:
        """
        Check if request is allowed under rate limit.

        Args:
            identifier: Unique identifier (session ID, IP, etc.)
            session_id: Session identifier for logging

        Returns:
            RateLimitStatus with limit details
        """
        bucket = self._get_bucket(identifier)

        if bucket.consume(1):
            return RateLimitStatus(
                is_allowed=True,
                remaining_requests=int(bucket.available_tokens),
                reset_time=time.time() + self.config.window_seconds
            )

        # Rate limited
        retry_after = bucket.time_until_available(1)

        self.logger.log_event(
            event_type=SecurityEventType.RATE_LIMIT,
            session_id=session_id,
            details={
                "identifier": identifier[:20] + "..." if len(identifier) > 20 else identifier,
                "retry_after": retry_after
            },
            severity="warning"
        )

        return RateLimitStatus(
            is_allowed=False,
            remaining_requests=0,
            reset_time=time.time() + retry_after,
            retry_after=retry_after
        )

    def _get_bucket(self, identifier: str) -> TokenBucket:
        """Get or create token bucket for identifier."""
        with self.lock:
            if identifier not in self.buckets:
                self.buckets[identifier] = TokenBucket(
                    capacity=self.config.burst_limit,
                    refill_rate=self.refill_rate
                )

            # Cleanup old buckets periodically
            if len(self.buckets) > 1000:
                self._cleanup_old_buckets()

            return self.buckets[identifier]

    def _cleanup_old_buckets(self):
        """Remove buckets that haven't been used recently."""
        current_time = time.time()
        timeout = self.config.window_seconds * 2

        # Find buckets to remove
        to_remove = []
        for identifier, bucket in self.buckets.items():
            if current_time - bucket.last_refill > timeout:
                to_remove.append(identifier)

        # Remove old buckets
        for identifier in to_remove:
            del self.buckets[identifier]

    def is_allowed(self, identifier: str) -> bool:
        """
        Quick check if request is allowed.

        Args:
            identifier: Unique identifier

        Returns:
            True if allowed
        """
        return self.check(identifier).is_allowed

    def reset(self, identifier: str):
        """
        Reset rate limit for identifier.

        Args:
            identifier: Identifier to reset
        """
        with self.lock:
            if identifier in self.buckets:
                del self.buckets[identifier]
