"""
Safe parser module to replace dangerous eval() calls.
Uses json.loads with ast.literal_eval as fallback, plus Pydantic validation.
"""

import json
import ast
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type, TypeVar
from pydantic import BaseModel, ValidationError

from security.logging.security_logger import SecurityLogger, SecurityEventType


T = TypeVar('T', bound=BaseModel)


@dataclass
class ParseResult:
    """Result of a safe parsing operation."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    method_used: Optional[str] = None


class SafeParser:
    """
    Safe parser for LLM outputs.
    Replaces eval() with secure alternatives.
    """

    def __init__(self, logger: Optional[SecurityLogger] = None):
        self.logger = logger or SecurityLogger()

    def parse_json(self, text: str, session_id: str = "unknown") -> ParseResult:
        """
        Safely parse JSON/dict string from LLM output.

        Args:
            text: The text to parse (expected to be JSON or Python dict literal)
            session_id: Session identifier for logging

        Returns:
            ParseResult with parsed data or error information
        """
        if not text or not isinstance(text, str):
            return ParseResult(
                success=False,
                error="Empty or invalid input"
            )

        # Clean up the text - extract JSON from markdown code blocks if present
        cleaned_text = self._extract_json_from_text(text)

        # Try json.loads first (safest)
        try:
            data = json.loads(cleaned_text)
            self.logger.log_event(
                event_type=SecurityEventType.OUTPUT_VALIDATION,
                session_id=session_id,
                details={"method": "json.loads", "success": True}
            )
            return ParseResult(
                success=True,
                data=data,
                method_used="json.loads"
            )
        except json.JSONDecodeError:
            pass

        # Try ast.literal_eval as fallback (safe for Python literals)
        try:
            data = ast.literal_eval(cleaned_text)
            if isinstance(data, dict):
                self.logger.log_event(
                    event_type=SecurityEventType.OUTPUT_VALIDATION,
                    session_id=session_id,
                    details={"method": "ast.literal_eval", "success": True}
                )
                return ParseResult(
                    success=True,
                    data=data,
                    method_used="ast.literal_eval"
                )
            else:
                return ParseResult(
                    success=False,
                    error=f"Parsed result is not a dict: {type(data).__name__}"
                )
        except (ValueError, SyntaxError) as e:
            self.logger.log_event(
                event_type=SecurityEventType.OUTPUT_VALIDATION,
                session_id=session_id,
                details={"method": "all_failed", "error": str(e)},
                severity="warning"
            )
            return ParseResult(
                success=False,
                error=f"Failed to parse: {str(e)}"
            )

    def parse_and_validate(
        self,
        text: str,
        schema: Type[T],
        session_id: str = "unknown"
    ) -> tuple[bool, Optional[T], Optional[str]]:
        """
        Parse text and validate against a Pydantic schema.

        Args:
            text: The text to parse
            schema: Pydantic model class for validation
            session_id: Session identifier for logging

        Returns:
            Tuple of (success, validated_model, error_message)
        """
        # First, parse the text
        parse_result = self.parse_json(text, session_id)

        if not parse_result.success:
            return False, None, parse_result.error

        # Validate against schema
        try:
            validated = schema.model_validate(parse_result.data)
            self.logger.log_event(
                event_type=SecurityEventType.OUTPUT_VALIDATION,
                session_id=session_id,
                details={
                    "schema": schema.__name__,
                    "validation": "passed"
                }
            )
            return True, validated, None
        except ValidationError as e:
            error_msg = str(e)
            self.logger.log_event(
                event_type=SecurityEventType.OUTPUT_VALIDATION,
                session_id=session_id,
                details={
                    "schema": schema.__name__,
                    "validation": "failed",
                    "error": error_msg
                },
                severity="warning"
            )
            return False, None, f"Validation failed: {error_msg}"

    def _extract_json_from_text(self, text: str) -> str:
        """
        Extract JSON from text, handling markdown code blocks and other wrappers.
        """
        text = text.strip()

        # Remove markdown code blocks
        code_block_pattern = r'```(?:json|python)?\s*\n?(.*?)\n?```'
        match = re.search(code_block_pattern, text, re.DOTALL)
        if match:
            text = match.group(1).strip()

        # Try to find JSON object boundaries
        if not text.startswith('{'):
            start_idx = text.find('{')
            if start_idx != -1:
                # Find matching closing brace
                brace_count = 0
                for i, char in enumerate(text[start_idx:], start_idx):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            text = text[start_idx:i+1]
                            break

        return text

    @staticmethod
    def safe_get(data: Dict[str, Any], key: str, default: Any = "") -> Any:
        """
        Safely get a value from a dictionary with type checking.

        Args:
            data: Dictionary to get value from
            key: Key to look up
            default: Default value if key not found

        Returns:
            Value from dictionary or default
        """
        if not isinstance(data, dict):
            return default
        return data.get(key, default)
