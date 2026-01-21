"""Prompt sanitization utilities for LLM security.

This module provides functions to sanitize user input before including it in LLM prompts
to prevent prompt injection attacks and ensure prompt integrity.
"""

import json
import re
from typing import Any

from .logging import get_logger

logger = get_logger(__name__)


class PromptSanitizationError(Exception):
    """Raised when prompt sanitization fails."""

    pass


def sanitize_string_for_prompt(text: str, max_length: int = 10000) -> str:
    """Sanitize a string for safe inclusion in LLM prompts.

    This function:
    - Truncates overly long strings
    - Escapes control characters
    - Removes or escapes prompt injection patterns
    - Preserves printable characters

    Args:
        text: String to sanitize
        max_length: Maximum length (default: 10000)

    Returns:
        Sanitized string safe for prompt inclusion
    """
    if not isinstance(text, str):
        # Convert to string representation
        text = str(text)

    # Truncate if too long
    if len(text) > max_length:
        logger.warning(f"String truncated from {len(text)} to {max_length} characters")
        text = text[:max_length]

    # Remove null bytes and other control characters (except newlines and tabs)
    # Keep printable characters, newlines (\n), and tabs (\t)
    sanitized = "".join(
        c
        for c in text
        if c.isprintable() or c in ["\n", "\t"]
    )

    # Remove or escape common prompt injection patterns
    # These patterns could be used to inject instructions into the prompt
    injection_patterns = [
        r"(?i)ignore\s+(previous|above|all)\s+instructions?",
        r"(?i)forget\s+(previous|above|all)\s+instructions?",
        r"(?i)system\s*:",
        r"(?i)assistant\s*:",
        r"(?i)user\s*:",
        r"(?i)new\s+instructions?",
        r"(?i)override",
    ]

    for pattern in injection_patterns:
        # Replace with escaped version (keep the text but make it less likely to be interpreted as instruction)
        sanitized = re.sub(pattern, lambda m: f"[sanitized: {m.group(0)}]", sanitized)

    return sanitized


def sanitize_column_name(name: str) -> str:
    """Sanitize a column name for safe inclusion in prompts.

    Column names are typically safe, but we ensure they don't contain
    prompt injection patterns or control characters.

    Args:
        name: Column name to sanitize

    Returns:
        Sanitized column name
    """
    if not isinstance(name, str):
        name = str(name)

    # Remove control characters (keep printable only)
    sanitized = "".join(c for c in name if c.isprintable())

    # Truncate if unreasonably long (column names shouldn't be > 200 chars)
    if len(sanitized) > 200:
        logger.warning(f"Column name truncated from {len(sanitized)} to 200 characters")
        sanitized = sanitized[:200]

    return sanitized


def sanitize_json_for_prompt(data: Any, max_string_length: int = 5000) -> str:
    """Sanitize data and convert to JSON string for prompt inclusion.

    This function:
    - Converts data to JSON
    - Sanitizes string values in the JSON
    - Handles nested structures
    - Ensures valid JSON output

    Args:
        data: Data to convert to JSON (dict, list, etc.)
        max_string_length: Maximum length for string values in JSON

    Returns:
        Sanitized JSON string safe for prompt inclusion

    Raises:
        PromptSanitizationError: If data cannot be serialized or sanitized
    """
    try:
        # Recursively sanitize string values in the data structure
        def sanitize_value(value: Any) -> Any:
            if isinstance(value, str):
                return sanitize_string_for_prompt(value, max_length=max_string_length)
            elif isinstance(value, dict):
                return {k: sanitize_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [sanitize_value(item) for item in value]
            elif isinstance(value, (int, float, bool)) or value is None:
                return value
            else:
                # For other types, convert to string and sanitize
                return sanitize_string_for_prompt(str(value), max_length=max_string_length)

        sanitized_data = sanitize_value(data)

        # Convert to JSON with proper escaping
        json_str = json.dumps(sanitized_data, indent=2, ensure_ascii=False)

        # Additional safety: ensure the JSON string itself doesn't contain injection patterns
        # This is a final check in case JSON encoding introduced something
        json_str = sanitize_string_for_prompt(json_str, max_length=100000)

        return json_str

    except (TypeError, ValueError) as e:
        raise PromptSanitizationError(
            f"Failed to sanitize data for prompt: {e}"
        ) from e


def sanitize_prompt_variable(value: Any, var_type: str = "string") -> str:
    """Sanitize a variable value for inclusion in f-string prompts.

    This is a convenience function that handles different types of values
    commonly used in prompt construction.

    Args:
        value: Value to sanitize (string, number, enum, etc.)
        var_type: Type hint ("string", "column_name", "json", "number")

    Returns:
        Sanitized string representation safe for prompt inclusion
    """
    if var_type == "column_name":
        return sanitize_column_name(str(value))
    elif var_type == "json":
        if isinstance(value, (dict, list)):
            return sanitize_json_for_prompt(value)
        else:
            return sanitize_string_for_prompt(str(value))
    elif var_type == "number":
        # Numbers are generally safe, but ensure they're reasonable
        if isinstance(value, (int, float)):
            # Check for NaN, Inf, etc.
            if isinstance(value, float) and (value != value or abs(value) == float("inf")):
                logger.warning(f"Invalid number value: {value}, using 0")
                return "0"
            return str(value)
        else:
            return sanitize_string_for_prompt(str(value))
    else:  # string or default
        return sanitize_string_for_prompt(str(value))
