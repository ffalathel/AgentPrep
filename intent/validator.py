"""Intent validator for loading and validating user configuration.

This module handles loading YAML/JSON configuration files and validating
them against the IntentSchema. It provides clear, user-friendly error messages.
"""

import json
import pathlib
from typing import Union

import yaml

from intent.schema import IntentSchema
from utils import PathValidationError, validate_path_safe


class IntentValidationError(Exception):
    """Raised when intent validation fails."""

    pass


def load_config_file(config_path: Union[str, pathlib.Path]) -> dict:
    """Load configuration from YAML or JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        Dictionary containing configuration

    Raises:
        IntentValidationError: If file cannot be loaded or parsed
        PathValidationError: If path contains directory traversal or security issues
    """
    # Validate path for security (prevents directory traversal)
    try:
        config_path = validate_path_safe(
            config_path, must_exist=True, must_be_file=True
        )
    except PathValidationError as e:
        raise IntentValidationError(f"Invalid configuration path: {e}") from e
    except FileNotFoundError as e:
        raise IntentValidationError(f"Configuration file not found: {config_path}") from e

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            if config_path.suffix.lower() in (".yaml", ".yml"):
                config = yaml.safe_load(f)
            elif config_path.suffix.lower() == ".json":
                config = json.load(f)
            else:
                raise IntentValidationError(
                    f"Unsupported file format: {config_path.suffix}. "
                    "Supported formats: .yaml, .yml, .json"
                )
    except (OSError, IOError) as e:
        raise IntentValidationError(f"Failed to read configuration file {config_path}: I/O error: {e}") from e
    except yaml.YAMLError as e:
        raise IntentValidationError(f"Invalid YAML syntax in {config_path}: {e}") from e
    except json.JSONDecodeError as e:
        raise IntentValidationError(f"Invalid JSON syntax in {config_path}: {e}") from e
    except UnicodeDecodeError as e:
        raise IntentValidationError(f"Failed to decode configuration file {config_path}: Encoding error: {e}") from e
    except Exception as e:
        raise IntentValidationError(
            f"Unexpected error reading configuration file {config_path}: {e}"
        ) from e

    if config is None:
        raise IntentValidationError("Configuration file is empty")

    if not isinstance(config, dict):
        raise IntentValidationError(
            f"Configuration must be a dictionary, got {type(config).__name__}"
        )

    return config


def validate_intent(config: dict) -> IntentSchema:
    """Validate configuration against IntentSchema.

    Args:
        config: Configuration dictionary

    Returns:
        Validated IntentSchema instance

    Raises:
        IntentValidationError: If validation fails with user-friendly error message
    """
    try:
        intent = IntentSchema(**config)
        return intent
    except Exception as e:
        # Format Pydantic validation errors for better readability
        error_msg = _format_validation_error(e)
        raise IntentValidationError(f"Intent validation failed:\n{error_msg}") from e


def _format_validation_error(error: Exception) -> str:
    """Format validation error for user-friendly display.

    Args:
        error: Exception from Pydantic validation

    Returns:
        Formatted error message
    """
    error_str = str(error)

    # If it's a Pydantic ValidationError, try to extract field-specific errors
    if hasattr(error, "errors"):
        errors = []
        for err in error.errors():
            field_path = " -> ".join(str(loc) for loc in err.get("loc", []))
            error_msg = err.get("msg", "Validation error")
            error_type = err.get("type", "unknown")
            errors.append(f"  {field_path}: {error_msg} ({error_type})")
        return "\n".join(errors)

    return error_str


def load_and_validate_intent(config_path: Union[str, pathlib.Path]) -> IntentSchema:
    """Load and validate intent from configuration file.

    This is the main entry point for intent validation.

    Args:
        config_path: Path to YAML or JSON configuration file

    Returns:
        Validated IntentSchema instance

    Raises:
        IntentValidationError: If loading or validation fails
    """
    config = load_config_file(config_path)
    intent = validate_intent(config)
    return intent
