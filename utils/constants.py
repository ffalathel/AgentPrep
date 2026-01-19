"""Constants for AgentPrep.

This module defines shared constants used across the application.
"""

# Exit codes (matching CLI exit codes)
EXIT_SUCCESS = 0
EXIT_INVALID_INTENT = 1
EXIT_POLICY_VIOLATION = 2
EXIT_RUNTIME_ERROR = 3

# Application metadata
APP_NAME = "AgentPrep"
APP_VERSION = "1.0.0"

# Supported file formats
SUPPORTED_DATASET_FORMATS = ["csv", "parquet"]
SUPPORTED_CONFIG_FORMATS = ["yaml", "json"]

# Default values
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_OUTPUT_DIR = "output"
