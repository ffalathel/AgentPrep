"""Constants for Level 6 artifacts.

This module defines artifact names, default paths, and filenames.
All constants are deterministic and versionable.
"""

from pathlib import Path

# Artifact names
ARTIFACT_DATASET = "dataset"
ARTIFACT_METADATA = "metadata"
ARTIFACT_GOVERNANCE = "governance"
ARTIFACT_FEATURE_PROVENANCE = "feature_provenance"
ARTIFACT_SCHEMA = "schema"
ARTIFACT_QUALITY_PROFILE = "quality_profile"

# Default paths
DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_ARTIFACTS_DIR = Path("artifacts")
DEFAULT_REPORTS_DIR = Path("reports")
DEFAULT_EXPORTS_DIR = Path("exports")

# Filename patterns
DATASET_FILENAME_PATTERN = "dataset_{run_id}.parquet"
METADATA_FILENAME_PATTERN = "metadata_{run_id}.json"
GOVERNANCE_FILENAME_PATTERN = "governance_{run_id}.json"
PROVENANCE_FILENAME_PATTERN = "provenance_{run_id}.json"
SCHEMA_FILENAME_PATTERN = "schema_{run_id}.json"
QUALITY_FILENAME_PATTERN = "quality_{run_id}.json"
REPORT_FILENAME_PATTERN = "report_{run_id}.md"
MANIFEST_FILENAME_PATTERN = "manifest_{run_id}.json"

# File extensions
EXT_PARQUET = ".parquet"
EXT_CSV = ".csv"
EXT_JSON = ".json"
EXT_MARKDOWN = ".md"
