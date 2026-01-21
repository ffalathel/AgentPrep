"""Metadata writer for Level 3.

This module writes pipeline metadata to disk as JSON files.

All writes are:
- Deterministic (consistent formatting)
- Safe (never overwrites unless explicitly allowed)
- Atomic (creates directories if needed)
"""

import json
from pathlib import Path
from typing import Optional

from utils import PathValidationError, get_logger, validate_output_path

from .metadata_schema import PipelineMetadata

logger = get_logger(__name__)


class MetadataWriter:
    """Writes pipeline metadata to disk.

    This writer:
    - Writes metadata as JSON files
    - Uses deterministic formatting
    - Creates directories if needed
    - Never overwrites existing files unless explicitly allowed
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize the metadata writer.

        Args:
            output_dir: Optional output directory. If None, uses current directory.
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        logger.debug(f"MetadataWriter initialized with output_dir: {self.output_dir}")

    def write(
        self,
        metadata: PipelineMetadata,
        filename: Optional[str] = None,
        overwrite: bool = False,
    ) -> Path:
        """Write metadata to disk as JSON.

        Args:
            metadata: PipelineMetadata object to write
            filename: Optional filename. If None, uses timestamp-based name.
            overwrite: If True, overwrites existing file. Defaults to False.

        Returns:
            Path to written metadata file

        Raises:
            FileExistsError: If file exists and overwrite is False
            OSError: If directory creation or file write fails
        """
        # Generate filename if not provided
        if filename is None:
            timestamp = metadata.timestamp.replace(":", "-").replace(".", "-")
            filename = f"agentprep_metadata_{timestamp}.json"

        # Ensure filename ends with .json
        if not filename.endswith(".json"):
            filename += ".json"

        # Sanitize filename to prevent path injection
        from utils.file_helpers import sanitize_path_component
        filename_base = filename.rsplit('.', 1)[0] if '.' in filename else filename
        filename_ext = filename.rsplit('.', 1)[1] if '.' in filename else ''
        sanitized_base = sanitize_path_component(filename_base)
        sanitized_filename = f"{sanitized_base}.{filename_ext}" if filename_ext else sanitized_base

        # Build full path
        output_path = self.output_dir / sanitized_filename

        # Resolve path to prevent symlink attacks
        try:
            resolved_output_path = output_path.resolve()
            resolved_output_dir = resolved_output_path.parent
        except (OSError, RuntimeError) as e:
            logger.error(f"Failed to resolve output path {output_path}: {e}")
            raise OSError(f"Failed to resolve output path: {e}") from e

        # Check if file exists (using resolved path)
        if resolved_output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Metadata file already exists: {resolved_output_path}. "
                "Set overwrite=True to overwrite."
            )

        # Create directory if needed (using resolved path)
        try:
            resolved_output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create output directory {resolved_output_dir}: {e}")
            raise

        # Write metadata as JSON (using resolved path)
        try:
            json_str = metadata.to_json(indent=2)
            resolved_output_path.write_text(json_str, encoding="utf-8")
            logger.info(f"Metadata written to: {resolved_output_path}")
            return resolved_output_path
        except Exception as e:
            logger.error(f"Failed to write metadata to {output_path}: {e}")
            raise

    def write_to_default_location(
        self, metadata: PipelineMetadata, overwrite: bool = False
    ) -> Path:
        """Write metadata to default location with timestamp-based filename.

        Args:
            metadata: PipelineMetadata object to write
            overwrite: If True, overwrites existing file. Defaults to False.

        Returns:
            Path to written metadata file
        """
        return self.write(metadata, filename=None, overwrite=overwrite)
