"""Metadata writer for Level 3.

This module writes pipeline metadata to disk as JSON files.

All writes are:
- Deterministic (consistent formatting)
- Safe (never overwrites unless explicitly allowed)
- Atomic (creates directories if needed)
"""

import json
import logging
from pathlib import Path
from typing import Optional

from .metadata_schema import PipelineMetadata

logger = logging.getLogger(__name__)


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

        # Build full path
        output_path = self.output_dir / filename

        # Check if file exists
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Metadata file already exists: {output_path}. "
                "Set overwrite=True to overwrite."
            )

        # Create directory if needed
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create output directory {self.output_dir}: {e}")
            raise

        # Write metadata as JSON
        try:
            json_str = metadata.to_json(indent=2)
            output_path.write_text(json_str, encoding="utf-8")
            logger.info(f"Metadata written to: {output_path}")
            return output_path
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
