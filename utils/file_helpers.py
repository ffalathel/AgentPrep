"""File helper utilities for AgentPrep.

This module provides common file operations used across the application.
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class PathValidationError(Exception):
    """Raised when path validation fails due to security concerns."""

    pass


def ensure_directory(path: Path) -> Path:
    """Ensure directory exists, creating it if necessary.

    Resolves path before operations to prevent symlink attacks.

    Args:
        path: Directory path to ensure

    Returns:
        Resolved Path object (for chaining)

    Raises:
        OSError: If directory creation fails
    """
    try:
        # Resolve path to prevent symlink attacks
        resolved_path = path.resolve()
        resolved_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory ensured: {resolved_path}")
        return resolved_path
    except (OSError, RuntimeError) as e:
        logger.error(f"Failed to resolve or create directory {path}: {e}")
        raise


def get_file_extension(file_path: str | Path) -> str:
    """Get file extension from path.

    Args:
        file_path: File path

    Returns:
        File extension (without dot), empty string if no extension
    """
    path = Path(file_path)
    return path.suffix.lstrip(".").lower()


def is_supported_dataset_format(file_path: str | Path) -> bool:
    """Check if file is a supported dataset format.

    Args:
        file_path: File path to check

    Returns:
        True if format is supported
    """
    extension = get_file_extension(file_path)
    return extension in ["csv", "parquet"]


def is_supported_config_format(file_path: str | Path) -> bool:
    """Check if file is a supported config format.

    Args:
        file_path: File path to check

    Returns:
        True if format is supported
    """
    extension = get_file_extension(file_path)
    return extension in ["yaml", "yml", "json"]


def validate_path_safe(
    file_path: str | Path,
    base_dir: Optional[Path] = None,
    must_exist: bool = False,
    must_be_file: bool = False,
    must_be_dir: bool = False,
) -> Path:
    """Validate path to prevent directory traversal attacks.

    This function:
    - Checks for directory traversal sequences (..)
    - Resolves paths to prevent symlink attacks
    - Optionally validates paths are within a base directory
    - Validates file/directory existence and type

    Args:
        file_path: Path to validate
        base_dir: Optional base directory to restrict paths within
        must_exist: If True, path must exist
        must_be_file: If True, path must be a file
        must_be_dir: If True, path must be a directory

    Returns:
        Resolved Path object

    Raises:
        PathValidationError: If path contains traversal or violates constraints
        FileNotFoundError: If must_exist=True and path doesn't exist
    """
    # Convert to Path and expand user home directory
    path = Path(file_path).expanduser()

    # Check for directory traversal in path parts (more reliable than string check)
    path_parts = path.parts
    if ".." in path_parts:
        raise PathValidationError(
            f"Path contains directory traversal sequence: {file_path}"
        )

    # Also check string representation for encoded traversal attempts
    path_str = str(path)
    # Check for various traversal patterns
    if ".." in path_str or "../" in path_str or "..\\" in path_str:
        raise PathValidationError(
            f"Path contains directory traversal sequence: {file_path}"
        )

    # Resolve path to prevent symlink attacks
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError) as e:
        raise PathValidationError(f"Failed to resolve path {file_path}: {e}") from e

    # Validate against base directory if provided
    if base_dir is not None:
        base_resolved = Path(base_dir).expanduser().resolve()
        try:
            # Check if resolved path is within base directory
            resolved_str = str(resolved)
            base_str = str(base_resolved)
            # Use os.path.commonpath for cross-platform compatibility
            common = os.path.commonpath([resolved_str, base_str])
            if common != base_str and not resolved_str.startswith(base_str + os.sep):
                raise PathValidationError(
                    f"Path {file_path} is outside allowed base directory {base_dir}"
                )
        except ValueError:
            # Paths on different drives (Windows) or invalid
            raise PathValidationError(
                f"Path {file_path} cannot be validated against base directory {base_dir}"
            )

    # Validate existence and type
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {file_path}")

    if must_be_file and not resolved.is_file():
        if resolved.exists():
            raise PathValidationError(f"Path is not a file: {file_path}")
        else:
            raise FileNotFoundError(f"File does not exist: {file_path}")

    if must_be_dir and not resolved.is_dir():
        if resolved.exists():
            raise PathValidationError(f"Path is not a directory: {file_path}")
        else:
            raise FileNotFoundError(f"Directory does not exist: {file_path}")

    return resolved


def sanitize_path_component(component: str) -> str:
    """Sanitize a single path component.

    Removes or replaces dangerous characters that could be used for path injection.

    Args:
        component: Path component to sanitize

    Returns:
        Sanitized path component
    """
    # Remove null bytes, control characters, and dangerous characters
    sanitized = "".join(
        c
        for c in component
        if c.isprintable()
        and c not in ['\x00', '<', '>', ':', '"', '|', '?', '*']
        and not (os.name == "nt" and c in ['/', '\\'])  # On Windows, these are path separators
    )
    # Remove leading/trailing dots and spaces (Windows issue)
    sanitized = sanitized.strip('. ')
    # Replace multiple spaces with single space
    sanitized = ' '.join(sanitized.split())
    return sanitized


def is_system_directory(path: Path) -> bool:
    """Check if path is a system directory that should not be written to.

    Args:
        path: Path to check (should be resolved)

    Returns:
        True if path is a system directory
    """
    # Check both original and resolved paths
    original_str = str(path).lower()
    try:
        resolved = path.resolve()
        resolved_str = str(resolved).lower()
    except (OSError, RuntimeError):
        resolved_str = original_str
    
    # Normalize path separators for comparison
    original_normalized = original_str.replace('\\', '/')
    resolved_normalized = resolved_str.replace('\\', '/')
    
    # Critical system directories that should never be written to (including subdirectories)
    # These are checked against both original and resolved paths
    critical_system_dirs = [
        '/bin', '/boot', '/dev', '/etc', '/lib', '/lib64', '/proc', '/root',
        '/sbin', '/sys', '/usr', '/opt',
        # Also check /private/etc, /private/var (macOS symlinks)
        '/private/etc', '/private/var/lib', '/private/var/log', '/private/var/run',
        # Windows
        'c:/windows', 'c:/system32', 'c:/syswow64',
    ]
    
    # Check both original and resolved paths
    for path_to_check in [original_normalized, resolved_normalized]:
        # Check critical system directories (including subdirectories)
        for sys_dir in critical_system_dirs:
            # Exact match
            if path_to_check == sys_dir:
                return True
            # Subdirectory check
            if path_to_check.startswith(sys_dir + '/'):
                return True
        
        # Special handling for /etc (which resolves to /private/etc on macOS)
        if path_to_check == '/etc' or path_to_check.startswith('/etc/'):
            return True
        if path_to_check == '/private/etc' or path_to_check.startswith('/private/etc/'):
            return True
        
        # For Windows, also check drive-relative paths (any drive letter)
        if os.name == "nt" and len(path_to_check) > 2 and path_to_check[1] == ':':
            # Extract system dir without drive (e.g., '/windows' from 'c:/windows')
            for sys_dir in critical_system_dirs:
                if sys_dir.startswith('c:/'):
                    sys_dir_no_drive = sys_dir[3:]  # Remove 'c:/'
                    path_no_drive = path_to_check[2:]  # Remove drive letter
                    if path_no_drive == sys_dir_no_drive or path_no_drive.startswith(sys_dir_no_drive + '/'):
                        return True
    
    return False


def validate_output_path(
    output_path: str | Path,
    base_dir: Optional[Path] = None,
    allow_existing: bool = True,
) -> Path:
    """Validate and sanitize output directory path.

    This function:
    - Validates path doesn't contain directory traversal
    - Sanitizes path components
    - Prevents writing to system directories
    - Optionally restricts to a base directory
    - Normalizes the path

    Args:
        output_path: Output path to validate and sanitize
        base_dir: Optional base directory to restrict output within
        allow_existing: If True, allow existing directories (default: True)

    Returns:
        Resolved, sanitized Path object

    Raises:
        PathValidationError: If path contains traversal, is a system directory, or violates constraints
    """
    # FIRST: Check for directory traversal in the original input string BEFORE any processing
    original_str = str(output_path)
    if ".." in original_str or "../" in original_str or "..\\" in original_str:
        raise PathValidationError(
            f"Output path contains directory traversal sequence: {output_path}"
        )

    # Convert to Path and expand user home directory
    path = Path(output_path).expanduser()

    # Check path parts for traversal (after expansion but before sanitization)
    if ".." in path.parts:
        raise PathValidationError(
            f"Output path contains directory traversal sequence: {output_path}"
        )

    # Sanitize each path component
    try:
        # Split path into components and sanitize each
        parts = []
        for part in path.parts:
            if part in ('', '.', '..'):
                # Skip empty, current, and parent directory markers
                continue
            sanitized_part = sanitize_path_component(part)
            if sanitized_part:
                parts.append(sanitized_part)
        
        # Reconstruct path from sanitized parts
        if parts:
            # Handle absolute vs relative paths
            if path.is_absolute():
                sanitized_path = Path(parts[0])
                for part in parts[1:]:
                    sanitized_path = sanitized_path / part
            else:
                sanitized_path = Path(*parts)
        else:
            sanitized_path = path  # Fallback to original if sanitization removes everything
    except Exception as e:
        # If sanitization fails, fall back to original path but still validate
        sanitized_path = path
        logger.warning(f"Path sanitization had issues, using original: {e}")

    # Validate path for security (prevents directory traversal and symlink attacks)
    try:
        validated_path = validate_path_safe(
            sanitized_path, base_dir=base_dir, must_be_dir=False
        )
    except PathValidationError as e:
        raise PathValidationError(f"Output path validation failed: {e}") from e

    # Check if path is a system directory
    if is_system_directory(validated_path):
        raise PathValidationError(
            f"Output path cannot be a system directory: {validated_path}"
        )

    # If path exists, validate it's actually a directory (not a file)
    if validated_path.exists():
        if not allow_existing:
            raise PathValidationError(
                f"Output path already exists: {validated_path}"
            )
        if not validated_path.is_dir():
            raise PathValidationError(
                f"Output path exists but is not a directory: {validated_path}"
            )

    return validated_path


def generate_run_id() -> str:
    """Generate a unique run ID.

    Returns:
        Run ID string (timestamp-based)
    """
    from datetime import datetime

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    return f"run_{timestamp}"
