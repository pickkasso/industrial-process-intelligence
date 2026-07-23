"""SHA-256 dataset fingerprint for raw CSV file content (Step 11B.11).

Fingerprints are derived only from file bytes. Absolute paths, filenames,
modification times, and Python ``hash()`` are never used as fingerprint inputs.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

_SHA256_HEX_LENGTH = 64
_HEX_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_CHUNK_SIZE = 1024 * 1024


def compute_dataset_content_fingerprint(path: Path) -> str:
    """Return the SHA-256 lowercase hex digest of a file's raw bytes.

    Args:
        path: Filesystem path to the raw uploaded CSV content.

    Returns:
        Exactly 64 lowercase hexadecimal characters.

    Raises:
        TypeError: If ``path`` is not a ``Path``.
        OSError: If the file cannot be read.
        ValueError: If ``path`` is not a regular file.
    """
    if not isinstance(path, Path):
        raise TypeError(f"path must be Path, got {type(path).__name__}")
    if not path.is_file():
        raise ValueError(f"path must be an existing regular file: {path.name}")

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def compute_dataset_content_fingerprint_from_bytes(content: bytes) -> str:
    """Return the SHA-256 lowercase hex digest of raw CSV bytes.

    Args:
        content: Raw uploaded CSV content bytes.

    Returns:
        Exactly 64 lowercase hexadecimal characters.

    Raises:
        TypeError: If ``content`` is not ``bytes``.
    """
    if not isinstance(content, bytes):
        raise TypeError(f"content must be bytes, got {type(content).__name__}")
    return hashlib.sha256(content).hexdigest()


def is_valid_dataset_fingerprint(value: object) -> bool:
    """Return True when ``value`` is a 64-character lowercase hex SHA-256 digest."""
    return isinstance(value, str) and _HEX_PATTERN.fullmatch(value) is not None


def normalize_optional_dataset_fingerprint(value: object) -> str | None:
    """Validate an optional dataset fingerprint for schema fields.

    Args:
        value: ``None`` or a candidate fingerprint string.

    Returns:
        ``None`` or a validated 64-character lowercase hex digest.

    Raises:
        ValueError: If ``value`` is present but not a valid fingerprint.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            "dataset_fingerprint must be str or None, "
            f"got {type(value).__name__}"
        )
    if not is_valid_dataset_fingerprint(value):
        raise ValueError(
            "dataset_fingerprint must be a 64-character lowercase hex SHA-256 "
            f"digest, got {value!r}"
        )
    return value
