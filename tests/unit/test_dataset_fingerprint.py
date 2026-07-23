"""Unit tests for dataset content fingerprinting (Step 11B.11)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from process_intelligence.workflow import (
    compute_dataset_content_fingerprint,
    compute_dataset_content_fingerprint_from_bytes,
    is_valid_dataset_fingerprint,
    normalize_optional_dataset_fingerprint,
)


def test_same_content_same_fingerprint(tmp_path: Path) -> None:
    content = b"a,b\n1,2\n"
    first = tmp_path / "one.csv"
    second = tmp_path / "two.csv"
    first.write_bytes(content)
    second.write_bytes(content)
    assert compute_dataset_content_fingerprint(first) == (
        compute_dataset_content_fingerprint(second)
    )


def test_different_content_different_fingerprint(tmp_path: Path) -> None:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    left.write_bytes(b"a,b\n1,2\n")
    right.write_bytes(b"a,b\n1,3\n")
    assert compute_dataset_content_fingerprint(left) != (
        compute_dataset_content_fingerprint(right)
    )


def test_fingerprint_is_64_char_lowercase_hex(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    path.write_bytes(b"x,y\n0,1\n")
    fingerprint = compute_dataset_content_fingerprint(path)
    assert len(fingerprint) == 64
    assert fingerprint == fingerprint.lower()
    assert all(character in "0123456789abcdef" for character in fingerprint)
    assert is_valid_dataset_fingerprint(fingerprint)


def test_filename_difference_does_not_change_fingerprint(tmp_path: Path) -> None:
    content = b"col\n1\n"
    alpha = tmp_path / "alpha.csv"
    beta = tmp_path / "beta.csv"
    alpha.write_bytes(content)
    beta.write_bytes(content)
    assert compute_dataset_content_fingerprint(alpha) == (
        compute_dataset_content_fingerprint(beta)
    )


def test_path_difference_does_not_change_fingerprint(tmp_path: Path) -> None:
    content = b"col\n2\n"
    nested = tmp_path / "nested"
    nested.mkdir()
    first = tmp_path / "root.csv"
    second = nested / "root.csv"
    first.write_bytes(content)
    second.write_bytes(content)
    assert compute_dataset_content_fingerprint(first) == (
        compute_dataset_content_fingerprint(second)
    )


def test_fingerprint_does_not_expose_file_path(tmp_path: Path) -> None:
    path = tmp_path / "secret_name.csv"
    path.write_bytes(b"a\n1\n")
    fingerprint = compute_dataset_content_fingerprint(path)
    assert str(path) not in fingerprint
    assert path.name not in fingerprint
    assert "secret_name" not in fingerprint


def test_fingerprint_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "repeat.csv"
    path.write_bytes(b"pressure,temperature\n1,2\n")
    first = compute_dataset_content_fingerprint(path)
    second = compute_dataset_content_fingerprint(path)
    assert first == second


def test_empty_file_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_bytes(b"")
    fingerprint = compute_dataset_content_fingerprint(path)
    assert fingerprint == hashlib.sha256(b"").hexdigest()
    assert is_valid_dataset_fingerprint(fingerprint)


def test_input_file_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "immutable.csv"
    original = b"lot_id,value\nA,1\n"
    path.write_bytes(original)
    compute_dataset_content_fingerprint(path)
    assert path.read_bytes() == original


def test_uses_standard_library_hashing(tmp_path: Path) -> None:
    content = b"standard,library\n1,2\n"
    path = tmp_path / "stdlib.csv"
    path.write_bytes(content)
    assert compute_dataset_content_fingerprint(path) == hashlib.sha256(content).hexdigest()
    assert compute_dataset_content_fingerprint_from_bytes(content) == (
        hashlib.sha256(content).hexdigest()
    )


def test_normalize_optional_fingerprint_rejects_invalid() -> None:
    assert normalize_optional_dataset_fingerprint(None) is None
    valid = hashlib.sha256(b"ok").hexdigest()
    assert normalize_optional_dataset_fingerprint(valid) == valid
    with pytest.raises(ValueError):
        normalize_optional_dataset_fingerprint("not-a-fingerprint")
    with pytest.raises(ValueError):
        normalize_optional_dataset_fingerprint(valid.upper())
    with pytest.raises(ValueError):
        normalize_optional_dataset_fingerprint(123)
