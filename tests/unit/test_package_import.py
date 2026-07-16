"""Verify the process_intelligence package is importable and versioned."""

import process_intelligence


def test_import_process_intelligence() -> None:
    assert process_intelligence is not None


def test_package_version_exists() -> None:
    assert hasattr(process_intelligence, "__version__")
    assert process_intelligence.__version__ is not None


def test_package_version_matches() -> None:
    assert process_intelligence.__version__ == "0.1.0"
