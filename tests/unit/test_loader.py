"""Unit tests for DatasetLoader (Step 2A)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest

from process_intelligence.core.exceptions import DataValidationError
from process_intelligence.core.schemas import DatasetMetadata
from process_intelligence.data import (
    ORIGINAL_ROW_ID_COLUMN,
    DatasetLoader,
    LoadedDataset,
)

Writer = Callable[[Path, pl.DataFrame], Path]


def _sample_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "lot_id": ["A", "B", "C"],
            "temperature": [100.0, 110.0, 105.0],
            "pressure": [1, 2, 3],
        }
    )


def _write_csv(path: Path, frame: pl.DataFrame) -> Path:
    frame.write_csv(path)
    return path


def _write_xlsx(path: Path, frame: pl.DataFrame) -> Path:
    frame.to_pandas().to_excel(path, index=False, engine="openpyxl")
    return path


def _write_parquet(path: Path, frame: pl.DataFrame) -> Path:
    frame.write_parquet(path)
    return path


def test_load_csv_successfully(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "sample.csv", _sample_frame())
    loaded = DatasetLoader().load(path)

    assert isinstance(loaded, LoadedDataset)
    assert loaded.metadata.file_format == "csv"
    assert loaded.frame.height == 3
    assert loaded.frame["lot_id"].to_list() == ["A", "B", "C"]


def test_load_xlsx_successfully(tmp_path: Path) -> None:
    path = _write_xlsx(tmp_path / "sample.xlsx", _sample_frame())
    loaded = DatasetLoader().load(path)

    assert isinstance(loaded, LoadedDataset)
    assert loaded.metadata.file_format == "xlsx"
    assert loaded.frame.height == 3
    assert loaded.frame["lot_id"].to_list() == ["A", "B", "C"]


def test_load_parquet_successfully(tmp_path: Path) -> None:
    path = _write_parquet(tmp_path / "sample.parquet", _sample_frame())
    loaded = DatasetLoader().load(path)

    assert isinstance(loaded, LoadedDataset)
    assert loaded.metadata.file_format == "parquet"
    assert loaded.frame.height == 3
    assert loaded.frame["lot_id"].to_list() == ["A", "B", "C"]


@pytest.mark.parametrize(
    ("filename", "writer"),
    [
        ("sample.csv", _write_csv),
        ("sample.xlsx", _write_xlsx),
        ("sample.parquet", _write_parquet),
    ],
)
def test_loaded_frame_is_polars_dataframe(
    tmp_path: Path,
    filename: str,
    writer: Writer,
) -> None:
    path = writer(tmp_path / filename, _sample_frame())
    loaded = DatasetLoader().load(path)
    assert isinstance(loaded.frame, pl.DataFrame)


@pytest.mark.parametrize(
    ("filename", "writer"),
    [
        ("ordered.csv", _write_csv),
        ("ordered.xlsx", _write_xlsx),
        ("ordered.parquet", _write_parquet),
    ],
)
def test_original_row_order_is_preserved(
    tmp_path: Path,
    filename: str,
    writer: Writer,
) -> None:
    frame = pl.DataFrame(
        {
            "seq": [10, 20, 30, 40],
            "label": ["first", "second", "third", "fourth"],
        }
    )
    path = writer(tmp_path / filename, frame)
    loaded = DatasetLoader().load(path)

    assert loaded.frame["seq"].to_list() == [10, 20, 30, 40]
    assert loaded.frame["label"].to_list() == ["first", "second", "third", "fourth"]
    assert loaded.frame[ORIGINAL_ROW_ID_COLUMN].to_list() == [0, 1, 2, 3]


@pytest.mark.parametrize(
    ("filename", "writer"),
    [
        ("ids.csv", _write_csv),
        ("ids.xlsx", _write_xlsx),
        ("ids.parquet", _write_parquet),
    ],
)
def test_original_row_id_is_contiguous_from_zero(
    tmp_path: Path,
    filename: str,
    writer: Writer,
) -> None:
    path = writer(tmp_path / filename, _sample_frame())
    loaded = DatasetLoader().load(path)

    row_ids = loaded.frame[ORIGINAL_ROW_ID_COLUMN].to_list()
    assert row_ids == list(range(loaded.frame.height))
    assert ORIGINAL_ROW_ID_COLUMN in loaded.frame.columns


@pytest.mark.parametrize(
    ("filename", "writer", "file_format"),
    [
        ("meta.csv", _write_csv, "csv"),
        ("meta.xlsx", _write_xlsx, "xlsx"),
        ("meta.parquet", _write_parquet, "parquet"),
    ],
)
def test_metadata_excludes_original_row_id_and_preserves_columns(
    tmp_path: Path,
    filename: str,
    writer: Writer,
    file_format: str,
) -> None:
    path = writer(tmp_path / filename, _sample_frame())
    loaded = DatasetLoader().load(path)

    assert ORIGINAL_ROW_ID_COLUMN not in loaded.metadata.column_names
    assert ORIGINAL_ROW_ID_COLUMN not in loaded.metadata.dtypes
    assert loaded.metadata.column_names == ["lot_id", "temperature", "pressure"]
    assert list(loaded.metadata.dtypes.keys()) == ["lot_id", "temperature", "pressure"]
    assert loaded.metadata.file_name == filename
    assert loaded.metadata.file_format == file_format
    assert loaded.metadata.row_count == 3
    assert loaded.metadata.user_description is None


def test_reserved_original_row_id_column_raises(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        {
            ORIGINAL_ROW_ID_COLUMN: [9, 8, 7],
            "value": [1.0, 2.0, 3.0],
        }
    )
    path = _write_csv(tmp_path / "reserved.csv", frame)

    with pytest.raises(DataValidationError, match=ORIGINAL_ROW_ID_COLUMN):
        DatasetLoader().load(path)


def test_unsupported_extension_raises(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(DataValidationError, match="Unsupported file format"):
        DatasetLoader().load(path)


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.csv"

    with pytest.raises(FileNotFoundError):
        DatasetLoader().load(missing)


def test_directory_input_raises_data_validation_error(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="directory"):
        DatasetLoader().load(tmp_path)


@pytest.mark.parametrize(
    ("filename", "writer", "file_format"),
    [
        ("UPPER.CSV", _write_csv, "csv"),
        ("UPPER.XLSX", _write_xlsx, "xlsx"),
        ("UPPER.PARQUET", _write_parquet, "parquet"),
    ],
)
def test_uppercase_extensions_are_supported(
    tmp_path: Path,
    filename: str,
    writer: Writer,
    file_format: str,
) -> None:
    path = writer(tmp_path / filename, _sample_frame())
    loaded = DatasetLoader().load(path)
    assert loaded.metadata.file_format == file_format
    assert loaded.metadata.file_name == filename


def test_consecutive_loads_do_not_share_loader_state(tmp_path: Path) -> None:
    first_path = _write_csv(
        tmp_path / "first.csv",
        pl.DataFrame({"x": [1, 2], "y": ["a", "b"]}),
    )
    second_path = _write_csv(
        tmp_path / "second.csv",
        pl.DataFrame({"p": [9.0], "q": [8.0], "r": [7.0]}),
    )

    loader = DatasetLoader()
    first = loader.load(first_path)
    second = loader.load(second_path)

    assert first.metadata.column_names == ["x", "y"]
    assert second.metadata.column_names == ["p", "q", "r"]
    assert first.frame.columns != second.frame.columns
    assert first.frame.height == 2
    assert second.frame.height == 1
    assert first.frame[ORIGINAL_ROW_ID_COLUMN].to_list() == [0, 1]
    assert second.frame[ORIGINAL_ROW_ID_COLUMN].to_list() == [0]


def test_load_does_not_modify_source_file_columns(tmp_path: Path) -> None:
    source = _sample_frame()
    path = _write_csv(tmp_path / "immutable.csv", source)
    before = path.read_bytes()

    loaded = DatasetLoader().load(path)
    after = path.read_bytes()

    assert before == after
    reread = pl.read_csv(path)
    assert reread.columns == source.columns
    assert ORIGINAL_ROW_ID_COLUMN not in reread.columns
    assert loaded.frame.select(source.columns).equals(reread)


def test_original_row_id_constant_value() -> None:
    assert ORIGINAL_ROW_ID_COLUMN == "_original_row_id"


def test_loaded_dataset_is_frozen() -> None:
    loaded = LoadedDataset(
        frame=pl.DataFrame({"a": [1], ORIGINAL_ROW_ID_COLUMN: [0]}),
        metadata=DatasetMetadata(
            file_name="demo.csv",
            file_format="csv",
            row_count=1,
            column_names=["a"],
            dtypes={"a": "Int64"},
            user_description=None,
        ),
    )
    with pytest.raises(AttributeError):
        loaded.frame = pl.DataFrame({"b": [2]})  # type: ignore[misc]
