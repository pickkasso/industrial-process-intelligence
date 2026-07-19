"""Dataset loading for CSV, XLSX, and Parquet with original-row tracking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import polars as pl

from process_intelligence.core.exceptions import DataValidationError
from process_intelligence.core.schemas import DatasetMetadata

ORIGINAL_ROW_ID_COLUMN = "_original_row_id"

_SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".csv": "csv",
    ".xlsx": "xlsx",
    ".parquet": "parquet",
}


@dataclass(frozen=True, slots=True)
class LoadedDataset:
    """Immutable container pairing a loaded frame with its dataset metadata."""

    frame: pl.DataFrame
    metadata: DatasetMetadata


class DatasetLoader:
    """Load tabular datasets from CSV, XLSX, or Parquet into a Polars frame."""

    def load(self, file_path: str | Path) -> LoadedDataset:
        """Load a supported tabular file and attach original-row identifiers.

        Args:
            file_path: Path to a CSV, XLSX, or Parquet file.

        Returns:
            A ``LoadedDataset`` containing the Polars frame and metadata.

        Raises:
            FileNotFoundError: If ``file_path`` does not exist.
            DataValidationError: If the path is a directory, the extension is
                unsupported, or the reserved ``_original_row_id`` column is
                already present in the source file.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        if path.is_dir():
            raise DataValidationError(f"Expected a file path, got a directory: {path}")

        file_format = _SUPPORTED_EXTENSIONS.get(path.suffix.lower())
        if file_format is None:
            raise DataValidationError(
                f"Unsupported file format '{path.suffix}'. "
                f"Supported formats: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
            )

        frame = self._read_frame(path, file_format)
        return self._build_loaded_dataset(frame, path, file_format)

    def _read_frame(self, path: Path, file_format: str) -> pl.DataFrame:
        if file_format == "csv":
            return pl.read_csv(path)
        if file_format == "xlsx":
            pandas_frame = pd.read_excel(path, sheet_name=0, engine="openpyxl")
            return pl.from_pandas(pandas_frame)
        return pl.read_parquet(path)

    def _build_loaded_dataset(
        self,
        frame: pl.DataFrame,
        path: Path,
        file_format: str,
    ) -> LoadedDataset:
        original_column_names = list(frame.columns)

        if ORIGINAL_ROW_ID_COLUMN in original_column_names:
            raise DataValidationError(
                f"Reserved column '{ORIGINAL_ROW_ID_COLUMN}' is already present "
                "in the source file and cannot be overwritten."
            )

        original_dtypes = {name: str(dtype) for name, dtype in frame.schema.items()}

        frame_with_ids = frame.with_columns(
            pl.int_range(0, pl.len(), dtype=pl.Int64).alias(ORIGINAL_ROW_ID_COLUMN)
        )

        metadata = DatasetMetadata(
            file_name=path.name,
            file_format=file_format,
            row_count=frame.height,
            column_names=original_column_names,
            dtypes=original_dtypes,
            user_description=None,
        )
        return LoadedDataset(frame=frame_with_ids, metadata=metadata)
