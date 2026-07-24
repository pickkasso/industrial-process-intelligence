"""CLI entry point for generating a deterministic manufacturing demo CSV."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from process_intelligence.demo_data import (
    DemoDatasetConfiguration,
    generate_demo_dataset,
    summarize_demo_dataset,
    write_demo_workflow_csv,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a deterministic synthetic manufacturing-process demo dataset CSV."
        )
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output CSV path (parent directories are created when needed).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=1500,
        help="Number of rows to generate (default: 1500).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation (default: 42).",
    )
    parser.add_argument(
        "--anomaly-fraction",
        type=float,
        default=0.03,
        help="Target fraction of anomalous rows (default: 0.03).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    configuration = DemoDatasetConfiguration(
        row_count=args.rows,
        random_seed=args.seed,
        anomaly_fraction=args.anomaly_fraction,
        start_timestamp=datetime(2024, 1, 1, 8, 0, 0, tzinfo=UTC),
        sampling_interval_minutes=5.0,
    )
    frame = generate_demo_dataset(configuration)

    output_path: Path = args.output
    # Serialize timestamps as UTC unix seconds for TIME-split workflow compatibility.
    write_demo_workflow_csv(frame, output_path)

    summary = summarize_demo_dataset(frame)
    type_counts = summary["anomaly_type_counts"]
    assert isinstance(type_counts, dict)

    print(f"output_path: {output_path.resolve()}")
    print(f"row_count: {summary['row_count']}")
    print(f"column_count: {summary['column_count']}")
    print(f"anomaly_row_count: {summary['anomaly_row_count']}")
    print(f"anomaly_event_count: {summary['anomaly_event_count']}")
    print("anomaly_type_counts:")
    if type_counts:
        for name in sorted(type_counts):
            print(f"  {name}: {type_counts[name]}")
    else:
        print("  (none)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
