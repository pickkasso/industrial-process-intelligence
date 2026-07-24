"""CLI entry point for synthetic demo workflow acceptance validation."""

from __future__ import annotations

import argparse
import sys

from process_intelligence.demo_data import DemoDatasetConfiguration
from process_intelligence.demo_validation import (
    MIN_ANOMALY_ENRICHMENT_FACTOR,
    format_demo_validation_report,
    validate_demo_workflows,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate that the deterministic manufacturing demo dataset produces "
            "useful supervised and anomaly-only results through the public workflow."
        )
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=1500,
        help="Demo dataset row count (default: 1500).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Demo dataset random seed (default: 42).",
    )
    parser.add_argument(
        "--anomaly-fraction",
        type=float,
        default=0.03,
        help="Target injected anomaly fraction (default: 0.03).",
    )
    parser.add_argument(
        "--enrichment-threshold",
        type=float,
        default=MIN_ANOMALY_ENRICHMENT_FACTOR,
        help=(
            "Minimum anomaly enrichment factor required for acceptance "
            f"(default: {MIN_ANOMALY_ENRICHMENT_FACTOR})."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    configuration = DemoDatasetConfiguration(
        row_count=args.rows,
        random_seed=args.seed,
        anomaly_fraction=args.anomaly_fraction,
    )
    report = validate_demo_workflows(
        configuration,
        enrichment_threshold=args.enrichment_threshold,
    )
    print(format_demo_validation_report(report))
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
