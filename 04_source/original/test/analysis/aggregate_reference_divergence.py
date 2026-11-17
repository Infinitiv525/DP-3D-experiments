"""Aggregate per-scenario reference-divergence results without opening field data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    records: list[dict[str, str]] = []
    for path in sorted(args.input_dir.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as stream:
            records.extend(csv.DictReader(stream))
    if len(records) != 8 * 201:
        raise RuntimeError(f"Expected 1608 records, found {len(records)}")
    dimensional = np.asarray([float(row["divergence_rms_s_inv"]) for row in records])
    normalized = np.asarray([float(row["normalized_divergence_rms"]) for row in records])
    summary = {
        "status": "REFERENCE_DIVERGENCE_AGGREGATE_OK",
        "scenarios": sorted({row["scenario_id"] for row in records}),
        "states": len(records),
        "divergence_rms_s_inv": {
            "mean": float(np.mean(dimensional)),
            "p50": float(np.percentile(dimensional, 50)),
            "p95": float(np.percentile(dimensional, 95)),
            "p99": float(np.percentile(dimensional, 99)),
            "max": float(np.max(dimensional)),
        },
        "normalized_divergence_rms": {
            "mean": float(np.mean(normalized)),
            "p50": float(np.percentile(normalized, 50)),
            "p95": float(np.percentile(normalized, 95)),
            "p99": float(np.percentile(normalized, 99)),
            "max": float(np.max(normalized)),
        },
    }
    output = args.input_dir / "aggregate.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
