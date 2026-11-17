"""Measure the discrete divergence floor of reference train/validation fields."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def scenario_files(dataset_root: Path, scenario: str) -> list[tuple[float, Path]]:
    with (dataset_root / scenario / "series_index.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: float(row["time_s"]))
    result = [(float(row["time_s"]), dataset_root / scenario / row["filename"]) for row in rows]
    if len(result) != 201 or any(not path.is_file() for _, path in result):
        raise RuntimeError(f"Incomplete scenario: {scenario}")
    return result


def divergence_rms(path: Path) -> tuple[float, int]:
    with np.load(path, allow_pickle=False) as data:
        velocity = data["U"].astype(np.float64)
        valid = data["valid"].astype(bool)
        x, y, z = data["x"], data["y"], data["z"]
    dx, dy, dz = float(np.mean(np.diff(x))), float(np.mean(np.diff(y))), float(np.mean(np.diff(z)))
    divergence = (
        (velocity[2:, 1:-1, 1:-1, 0] - velocity[:-2, 1:-1, 1:-1, 0]) / (2.0 * dx)
        + (velocity[1:-1, 2:, 1:-1, 1] - velocity[1:-1, :-2, 1:-1, 1]) / (2.0 * dy)
        + (velocity[1:-1, 1:-1, 2:, 2] - velocity[1:-1, 1:-1, :-2, 2]) / (2.0 * dz)
    )
    stencil = valid[1:-1, 1:-1, 1:-1].copy()
    stencil &= valid[2:, 1:-1, 1:-1]
    stencil &= valid[:-2, 1:-1, 1:-1]
    stencil &= valid[1:-1, 2:, 1:-1]
    stencil &= valid[1:-1, :-2, 1:-1]
    stencil &= valid[1:-1, 1:-1, 2:]
    stencil &= valid[1:-1, 1:-1, :-2]
    values = divergence[stencil]
    return float(np.sqrt(np.mean(values**2))), int(values.size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--scenario-table", type=Path, required=True)
    parser.add_argument("--scenario-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    allowed: list[dict[str, str]] = []
    protected: list[str] = []
    with args.scenario_table.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["split"] in {"train", "validation"}:
                allowed.append(row)
            else:
                protected.append(row["scenario_id"])
    if not 0 <= args.scenario_index < len(allowed):
        raise RuntimeError("Scenario index is outside train/validation range")
    row = allowed[args.scenario_index]
    scenario = row["scenario_id"]
    velocity = float(row["u_m_s"])
    diameter = float(row["diameter_m"])
    scale = velocity / diameter
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, float | int | str]] = []
    for time_s, path in scenario_files(args.dataset_root, scenario):
        rms, stencil_cells = divergence_rms(path)
        records.append(
            {
                "scenario_id": scenario,
                "split": row["split"],
                "time_s": time_s,
                "divergence_rms_s_inv": rms,
                "normalized_divergence_rms": rms / scale,
                "stencil_cells": stencil_cells,
            }
        )
    output_csv = args.output_dir / f"{scenario}.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    dimensional = np.asarray([record["divergence_rms_s_inv"] for record in records])
    normalized = np.asarray([record["normalized_divergence_rms"] for record in records])
    summary = {
        "status": "REFERENCE_DIVERGENCE_OK",
        "scenario_id": scenario,
        "split": row["split"],
        "states": len(records),
        "divergence_rms_s_inv": {
            "mean": float(np.mean(dimensional)),
            "p50": float(np.percentile(dimensional, 50)),
            "p95": float(np.percentile(dimensional, 95)),
            "max": float(np.max(dimensional)),
        },
        "normalized_divergence_rms": {
            "mean": float(np.mean(normalized)),
            "p50": float(np.percentile(normalized, 50)),
            "p95": float(np.percentile(normalized, 95)),
            "max": float(np.max(normalized)),
        },
        "protected_scenarios_not_read": protected,
    }
    (args.output_dir / f"{scenario}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
