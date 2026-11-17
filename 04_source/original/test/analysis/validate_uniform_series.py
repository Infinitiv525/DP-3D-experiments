"""Validate one complete compact Dataset A scenario and create its index."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--start", type=float, default=160.0)
    parser.add_argument("--end", type=float, default=200.0)
    parser.add_argument("--step", type=float, default=0.2)
    parser.add_argument("--dimensions", nargs=3, type=int, default=[64, 32, 16])
    args = parser.parse_args()

    expected_times = np.arange(args.start, args.end + args.step / 2, args.step)
    records: list[dict[str, object]] = []
    reference_coordinates: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    for path in args.input_dir.glob(f"{args.scenario_id}_t*.npz"):
        with np.load(path, allow_pickle=False) as data:
            time = float(data["time_s"])
            nx, ny, nz = args.dimensions
            if data["U"].shape != (nx, ny, nz, 3):
                raise RuntimeError(f"Unexpected U shape in {path}: {data['U'].shape}")
            if data["p"].shape != (nx, ny, nz):
                raise RuntimeError(f"Unexpected p shape in {path}: {data['p'].shape}")
            if data["valid"].shape != (nx, ny, nz):
                raise RuntimeError(f"Unexpected valid shape in {path}: {data['valid'].shape}")
            if not np.isfinite(data["U"]).all() or not np.isfinite(data["p"]).all():
                raise RuntimeError(f"Non-finite field value in {path}")
            coordinates = (data["x"], data["y"], data["z"])
            if reference_coordinates is None:
                reference_coordinates = tuple(item.copy() for item in coordinates)
            elif not all(
                np.array_equal(current, reference)
                for current, reference in zip(coordinates, reference_coordinates)
            ):
                raise RuntimeError(f"Coordinate mismatch in {path}")
            valid_fraction = float(np.mean(data["valid"]))
            if not 0.90 < valid_fraction <= 1.0:
                raise RuntimeError(f"Implausible valid fraction in {path}: {valid_fraction}")
            records.append(
                {
                    "time_s": time,
                    "filename": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "valid_fraction": valid_fraction,
                    "u_min": float(np.min(data["U"])),
                    "u_max": float(np.max(data["U"])),
                    "p_min": float(np.min(data["p"])),
                    "p_max": float(np.max(data["p"])),
                }
            )

    records.sort(key=lambda row: float(row["time_s"]))
    observed_times = np.asarray([row["time_s"] for row in records])
    if len(records) != len(expected_times):
        raise RuntimeError(f"Expected {len(expected_times)} states, found {len(records)}")
    if not np.allclose(observed_times, expected_times, atol=1e-6, rtol=0.0):
        raise RuntimeError("Exported time sequence is incomplete or irregular")

    with (args.input_dir / "series_index.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "status": "UNIFORM_SERIES_VALIDATION_OK",
        "scenario_id": args.scenario_id,
        "state_count": len(records),
        "start_time_s": float(observed_times[0]),
        "end_time_s": float(observed_times[-1]),
        "time_step_s": args.step,
        "dimensions": args.dimensions,
        "total_npz_bytes": int(sum(int(row["bytes"]) for row in records)),
        "valid_fraction_min": min(float(row["valid_fraction"]) for row in records),
        "valid_fraction_max": max(float(row["valid_fraction"]) for row in records),
    }
    (args.input_dir / "series_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
