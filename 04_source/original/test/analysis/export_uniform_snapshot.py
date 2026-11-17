"""Export one decomposed OpenFOAM state to a compact regular NumPy grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv


def find_internal_mesh(dataset: pv.DataObject) -> pv.DataSet:
    if isinstance(dataset, pv.MultiBlock):
        for index, name in enumerate(dataset.keys()):
            block = dataset[index]
            if name == "internalMesh" and isinstance(block, pv.DataSet):
                return block
            if isinstance(block, pv.MultiBlock):
                try:
                    return find_internal_mesh(block)
                except RuntimeError:
                    pass
    raise RuntimeError("internalMesh was not found in the OpenFOAM reader output")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--time", type=float, required=True)
    parser.add_argument("--diameter", type=float, required=True)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dimensions", nargs=3, type=int, default=[64, 32, 16])
    parser.add_argument("--bounds", nargs=6, type=float,
                        help="Explicit xmin xmax ymin ymax zmin zmax; otherwise use pilot bounds")
    args = parser.parse_args()

    marker = args.case_dir / "case.foam"
    marker.touch(exist_ok=True)
    reader = pv.OpenFOAMReader(str(marker))
    available_times = np.asarray(reader.time_values, dtype=float)
    nearest = int(np.argmin(np.abs(available_times - args.time)))
    selected_time = float(available_times[nearest])
    if abs(selected_time - args.time) > 1e-6:
        raise RuntimeError(f"Requested time {args.time} unavailable: {available_times}")
    reader.set_active_time_value(selected_time)
    internal = find_internal_mesh(reader.read())

    nx, ny, nz = args.dimensions
    bounds = tuple(args.bounds) if args.bounds is not None else (
        -0.05, 0.15, -0.05, 0.05, -2 * args.diameter, 2 * args.diameter)
    if any(bounds[i] >= bounds[i + 1] for i in (0, 2, 4)):
        raise ValueError("Each lower grid bound must be smaller than its upper bound")
    spacing = (
        (bounds[1] - bounds[0]) / (nx - 1),
        (bounds[3] - bounds[2]) / (ny - 1),
        (bounds[5] - bounds[4]) / (nz - 1),
    )
    grid = pv.ImageData(
        dimensions=(nx, ny, nz), spacing=spacing,
        origin=(bounds[0], bounds[2], bounds[4]))
    sampled = grid.sample(internal)
    if "U" not in sampled.point_data or "p" not in sampled.point_data:
        raise RuntimeError(
            f"Expected U and p; point arrays are {list(sampled.point_data.keys())}")
    velocity = np.asarray(sampled.point_data["U"], dtype=np.float32).reshape(
        (nx, ny, nz, 3), order="F")
    pressure = np.asarray(sampled.point_data["p"], dtype=np.float32).reshape(
        (nx, ny, nz), order="F")
    valid = np.asarray(sampled.point_data["vtkValidPointMask"], dtype=np.uint8).reshape(
        (nx, ny, nz), order="F")
    if not np.any(valid):
        raise RuntimeError("Sampling grid contains no valid fluid points")
    if not np.isfinite(velocity).all() or not np.isfinite(pressure).all():
        raise RuntimeError("Export contains non-finite velocity or pressure")
    coordinates = {
        "x": np.linspace(bounds[0], bounds[1], nx, dtype=np.float32),
        "y": np.linspace(bounds[2], bounds[3], ny, dtype=np.float32),
        "z": np.linspace(bounds[4], bounds[5], nz, dtype=np.float32),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.scenario_id}_t{selected_time:g}.npz"
    np.savez_compressed(
        output, U=velocity, p=pressure, valid=valid,
        x=coordinates["x"], y=coordinates["y"], z=coordinates["z"],
        time_s=np.float64(selected_time), diameter_m=np.float64(args.diameter))
    metadata = {
        "status": "UNIFORM_SNAPSHOT_EXPORT_OK",
        "scenario_id": args.scenario_id,
        "source_case": str(args.case_dir),
        "time_s": selected_time,
        "dimensions": [nx, ny, nz],
        "bounds_m": list(bounds),
        "valid_points": int(np.count_nonzero(valid)),
        "total_points": int(valid.size),
        "valid_fraction": float(np.mean(valid)),
        "velocity_min": np.min(velocity, axis=(0, 1, 2)).astype(float).tolist(),
        "velocity_max": np.max(velocity, axis=(0, 1, 2)).astype(float).tolist(),
        "pressure_min": float(np.min(pressure)),
        "pressure_max": float(np.max(pressure)),
        "npz_bytes": output.stat().st_size,
    }
    (args.output_dir / f"{args.scenario_id}_t{selected_time:g}.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
