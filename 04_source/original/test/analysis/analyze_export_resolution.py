"""Compare uniform-grid export resolutions against a 256x128x64 reference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional


RESOLUTIONS = {
    "grid64": (64, 32, 16),
    "grid128": (128, 64, 32),
    "grid256": (256, 128, 64),
}


def load_parameters(table: Path) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    with table.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            result[row["scenario_id"]] = {
                "diameter": float(row["diameter_m"]),
                "center_x": float(row["center_x_m"]),
                "center_y": float(row["center_y_m"]),
            }
    return result


def interpolate(values: np.ndarray, size: tuple[int, int, int], mode: str) -> np.ndarray:
    tensor = torch.from_numpy(np.moveaxis(values, -1, 0)[None].copy()).float()
    if mode == "nearest":
        resized = functional.interpolate(tensor, size=size, mode=mode)
    else:
        resized = functional.interpolate(tensor, size=size, mode=mode, align_corners=True)
    return np.moveaxis(resized[0].numpy(), 0, -1)


def divergence_rms(velocity: np.ndarray, valid: np.ndarray, axes: tuple[np.ndarray, ...]) -> float:
    dx, dy, dz = (float(axis[1] - axis[0]) for axis in axes)
    divergence = (
        (velocity[2:, 1:-1, 1:-1, 0] - velocity[:-2, 1:-1, 1:-1, 0]) / (2 * dx)
        + (velocity[1:-1, 2:, 1:-1, 1] - velocity[1:-1, :-2, 1:-1, 1]) / (2 * dy)
        + (velocity[1:-1, 1:-1, 2:, 2] - velocity[1:-1, 1:-1, :-2, 2]) / (2 * dz)
    )
    stencil = (
        valid[1:-1, 1:-1, 1:-1]
        & valid[2:, 1:-1, 1:-1]
        & valid[:-2, 1:-1, 1:-1]
        & valid[1:-1, 2:, 1:-1]
        & valid[1:-1, :-2, 1:-1]
        & valid[1:-1, 1:-1, 2:]
        & valid[1:-1, 1:-1, :-2]
    )
    return float(np.sqrt(np.mean(np.square(divergence[stencil], dtype=np.float64))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--scenario-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parameters = load_parameters(args.scenario_table)
    rows: list[dict[str, float | int | str]] = []

    for scenario_dir in sorted(path for path in args.input_root.iterdir() if path.is_dir()):
        scenario = scenario_dir.name
        if scenario not in parameters:
            continue
        params = parameters[scenario]
        reference_paths = sorted((scenario_dir / "grid256").glob(f"{scenario}_t*.npz"))
        for reference_path in reference_paths:
            with np.load(reference_path, allow_pickle=False) as reference_data:
                reference_u = reference_data["U"].astype(np.float32)
                reference_p = reference_data["p"].astype(np.float32)
                reference_valid = reference_data["valid"].astype(bool)
                reference_time = float(reference_data["time_s"])
                reference_axes = tuple(reference_data[name].copy() for name in ("x", "y", "z"))
            for label in ("grid64", "grid128"):
                path = scenario_dir / label / f"{scenario}_t{reference_time:g}.npz"
                with np.load(path, allow_pickle=False) as data:
                    velocity = data["U"].astype(np.float32)
                    pressure = data["p"].astype(np.float32)
                    valid = data["valid"].astype(bool)
                    axes = tuple(data[name].copy() for name in ("x", "y", "z"))
                size = RESOLUTIONS[label]
                ref_u = interpolate(reference_u, size, "trilinear")
                ref_p = interpolate(reference_p[..., None], size, "trilinear")[..., 0]
                ref_valid = interpolate(reference_valid[..., None].astype(np.float32), size, "nearest")[..., 0] > 0.5
                common = valid & ref_valid
                x_grid, y_grid, _ = np.meshgrid(*axes, indexing="ij")
                wake = (
                    common
                    & (x_grid >= params["center_x"])
                    & (x_grid <= params["center_x"] + 10 * params["diameter"])
                    & (np.abs(y_grid - params["center_y"]) <= 1.5 * params["diameter"])
                )
                velocity_error = velocity - ref_u
                pressure_centered = pressure - float(np.mean(pressure[common]))
                ref_pressure_centered = ref_p - float(np.mean(ref_p[common]))
                rows.append(
                    {
                        "scenario_id": scenario,
                        "time_s": reference_time,
                        "resolution": label,
                        "nx": size[0],
                        "ny": size[1],
                        "nz": size[2],
                        "cells_per_diameter_x": params["diameter"] / float(axes[0][1] - axes[0][0]),
                        "mask_disagreement_fraction": float(np.mean(valid != ref_valid)),
                        "velocity_relative_l2_global": float(
                            np.sqrt(np.sum(velocity_error[common] ** 2) / np.sum(ref_u[common] ** 2))
                        ),
                        "velocity_relative_l2_wake": float(
                            np.sqrt(np.sum(velocity_error[wake] ** 2) / np.sum(ref_u[wake] ** 2))
                        ),
                        "pressure_nrmse_global": float(
                            np.sqrt(np.mean((pressure_centered[common] - ref_pressure_centered[common]) ** 2))
                            / np.std(ref_pressure_centered[common])
                        ),
                        "divergence_rms_s_inv": divergence_rms(velocity, valid, axes),
                        "reference256_divergence_rms_s_inv": divergence_rms(
                            reference_u, reference_valid, reference_axes
                        ),
                    }
                )

    if len(rows) != 12:
        raise RuntimeError(f"Expected 12 comparison rows, found {len(rows)}")
    with (args.output_dir / "resolution_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregates: dict[str, dict[str, float]] = {}
    for label in ("grid64", "grid128"):
        selected = [row for row in rows if row["resolution"] == label]
        aggregates[label] = {
            key: float(np.mean([float(row[key]) for row in selected]))
            for key in (
                "cells_per_diameter_x",
                "mask_disagreement_fraction",
                "velocity_relative_l2_global",
                "velocity_relative_l2_wake",
                "pressure_nrmse_global",
                "divergence_rms_s_inv",
            )
        }
    summary = {
        "status": "EXPORT_RESOLUTION_STUDY_OK",
        "reference_resolution": list(RESOLUTIONS["grid256"]),
        "scenarios": sorted({str(row["scenario_id"]) for row in rows}),
        "times_s": sorted({float(row["time_s"]) for row in rows}),
        "comparison_rows": len(rows),
        "aggregates": aggregates,
        "interpretation_note": "This tests regular-grid sampling convergence, not CFD mesh convergence.",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
