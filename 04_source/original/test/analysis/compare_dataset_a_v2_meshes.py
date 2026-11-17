"""Compare paired fine/coarse training scenarios on their common 3D raster."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summary(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def compare(root: Path, job: str, scenario: str, index: int, u_inlet: float,
            fine_job: str | None = None) -> dict:
    jobs = {"fine": fine_job or job, "coarse": job}
    runs = {label: root / f"{scenario}_solvgate_{jobs[label]}_{index}_{label}"
            for label in ("fine", "coarse")}
    summaries = {label: summary(path / "solver_summary.txt") for label, path in runs.items()}
    for key in ("target_max_courant", "initial_delta_tau", "tau_end", "scenario_id"):
        if summaries["fine"].get(key) != summaries["coarse"].get(key):
            raise ValueError(f"Reference settings differ: {key}")
    data = {}
    for label, path in runs.items():
        files = list((path / "compact").glob("*.npz"))
        if len(files) != 1:
            raise ValueError(f"Expected exactly one terminal snapshot: {path}")
        with np.load(files[0], allow_pickle=False) as archive:
            data[label] = {key: archive[key] for key in archive.files}
    fine, coarse = data["fine"], data["coarse"]
    if abs(float(fine["time_s"]) - float(coarse["time_s"])) > 1e-6:
        raise ValueError("Terminal physical times differ")
    for axis in ("x", "y", "z"):
        if not np.array_equal(fine[axis], coarse[axis]):
            raise ValueError(f"Grid differs along {axis}")
    fmask, cmask = fine["valid"].astype(bool), coarse["valid"].astype(bool)
    common = fmask & cmask
    if not np.any(common):
        raise ValueError("No common fluid points")
    uf, uc = fine["U"][common].astype(float), coarse["U"][common].astype(float)
    pf, pc = fine["p"][common].astype(float), coarse["p"][common].astype(float)
    for values in (uf, uc, pf, pc):
        if not np.isfinite(values).all():
            raise ValueError("Non-finite field values")
    # Kinematic pressure is defined up to a constant; align means on common fluid.
    pf, pc = pf - pf.mean(), pc - pc.mean()
    velocity_error = float(np.linalg.norm(uc - uf) / max(np.linalg.norm(uf), 1e-30))
    pressure_error = float(np.linalg.norm(pc - pf) / max(np.linalg.norm(pf), 1e-30))
    fine_wall = int(summaries["fine"]["wall_seconds"])
    coarse_wall = int(summaries["coarse"]["wall_seconds"])
    # These are exploratory screening thresholds, not physical certification.
    screen = velocity_error <= 0.05 and pressure_error <= 0.10
    return {
        "scenario_id": scenario,
        "tau_end": 2,
        "velocity_relative_l2_common_fluid": velocity_error,
        "pressure_mean_aligned_relative_l2_common_fluid": pressure_error,
        "velocity_vector_rmse_over_u_inlet": float(np.sqrt(np.mean(np.sum((uc-uf)**2, axis=1))) / u_inlet),
        "pressure_rmse_over_u_inlet_squared": float(np.sqrt(np.mean((pc-pf)**2)) / u_inlet**2),
        "mask_disagreement_fraction": float(np.mean(fmask != cmask)),
        "common_fluid_points": int(common.sum()),
        "fine_solver_wall_s": fine_wall,
        "coarse_solver_wall_s": coarse_wall,
        "solver_wall_speedup": fine_wall / coarse_wall,
        "fine_solver_core_h": fine_wall * 32 / 3600,
        "coarse_solver_core_h": coarse_wall * 32 / 3600,
        "fine_max_courant": float(summaries["fine"]["max_courant"]),
        "coarse_max_courant": float(summaries["coarse"]["max_courant"]),
        "exploratory_field_screen_passed": screen,
        "production_mesh_accepted": False,
        "limitation": "Startup at tau=2 only; no late-time, force or time-step convergence validation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--solver-job", required=True)
    parser.add_argument("--fine-job", help="Reuse a fine reference with identical solver settings")
    parser.add_argument("--scenario", choices=("all", "sphere", "mixed_pair"), default="all")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [("sphere", "a2_sp_001", 1, 0.01875), ("mixed_pair", "a2_mp_001", 5, 0.02375)]
    results = [compare(args.results_root, args.solver_job, s, i, u, args.fine_job)
               for family, s, i, u in cases if args.scenario in ("all", family)]
    report = {
        "status": "PAIRED_STARTUP_MESH_COMPARISON_COMPLETE",
        "solver_job": args.solver_job,
        "fine_reference_job": args.fine_job or args.solver_job,
        "screening_thresholds": {"velocity_relative_l2": 0.05, "pressure_relative_l2": 0.10},
        "interpretation": "Difference from the finer numerical solution, not error against physical truth.",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
