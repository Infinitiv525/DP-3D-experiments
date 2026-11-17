"""Compare two fine-grid continuations with different Courant limits."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_grid_verification import frequency_metrics, load_table


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis-start", type=float, default=110.0)
    parser.add_argument("--diameter", type=float, default=0.01)
    parser.add_argument("--velocity", type=float, default=0.02)
    return parser.parse_args()


def relative_difference(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-30)


def main() -> None:
    args = arguments()
    with args.runs_csv.open(newline="", encoding="utf-8") as stream:
        runs = list(csv.DictReader(stream))
    if [row["label"] for row in runs] != ["co050", "co025"]:
        raise RuntimeError("Expected co050 and co025 rows")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics, series = [], {}
    for run in runs:
        result_dir = Path(run["result_dir"])
        coefficient = next(result_dir.glob("tables/forceCoefficients/*/coefficient.dat"))
        data = load_table(coefficient)
        time, cd, cl = data[:, 0], data[:, 1], data[:, 4]
        mask = time >= args.analysis_start
        at, acd, acl = time[mask], cd[mask], cl[mask]
        if len(at) < 1000 or at[-1] < 159.999:
            raise RuntimeError(f"Incomplete analysis interval for {run['label']}")
        midpoint = 0.5 * (at[0] + at[-1])
        first, second = at <= midpoint, at >= midpoint
        frequencies = frequency_metrics(at, acl)
        selected_frequency = float(frequencies["frequency_crossing_hz"])
        mean_cd_first, mean_cd_second = float(np.mean(acd[first])), float(np.mean(acd[second]))
        rms_cl_first = float(np.sqrt(np.mean(acl[first] ** 2)))
        rms_cl_second = float(np.sqrt(np.mean(acl[second] ** 2)))
        log_text = (result_dir / "logs" / "log.pimpleFoam").read_text(
            encoding="utf-8", errors="replace")
        courant = [float(x) for x in re.findall(r"Courant Number mean: \S+ max: (\S+)", log_text)]
        continuity = [float(x) for x in re.findall(r"cumulative = (\S+)", log_text)]
        row = {
            "label": run["label"], "job_id": run["job_id"],
            "max_co_setting": float(run["max_co"]),
            "analysis_start_s": args.analysis_start, "analysis_end_s": float(at[-1]),
            "samples": len(at), "mean_cd": float(np.mean(acd)),
            "std_cd": float(np.std(acd)), "rms_cl": float(np.sqrt(np.mean(acl**2))),
            "peak_to_peak_cl": float(np.ptp(acl)), **frequencies,
            "strouhal_number": selected_frequency * args.diameter / args.velocity,
            "mean_cd_first_half": mean_cd_first, "mean_cd_second_half": mean_cd_second,
            "relative_mean_cd_window_difference": relative_difference(mean_cd_first, mean_cd_second),
            "rms_cl_first_half": rms_cl_first, "rms_cl_second_half": rms_cl_second,
            "relative_rms_cl_window_difference": relative_difference(rms_cl_first, rms_cl_second),
            "max_courant_number": max(courant),
            "last_cumulative_continuity_error": continuity[-1],
        }
        metrics.append(row)
        series[run["label"]] = (time, cd, cl)

    coarse_dt, fine_dt = metrics[0], metrics[1]
    st_difference = relative_difference(
        float(coarse_dt["strouhal_number"]), float(fine_dt["strouhal_number"]))
    cd_difference = relative_difference(float(coarse_dt["mean_cd"]), float(fine_dt["mean_cd"]))
    rms_difference = relative_difference(float(coarse_dt["rms_cl"]), float(fine_dt["rms_cl"]))
    time_step_pass = st_difference <= 0.01 and cd_difference <= 0.02
    frequency_stationary = all(
        float(row["relative_frequency_window_difference"]) <= 0.05 for row in metrics)
    amplitude_diagnostics = all(
        float(row["relative_mean_cd_window_difference"]) <= 0.02
        and float(row["relative_rms_cl_window_difference"]) <= 0.05
        for row in metrics)
    summary = {
        "status": "CYLINDER3D_TIME_STEP_ANALYSIS_OK",
        "runs": metrics,
        "relative_strouhal_difference": st_difference,
        "relative_mean_cd_difference": cd_difference,
        "relative_rms_cl_difference": rms_difference,
        "time_step_acceptance_passed": time_step_pass,
        "frequency_stationarity_passed": frequency_stationary,
        "amplitude_stationarity_diagnostic_passed": amplitude_diagnostics,
        "validation_status": (
            "time_and_stationarity_passed" if time_step_pass and frequency_stationary
            and amplitude_diagnostics else "additional_investigation_required"
        ),
    }
    (args.output_dir / "time_step_verification.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (args.output_dir / "time_step_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)

    colours = {"co050": "#1565c0", "co025": "#c62828"}
    labels = {"co050": "maxCo 0.50", "co025": "maxCo 0.25"}
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for label in ("co050", "co025"):
        time, cd, cl = series[label]
        late = time >= args.analysis_start
        axes[0].plot(time[late], cd[late], label=labels[label], color=colours[label], linewidth=0.9)
        axes[1].plot(time[late], cl[late], label=labels[label], color=colours[label], linewidth=0.9)
    axes[0].set_ylabel("Cd")
    axes[1].set_ylabel("Cl")
    axes[1].set_xlabel("Physical time [s]")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Fine-grid time-step comparison, Re=200")
    figure.tight_layout()
    figure.savefig(args.output_dir / "time_step_force_comparison.png", dpi=170)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    names = [("strouhal_number", "St"), ("mean_cd", "mean Cd"), ("rms_cl", "RMS Cl")]
    x = [0.5, 0.25]
    for axis, (key, ylabel) in zip(axes, names):
        axis.plot(x, [float(row[key]) for row in metrics], marker="o")
        axis.set_xlabel("Maximum Courant number")
        axis.set_ylabel(ylabel)
        axis.invert_xaxis()
        axis.grid(alpha=0.25)
    figure.suptitle("Time-step sensitivity quantities")
    figure.tight_layout()
    figure.savefig(args.output_dir / "time_step_sensitivity.png", dpi=170)
    plt.close(figure)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
