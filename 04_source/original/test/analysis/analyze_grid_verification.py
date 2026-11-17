"""Compare three verified 3D cylinder meshes and estimate grid uncertainty."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis-start", type=float, default=30.0)
    parser.add_argument("--diameter", type=float, default=0.01)
    parser.add_argument("--velocity", type=float, default=0.02)
    return parser.parse_args()


def load_table(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            rows.append([float(value) for value in line.split()])
    data = np.asarray(rows)
    if data.ndim != 2 or data.shape[1] < 5:
        raise RuntimeError(f"Unexpected coefficient table: {path}")
    return data


def fft_frequency(time: np.ndarray, signal: np.ndarray) -> float:
    dt = float(np.median(np.diff(time)))
    uniform_time = np.arange(time[0], time[-1] + 0.5 * dt, dt)
    uniform_signal = np.interp(uniform_time, time, signal)
    windowed = (uniform_signal - np.mean(uniform_signal)) * np.hanning(len(uniform_signal))
    amplitude = np.abs(np.fft.rfft(windowed))
    frequencies = np.fft.rfftfreq(len(windowed), dt)
    amplitude[0] = 0.0
    return float(frequencies[int(np.argmax(amplitude))])


def crossing_frequency(time: np.ndarray, signal: np.ndarray) -> tuple[float, int]:
    centred = signal - np.mean(signal)
    indices = np.flatnonzero((centred[:-1] <= 0.0) & (centred[1:] > 0.0))
    crossings = []
    for index in indices:
        y0, y1 = centred[index], centred[index + 1]
        fraction = -y0 / (y1 - y0)
        crossings.append(time[index] + fraction * (time[index + 1] - time[index]))
    if len(crossings) < 4:
        return math.nan, len(crossings)
    periods = np.diff(np.asarray(crossings))
    return float(1.0 / np.mean(periods)), len(periods)


def frequency_metrics(time: np.ndarray, signal: np.ndarray) -> dict[str, float | int]:
    crossing, periods = crossing_frequency(time, signal)
    fft = fft_frequency(time, signal)
    midpoint = 0.5 * (time[0] + time[-1])
    first = time <= midpoint
    second = time >= midpoint
    first_frequency, first_periods = crossing_frequency(time[first], signal[first])
    second_frequency, second_periods = crossing_frequency(time[second], signal[second])
    selected = crossing if math.isfinite(crossing) else fft
    if math.isfinite(first_frequency) and math.isfinite(second_frequency):
        stability = abs(first_frequency - second_frequency) / selected
    else:
        stability = math.nan
    return {
        "frequency_crossing_hz": crossing,
        "frequency_fft_hz": fft,
        "observed_periods": periods,
        "frequency_first_half_hz": first_frequency,
        "frequency_second_half_hz": second_frequency,
        "periods_first_half": first_periods,
        "periods_second_half": second_periods,
        "relative_frequency_window_difference": stability,
    }


def gci(values: dict[str, float], cells: dict[str, int]) -> dict[str, float | str | None]:
    phi3, phi2, phi1 = values["coarse"], values["medium"], values["fine"]
    r32 = (cells["medium"] / cells["coarse"]) ** (1.0 / 3.0)
    r21 = (cells["fine"] / cells["medium"]) ** (1.0 / 3.0)
    e32, e21 = phi3 - phi2, phi2 - phi1
    if e32 == 0.0 or e21 == 0.0 or e32 * e21 <= 0.0:
        return {"convergence": "non_monotonic_or_exact", "r32": r32, "r21": r21,
                "observed_order": None, "extrapolated_fine_limit": None,
                "gci_fine_percent": None}
    ratio = abs(e32 / e21)
    rmean = math.sqrt(r32 * r21)
    order = abs(math.log(ratio) / math.log(rmean))
    if not math.isfinite(order) or order <= 0.0:
        return {"convergence": "order_unavailable", "r32": r32, "r21": r21,
                "observed_order": None, "extrapolated_fine_limit": None,
                "gci_fine_percent": None}
    denominator = r21**order - 1.0
    extrapolated = (r21**order * phi1 - phi2) / denominator
    relative_21 = abs((phi1 - phi2) / phi1)
    return {"convergence": "monotonic", "r32": r32, "r21": r21,
            "observed_order": order, "extrapolated_fine_limit": extrapolated,
            "gci_fine_percent": 1.25 * relative_21 / denominator * 100.0}


def main() -> None:
    args = arguments()
    with args.runs_csv.open(newline="", encoding="utf-8") as stream:
        runs = list(csv.DictReader(stream))
    if [run["grid"] for run in runs] != ["coarse", "medium", "fine"]:
        raise RuntimeError("Runs CSV must be ordered coarse, medium, fine")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics, series = [], {}
    for run in runs:
        result_dir = Path(run["result_dir"])
        coefficient = next(result_dir.glob("tables/forceCoefficients/*/coefficient.dat"))
        data = load_table(coefficient)
        time, cd, cl = data[:, 0], data[:, 1], data[:, 4]
        mask = time >= args.analysis_start
        at, acd, acl = time[mask], cd[mask], cl[mask]
        if len(at) < 1000:
            raise RuntimeError(f"Not enough post-transient samples for {run['grid']}")
        frequencies = frequency_metrics(at, acl)
        log_text = (result_dir / "logs" / "log.pimpleFoam").read_text(
            encoding="utf-8", errors="replace")
        courant = [float(x) for x in re.findall(r"Courant Number mean: \S+ max: (\S+)", log_text)]
        continuity = [float(x) for x in re.findall(r"cumulative = (\S+)", log_text)]
        selected_frequency = float(frequencies["frequency_crossing_hz"])
        row = {
            "grid": run["grid"], "job_id": run["job_id"],
            "cell_count": int(run["cell_count"]),
            "analysis_start_s": args.analysis_start, "analysis_end_s": float(at[-1]),
            "samples": len(at), "mean_cd": float(np.mean(acd)),
            "std_cd": float(np.std(acd)), "mean_cl": float(np.mean(acl)),
            "rms_cl": float(np.sqrt(np.mean(acl**2))),
            "peak_to_peak_cl": float(np.ptp(acl)), **frequencies,
            "strouhal_number": selected_frequency * args.diameter / args.velocity,
            "max_courant_number": max(courant),
            "last_cumulative_continuity_error": continuity[-1],
        }
        metrics.append(row)
        series[run["grid"]] = (time, cd, cl)

    by_grid = {row["grid"]: row for row in metrics}
    cells = {grid: int(by_grid[grid]["cell_count"]) for grid in by_grid}
    quantities = {
        "strouhal_number": {g: float(by_grid[g]["strouhal_number"]) for g in by_grid},
        "mean_cd": {g: float(by_grid[g]["mean_cd"]) for g in by_grid},
        "rms_cl": {g: float(by_grid[g]["rms_cl"]) for g in by_grid},
    }
    uncertainty = {name: gci(values, cells) for name, values in quantities.items()}
    st_change = abs(quantities["strouhal_number"]["fine"] - quantities["strouhal_number"]["medium"]) / abs(quantities["strouhal_number"]["fine"])
    cd_change = abs(quantities["mean_cd"]["fine"] - quantities["mean_cd"]["medium"]) / abs(quantities["mean_cd"]["fine"])
    frequency_stable = all(float(row["relative_frequency_window_difference"]) <= 0.05 for row in metrics)
    grid_pass = frequency_stable and st_change <= 0.02 and cd_change <= 0.03
    summary = {
        "status": "CYLINDER3D_GRID_VERIFICATION_ANALYSIS_OK",
        "analysis_start_s": args.analysis_start, "runs": metrics, "gci": uncertainty,
        "medium_fine_relative_st_difference": st_change,
        "medium_fine_relative_mean_cd_difference": cd_change,
        "all_frequency_windows_within_5_percent": frequency_stable,
        "grid_acceptance_passed": grid_pass,
        "validation_status": "grid_pass_pending_time_step_study" if grid_pass else "grid_acceptance_failed",
    }
    (args.output_dir / "grid_verification.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (args.output_dir / "grid_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)

    colours = {"coarse": "#ef6c00", "medium": "#1565c0", "fine": "#2e7d32"}
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for grid in ("coarse", "medium", "fine"):
        time, cd, cl = series[grid]
        late = time >= args.analysis_start
        axes[0].plot(time[late], cd[late], label=grid, color=colours[grid], linewidth=0.9)
        axes[1].plot(time[late], cl[late], label=grid, color=colours[grid], linewidth=0.9)
    axes[0].set_ylabel("Cd")
    axes[1].set_ylabel("Cl")
    axes[1].set_xlabel("Physical time [s]")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("3D cylinder Re=200: grid comparison, analysis window")
    figure.tight_layout()
    figure.savefig(args.output_dir / "force_history_grid_comparison.png", dpi=170)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    labels = [("strouhal_number", "St"), ("mean_cd", "mean Cd"), ("rms_cl", "RMS Cl")]
    counts = np.asarray([cells[g] for g in ("coarse", "medium", "fine")])
    for axis, (key, label) in zip(axes, labels):
        values = [quantities[key][g] for g in ("coarse", "medium", "fine")]
        axis.plot(counts, values, marker="o")
        axis.set_xscale("log")
        axis.set_xlabel("Cell count")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    figure.suptitle("Grid convergence quantities")
    figure.tight_layout()
    figure.savefig(args.output_dir / "grid_convergence.png", dpi=170)
    plt.close(figure)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
