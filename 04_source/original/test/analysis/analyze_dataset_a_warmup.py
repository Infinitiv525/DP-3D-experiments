"""Assess stationarity of all Dataset A pilot warm-up simulations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_grid_verification import crossing_frequency, load_table


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-csv", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--array-job-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def relative_difference(first: float, second: float) -> float:
    scale = 0.5 * (abs(first) + abs(second))
    return abs(first - second) / scale if scale > 0 else 0.0


def interval_metrics(data: np.ndarray, start: float, end: float) -> dict[str, float | int]:
    selected = (data[:, 0] >= start) & (data[:, 0] <= end)
    time, cd, cl = data[selected, 0], data[selected, 1], data[selected, 4]
    frequency, periods = crossing_frequency(time, cl)
    return {
        "frequency_hz": frequency,
        "observed_periods": periods,
        "mean_cd": float(np.mean(cd)),
        "rms_cl": float(np.sqrt(np.mean(cl ** 2))),
    }


def main() -> None:
    args = arguments()
    with args.scenario_csv.open(newline="", encoding="utf-8") as stream:
        scenarios = list(csv.DictReader(stream))
    rows: list[dict[str, float | int | str | bool]] = []
    histories: list[tuple[str, np.ndarray]] = []
    for scenario in scenarios:
        scenario_id = scenario["scenario_id"]
        matches = list(args.result_root.glob(
            f"{scenario_id}_warmup160_{args.array_job_id}_*"))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one result for {scenario_id}, found {len(matches)}")
        result_dir = matches[0]
        summary = (result_dir / "solve_summary.txt").read_text(encoding="utf-8")
        if "status=DATASET_A_WARMUP_OK" not in summary:
            raise RuntimeError(f"Warm-up did not validate: {result_dir}")
        coefficient = next(result_dir.glob(
            "tables/forceCoefficients/*/coefficient.dat"))
        data = load_table(coefficient)
        first = interval_metrics(data, 80.0, 120.0)
        second = interval_metrics(data, 120.0, 160.0)
        if not math.isfinite(float(first["frequency_hz"])) or not math.isfinite(
                float(second["frequency_hz"])):
            raise RuntimeError(f"Insufficient lift cycles for {scenario_id}")
        freq_diff = relative_difference(float(first["frequency_hz"]),
                                        float(second["frequency_hz"]))
        cd_diff = relative_difference(float(first["mean_cd"]), float(second["mean_cd"]))
        cl_diff = relative_difference(float(first["rms_cl"]), float(second["rms_cl"]))
        stationarity_pass = freq_diff <= 0.05 and cd_diff <= 0.03 and cl_diff <= 0.10
        rows.append({
            "scenario_id": scenario_id,
            "split": scenario["split"],
            "reynolds": float(scenario["reynolds"]),
            "frequency_80_120_hz": first["frequency_hz"],
            "frequency_120_160_hz": second["frequency_hz"],
            "strouhal_120_160": float(second["frequency_hz"])
                * float(scenario["diameter_m"]) / float(scenario["u_m_s"]),
            "frequency_relative_difference": freq_diff,
            "mean_cd_80_120": first["mean_cd"],
            "mean_cd_120_160": second["mean_cd"],
            "mean_cd_relative_difference": cd_diff,
            "rms_cl_80_120": first["rms_cl"],
            "rms_cl_120_160": second["rms_cl"],
            "rms_cl_relative_difference": cl_diff,
            "periods_120_160": second["observed_periods"],
            "stationarity_pass": stationarity_pass,
        })
        histories.append((scenario_id, data))

    passed = sum(bool(row["stationarity_pass"]) for row in rows)
    summary = {
        "status": "DATASET_A_WARMUP_ANALYSIS_OK",
        "scenario_count": len(rows),
        "stationarity_pass_count": passed,
        "stationarity_fail_count": len(rows) - passed,
        "comparison_windows_s": [[80, 120], [120, 160]],
        "thresholds": {
            "frequency_relative_difference_max": 0.05,
            "mean_cd_relative_difference_max": 0.03,
            "rms_cl_relative_difference_max": 0.10,
        },
        "recommended_dense_window_start_s": 120 if passed == len(rows) else None,
        "interpretation": (
            "all_scenarios_ready_for_dense_sampling" if passed == len(rows)
            else "extend_only_failed_scenarios_before_dense_sampling"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "warmup_analysis.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (args.output_dir / "warmup_metrics.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(4, 3, figsize=(15, 12), sharex=True)
    for axis, (scenario_id, data) in zip(axes.flat, histories):
        stride = max(1, len(data) // 5000)
        axis.plot(data[::stride, 0], data[::stride, 4], linewidth=0.65)
        axis.axvline(80, color="gray", linestyle="--", linewidth=0.8)
        axis.axvline(120, color="gray", linestyle="--", linewidth=0.8)
        axis.set_title(scenario_id)
        axis.grid(alpha=0.2)
    for axis in axes[:, 0]:
        axis.set_ylabel("Cl")
    for axis in axes[-1]:
        axis.set_xlabel("Physical time [s]")
    figure.suptitle("Dataset A warm-up lift histories")
    figure.tight_layout()
    figure.savefig(args.output_dir / "warmup_lift_histories.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    labels = [str(row["scenario_id"]) for row in rows]
    x = np.arange(len(rows))
    for axis, key, threshold, label in (
        (axes[0], "frequency_relative_difference", 0.05, "Frequency difference"),
        (axes[1], "mean_cd_relative_difference", 0.03, "Mean Cd difference"),
        (axes[2], "rms_cl_relative_difference", 0.10, "RMS Cl difference"),
    ):
        axis.bar(x, [float(row[key]) * 100 for row in rows])
        axis.axhline(threshold * 100, color="red", linestyle="--")
        axis.set_ylabel(label + " [%]")
        axis.grid(axis="y", alpha=0.2)
    axes[-1].set_xticks(x, labels=labels, rotation=45)
    figure.tight_layout()
    figure.savefig(args.output_dir / "warmup_stationarity.png", dpi=160)
    plt.close(figure)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
