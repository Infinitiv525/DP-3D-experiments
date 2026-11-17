"""Re-test Dataset A stationarity using two late 20-second windows."""

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

from analyze_dataset_a_warmup import interval_metrics, relative_difference
from analyze_grid_verification import load_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-csv", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--array-job-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with args.scenario_csv.open(newline="", encoding="utf-8") as stream:
        scenarios = list(csv.DictReader(stream))

    rows = []
    for scenario in scenarios:
        scenario_id = scenario["scenario_id"]
        result_dir = next(args.result_root.glob(
            f"{scenario_id}_warmup160_{args.array_job_id}_*"))
        data = load_table(next(result_dir.glob(
            "tables/forceCoefficients/*/coefficient.dat")))
        first = interval_metrics(data, 120.0, 140.0)
        second = interval_metrics(data, 140.0, 160.0)
        if not all(math.isfinite(float(item["frequency_hz"]))
                   for item in (first, second)):
            raise RuntimeError(f"Insufficient late lift cycles for {scenario_id}")
        freq_diff = relative_difference(float(first["frequency_hz"]),
                                        float(second["frequency_hz"]))
        cd_diff = relative_difference(float(first["mean_cd"]), float(second["mean_cd"]))
        cl_diff = relative_difference(float(first["rms_cl"]), float(second["rms_cl"]))
        passed = freq_diff <= 0.05 and cd_diff <= 0.03 and cl_diff <= 0.10
        rows.append({
            "scenario_id": scenario_id,
            "split": scenario["split"],
            "reynolds": float(scenario["reynolds"]),
            "frequency_120_140_hz": first["frequency_hz"],
            "frequency_140_160_hz": second["frequency_hz"],
            "strouhal_140_160": float(second["frequency_hz"])
                * float(scenario["diameter_m"]) / float(scenario["u_m_s"]),
            "frequency_relative_difference": freq_diff,
            "mean_cd_120_140": first["mean_cd"],
            "mean_cd_140_160": second["mean_cd"],
            "mean_cd_relative_difference": cd_diff,
            "rms_cl_120_140": first["rms_cl"],
            "rms_cl_140_160": second["rms_cl"],
            "rms_cl_relative_difference": cl_diff,
            "periods_120_140": first["observed_periods"],
            "periods_140_160": second["observed_periods"],
            "late_stationarity_pass": passed,
        })

    pass_count = sum(bool(row["late_stationarity_pass"]) for row in rows)
    summary = {
        "status": "DATASET_A_LATE_STATIONARITY_ANALYSIS_OK",
        "scenario_count": len(rows),
        "late_stationarity_pass_count": pass_count,
        "late_stationarity_fail_count": len(rows) - pass_count,
        "comparison_windows_s": [[120, 140], [140, 160]],
        "thresholds": {
            "frequency_relative_difference_max": 0.05,
            "mean_cd_relative_difference_max": 0.03,
            "rms_cl_relative_difference_max": 0.10,
        },
        "recommended_dense_continuation_s": [160, 200]
            if pass_count == len(rows) else None,
        "interpretation": "all_ready" if pass_count == len(rows)
            else "extend_failed_scenarios",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "late_stationarity.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (args.output_dir / "late_stationarity_metrics.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    labels = [row["scenario_id"] for row in rows]
    x = np.arange(len(rows))
    for axis, key, limit, label in (
        (axes[0], "frequency_relative_difference", 0.05, "Frequency"),
        (axes[1], "mean_cd_relative_difference", 0.03, "Mean Cd"),
        (axes[2], "rms_cl_relative_difference", 0.10, "RMS Cl"),
    ):
        axis.bar(x, [100 * float(row[key]) for row in rows])
        axis.axhline(100 * limit, color="red", linestyle="--")
        axis.set_ylabel(label + " diff. [%]")
        axis.grid(axis="y", alpha=0.2)
    axes[-1].set_xticks(x, labels=labels, rotation=45)
    figure.tight_layout()
    figure.savefig(args.output_dir / "late_stationarity.png", dpi=160)
    plt.close(figure)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
