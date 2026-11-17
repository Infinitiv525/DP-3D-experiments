"""Analyse persistence of spanwise Fourier modes across saved CFD states."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_spanwise_mode import periodic_resample


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--span", type=float, default=0.04)
    parser.add_argument("--diameter", type=float, default=0.01)
    parser.add_argument("--max-physical-mode", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    time_dirs = sorted(
        (path for path in args.samples_root.iterdir() if path.is_dir()),
        key=lambda path: float(path.name),
    )
    if not time_dirs:
        raise FileNotFoundError(f"No time directories in {args.samples_root}")

    rows: list[dict[str, float | int | str]] = []
    aggregate_energy = np.zeros(args.max_physical_mode + 1)
    for time_dir in time_dirs:
        files = sorted(time_dir.glob("*_U_vorticity.xy"))
        if not files:
            raise FileNotFoundError(f"No profiles in {time_dir}")
        for path in files:
            table = np.loadtxt(path)
            _, values = periodic_resample(table[:, 2], table[:, 6], args.span)
            fluctuation = values - np.mean(values)
            energy = np.abs(np.fft.rfft(fluctuation)) ** 2
            physical = energy[1 : args.max_physical_mode + 1]
            total = float(np.sum(physical))
            dominant = int(np.argmax(physical)) + 1
            aggregate_energy[1:] += physical
            rows.append({
                "time_s": float(time_dir.name),
                "line": path.stem.removesuffix("_U_vorticity"),
                "omega_x_rms": float(np.sqrt(np.mean(fluctuation ** 2))),
                "dominant_mode": dominant,
                "dominant_wavelength_over_d": args.span / args.diameter / dominant,
                "mode1_energy_fraction": float(energy[1] / total),
                "mode2_energy_fraction": float(energy[2] / total),
                "mode3_energy_fraction": float(energy[3] / total),
                "resolved_energy": total,
                "mode1_energy": float(energy[1]),
            })

    lines = sorted({str(row["line"]) for row in rows})
    line_rows: list[dict[str, float | int | str]] = []
    for line in lines:
        selected = [row for row in rows if row["line"] == line]
        fractions = np.array([row["mode1_energy_fraction"] for row in selected])
        resolved = np.array([row["resolved_energy"] for row in selected])
        mode1 = np.array([row["mode1_energy"] for row in selected])
        modes = [int(row["dominant_mode"]) for row in selected]
        line_rows.append({
            "line": line,
            "time_samples": len(selected),
            "mode1_dominant_count": sum(mode == 1 for mode in modes),
            "mode1_dominant_fraction": sum(mode == 1 for mode in modes) / len(modes),
            "mode1_fraction_mean": float(np.mean(fractions)),
            "mode1_fraction_median": float(np.median(fractions)),
            "mode1_fraction_energy_weighted": float(np.sum(mode1) / np.sum(resolved)),
            "omega_x_rms_mean": float(np.mean([row["omega_x_rms"] for row in selected])),
        })

    all_fractions = np.array([row["mode1_energy_fraction"] for row in rows])
    all_modes = [int(row["dominant_mode"]) for row in rows]
    total_resolved = sum(float(row["resolved_energy"]) for row in rows)
    total_mode1 = sum(float(row["mode1_energy"]) for row in rows)
    mode_counts = Counter(all_modes)
    summary = {
        "status": "SPANWISE_MODE_SERIES_ANALYSIS_OK",
        "time_samples": len(time_dirs),
        "first_time_s": float(time_dirs[0].name),
        "last_time_s": float(time_dirs[-1].name),
        "profile_lines": len(lines),
        "total_profiles": len(rows),
        "span_over_diameter": args.span / args.diameter,
        "expected_mode_a_fourier_mode": 1,
        "mode1_dominant_profiles": mode_counts[1],
        "mode1_dominant_fraction": mode_counts[1] / len(rows),
        "mode1_energy_fraction_mean": float(np.mean(all_fractions)),
        "mode1_energy_fraction_median": float(np.median(all_fractions)),
        "mode1_energy_fraction_energy_weighted": float(total_mode1 / total_resolved),
        "dominant_mode_counts": {str(mode): count for mode, count in sorted(mode_counts.items())},
        "interpretation": (
            "persistent_mode1_supports_a_4D_spanwise_structure; classification_as_"
            "Mode_A_should_be_reported_as_consistent_with_not_uniquely_proven"
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "spanwise_series_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for filename, data in (("spanwise_series_metrics.csv", rows),
                           ("spanwise_series_by_line.csv", line_rows)):
        with (args.output_dir / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)

    times = np.array([float(path.name) for path in time_dirs])
    matrix = np.array([
        [[row["mode1_energy_fraction"] for row in rows
          if row["line"] == line and row["time_s"] == time][0]
         for time in times] for line in lines
    ], dtype=float)
    figure, axis = plt.subplots(figsize=(12, 4.8))
    image = axis.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis",
                        extent=[times[0] - 5, times[-1] + 5, len(lines) - 0.5, -0.5])
    axis.set_yticks(range(len(lines)), labels=lines)
    axis.set_xlabel("Physical time [s]")
    axis.set_title("Energy fraction of the 4D spanwise mode k=1")
    figure.colorbar(image, ax=axis, label="Mode-1 energy fraction")
    figure.tight_layout()
    figure.savefig(args.output_dir / "mode1_fraction_heatmap.png", dpi=170)
    plt.close(figure)

    normalized = aggregate_energy / np.sum(aggregate_energy[1:])
    figure, axis = plt.subplots(figsize=(9, 4.8))
    modes = np.arange(1, args.max_physical_mode + 1)
    axis.bar(modes, normalized[1:], color="#6a1b9a")
    axis.set_xticks(modes)
    axis.set_xlabel("Spanwise Fourier mode k")
    axis.set_ylabel("Aggregated energy fraction")
    axis.set_title("Streamwise-vorticity spectrum aggregated over time and wake lines")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(args.output_dir / "aggregate_spanwise_spectrum.png", dpi=170)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 5))
    for line in lines:
        selected = [row for row in rows if row["line"] == line]
        axis.plot([row["time_s"] for row in selected],
                  [row["mode1_energy_fraction"] for row in selected],
                  marker="o", markersize=3, linewidth=1, label=line)
    axis.set_xlabel("Physical time [s]")
    axis.set_ylabel("Mode-1 energy fraction")
    axis.set_ylim(0, 1.03)
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(args.output_dir / "mode1_fraction_time_series.png", dpi=170)
    plt.close(figure)

    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
