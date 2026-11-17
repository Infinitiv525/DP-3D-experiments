"""Analyse force coefficients from the extended 3D cylinder pilot."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--diameter", type=float, default=0.01)
    parser.add_argument("--velocity", type=float, default=0.02)
    parser.add_argument("--analysis-start", type=float, default=10.0)
    return parser.parse_args()


def load_coefficients(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        rows.append([float(value) for value in line.split()])
    data = np.asarray(rows)
    if data.shape[1] < 5:
        raise RuntimeError("Unexpected force coefficient table")
    return data


def dominant_frequency(time: np.ndarray, signal: np.ndarray) -> tuple[float, float]:
    dt = float(np.median(np.diff(time)))
    uniform_time = np.arange(time[0], time[-1] + 0.5 * dt, dt)
    uniform_signal = np.interp(uniform_time, time, signal)
    centred = uniform_signal - np.mean(uniform_signal)
    windowed = centred * np.hanning(len(centred))
    amplitude = np.abs(np.fft.rfft(windowed))
    frequencies = np.fft.rfftfreq(len(windowed), dt)
    amplitude[0] = 0
    peak = int(np.argmax(amplitude))
    return float(frequencies[peak]), float(amplitude[peak])


def main() -> None:
    args = parse_args()
    coefficient_file = next(args.result_dir.glob("tables/forceCoefficients/*/coefficient.dat"))
    data = load_coefficients(coefficient_file)
    time, cd, cl = data[:, 0], data[:, 1], data[:, 4]
    mask = time >= args.analysis_start
    analysis_time, analysis_cd, analysis_cl = time[mask], cd[mask], cl[mask]
    if len(analysis_time) < 100:
        raise RuntimeError("Not enough samples after transient interval")

    frequency, _ = dominant_frequency(analysis_time, analysis_cl)
    midpoint = 0.5 * (analysis_time[0] + analysis_time[-1])
    first = analysis_time <= midpoint
    second = analysis_time >= midpoint
    frequency_first, _ = dominant_frequency(analysis_time[first], analysis_cl[first])
    frequency_second, _ = dominant_frequency(analysis_time[second], analysis_cl[second])
    relative_frequency_difference = abs(frequency_first - frequency_second) / max(
        frequency, 1e-30
    )

    log_path = args.result_dir / "logs" / "log.pimpleFoam"
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    courant = [float(match) for match in re.findall(r"Courant Number mean: \S+ max: (\S+)", log_text)]
    continuity = [float(match) for match in re.findall(r"cumulative = (\S+)", log_text)]

    metrics = {
        "status": "CYLINDER3D_EXTENDED_ANALYSIS_OK",
        "run_id": args.result_dir.name,
        "analysis_start_s": args.analysis_start,
        "analysis_end_s": float(analysis_time[-1]),
        "analysis_samples": int(len(analysis_time)),
        "mean_cd": float(np.mean(analysis_cd)),
        "std_cd": float(np.std(analysis_cd)),
        "mean_cl": float(np.mean(analysis_cl)),
        "rms_cl": float(np.sqrt(np.mean(analysis_cl**2))),
        "peak_to_peak_cl": float(np.ptp(analysis_cl)),
        "dominant_frequency_hz": frequency,
        "strouhal_number": frequency * args.diameter / args.velocity,
        "frequency_first_half_hz": frequency_first,
        "frequency_second_half_hz": frequency_second,
        "relative_frequency_difference": relative_frequency_difference,
        "max_courant_number": max(courant),
        "last_cumulative_continuity_error": continuity[-1],
        "validation_status": "extended_pilot_not_final_reference",
    }

    analysis_dir = args.result_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    with (analysis_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)

    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(time, cd, color="#1565c0", linewidth=1)
    axes[0].axvspan(time[0], args.analysis_start, color="#dddddd", alpha=0.5)
    display_cd = cd[time >= 1.0]
    cd_low, cd_high = np.percentile(display_cd, [0.5, 99.5])
    cd_pad = 0.1 * (cd_high - cd_low)
    axes[0].set_ylim(cd_low - cd_pad, cd_high + cd_pad)
    axes[0].text(
        0.01,
        0.95,
        "Initial impulsive Cd values are outside the displayed y-range",
        transform=axes[0].transAxes,
        va="top",
        fontsize=9,
        color="#555555",
    )
    axes[0].set_ylabel("Cd")
    axes[0].grid(alpha=0.25)
    axes[1].plot(time, cl, color="#c62828", linewidth=1)
    axes[1].axvspan(time[0], args.analysis_start, color="#dddddd", alpha=0.5)
    axes[1].set_ylabel("Cl")
    axes[1].set_xlabel("Physical time [s]")
    axes[1].grid(alpha=0.25)
    figure.suptitle("Extended 3D cylinder pilot, Re=200 — PILOT, NOT VALIDATED")
    figure.tight_layout()
    figure.savefig(analysis_dir / "force_history.png", dpi=160)
    plt.close(figure)

    dt = float(np.median(np.diff(analysis_time)))
    uniform_time = np.arange(analysis_time[0], analysis_time[-1] + 0.5 * dt, dt)
    uniform_cl = np.interp(uniform_time, analysis_time, analysis_cl)
    spectrum = np.abs(np.fft.rfft((uniform_cl - np.mean(uniform_cl)) * np.hanning(len(uniform_cl))))
    frequencies = np.fft.rfftfreq(len(uniform_cl), dt)
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(frequencies, spectrum, color="#6a1b9a")
    axis.axvline(frequency, color="black", linestyle="--", label=f"peak {frequency:.4g} Hz")
    axis.set_xlim(0, min(3.0, frequencies[-1]))
    axis.set_xlabel("Frequency [Hz]")
    axis.set_ylabel("Windowed FFT amplitude")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(analysis_dir / "lift_spectrum.png", dpi=160)
    plt.close(figure)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
