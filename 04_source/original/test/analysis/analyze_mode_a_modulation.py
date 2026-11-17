"""Analyse low-frequency modulation in the full fine-grid force history."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_grid_verification import crossing_frequency, load_table


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-result", type=Path, required=True)
    parser.add_argument(
        "--continuation-result", type=Path, action="append", required=True,
        help="Continuation result directory; may be supplied more than once.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis-start", type=float, default=30.0)
    parser.add_argument("--diameter", type=float, default=0.01)
    parser.add_argument("--velocity", type=float, default=0.02)
    return parser.parse_args()


def analytic_envelope(signal: np.ndarray) -> np.ndarray:
    count = len(signal)
    spectrum = np.fft.fft(signal)
    multiplier = np.zeros(count)
    if count % 2 == 0:
        multiplier[0] = multiplier[count // 2] = 1.0
        multiplier[1 : count // 2] = 2.0
    else:
        multiplier[0] = 1.0
        multiplier[1 : (count + 1) // 2] = 2.0
    return np.abs(np.fft.ifft(spectrum * multiplier))


def moving_average(signal: np.ndarray, samples: int) -> np.ndarray:
    samples = max(1, int(samples))
    cumulative = np.cumsum(np.insert(signal, 0, 0.0))
    valid = (cumulative[samples:] - cumulative[:-samples]) / samples
    left = samples // 2
    right = len(signal) - len(valid) - left
    return np.pad(valid, (left, right), mode="edge")


def main() -> None:
    args = arguments()
    initial_file = next(args.initial_result.glob("tables/forceCoefficients/*/coefficient.dat"))
    initial = load_table(initial_file)
    segments = [initial]
    for result_dir in args.continuation_result:
        continuation_file = next(
            result_dir.glob("tables/forceCoefficients/*/coefficient.dat"))
        segments.append(load_table(continuation_file))
    combined = np.vstack(segments)
    combined = combined[np.argsort(combined[:, 0], kind="stable")]
    _, unique_indices = np.unique(combined[:, 0], return_index=True)
    combined = combined[np.sort(unique_indices)]
    time, cd, cl = combined[:, 0], combined[:, 1], combined[:, 4]
    mask = time >= args.analysis_start
    time, cd, cl = time[mask], cd[mask], cl[mask]

    sample_dt = 0.02
    uniform_time = np.arange(time[0], time[-1] + 0.5 * sample_dt, sample_dt)
    uniform_cd = np.interp(uniform_time, time, cd)
    uniform_cl = np.interp(uniform_time, time, cl)
    centred_cl = uniform_cl - np.mean(uniform_cl)
    envelope = analytic_envelope(centred_cl)
    smooth_envelope = moving_average(envelope, round(5.0 / sample_dt))

    primary_frequency, observed_periods = crossing_frequency(uniform_time, uniform_cl)
    detrended_envelope = smooth_envelope - np.polyval(
        np.polyfit(uniform_time, smooth_envelope, 1), uniform_time)
    window = np.hanning(len(detrended_envelope))
    frequencies = np.fft.rfftfreq(len(detrended_envelope), sample_dt)
    envelope_spectrum = np.abs(np.fft.rfft(detrended_envelope * window))
    low_band = (frequencies >= 0.005) & (frequencies <= 0.1)
    low_indices = np.flatnonzero(low_band)
    modulation_index = low_indices[int(np.argmax(envelope_spectrum[low_band]))]
    modulation_frequency = float(frequencies[modulation_index])
    modulation_period = 1.0 / modulation_frequency

    window_seconds, step_seconds = 20.0, 5.0
    window_rows = []
    start = args.analysis_start
    while start + window_seconds <= uniform_time[-1] + 1e-9:
        end = start + window_seconds
        selected = (uniform_time >= start) & (uniform_time <= end)
        local_frequency, local_periods = crossing_frequency(
            uniform_time[selected], uniform_cl[selected])
        window_rows.append({
            "start_s": start,
            "end_s": end,
            "mean_cd": float(np.mean(uniform_cd[selected])),
            "rms_cl": float(np.sqrt(np.mean(uniform_cl[selected] ** 2))),
            "mean_envelope_cl": float(np.mean(smooth_envelope[selected])),
            "frequency_hz": local_frequency,
            "strouhal_number": local_frequency * args.diameter / args.velocity,
            "observed_periods": local_periods,
        })
        start += step_seconds

    primary_st = primary_frequency * args.diameter / args.velocity
    modulation_st = modulation_frequency * args.diameter / args.velocity
    analysed_duration = float(uniform_time[-1] - uniform_time[0])
    summary = {
        "status": "CYLINDER3D_MODE_A_MODULATION_ANALYSIS_OK",
        "analysis_start_s": float(uniform_time[0]),
        "analysis_end_s": float(uniform_time[-1]),
        "analysis_duration_s": analysed_duration,
        "primary_frequency_hz": primary_frequency,
        "primary_strouhal_number": primary_st,
        "observed_primary_periods": observed_periods,
        "candidate_modulation_frequency_hz": modulation_frequency,
        "candidate_modulation_period_s": modulation_period,
        "candidate_modulation_strouhal_number": modulation_st,
        "modulation_to_primary_frequency_ratio": modulation_frequency / primary_frequency,
        "observed_modulation_cycles": analysed_duration * modulation_frequency,
        "interpretation": (
            "candidate_low_frequency_quasiperiodic_modulation; force history alone "
            "does not prove the spatial Mode A mechanism"
        ),
        "reference_doi": "10.1017/jfm.2026.11225",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "mode_a_modulation.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (args.output_dir / "sliding_window_metrics.csv").open(
        "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(window_rows[0]))
        writer.writeheader()
        writer.writerows(window_rows)

    figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(uniform_time, uniform_cd, color="#1565c0", linewidth=0.7)
    axes[0].set_ylabel("Cd")
    axes[1].plot(uniform_time, uniform_cl, color="#c62828", linewidth=0.7, alpha=0.65)
    axes[1].plot(uniform_time, smooth_envelope, color="#111111", linewidth=1.2,
                 label="smoothed amplitude envelope")
    axes[1].plot(uniform_time, -smooth_envelope, color="#111111", linewidth=1.2)
    axes[1].set_ylabel("Cl")
    axes[1].legend()
    centres = [0.5 * (row["start_s"] + row["end_s"]) for row in window_rows]
    axes[2].plot(centres, [row["mean_cd"] for row in window_rows], marker="o",
                 label="window mean Cd")
    second_axis = axes[2].twinx()
    second_axis.plot(centres, [row["rms_cl"] for row in window_rows], marker="s",
                     color="#2e7d32", label="window RMS Cl")
    axes[2].set_ylabel("Mean Cd")
    second_axis.set_ylabel("RMS Cl")
    axes[2].set_xlabel("Physical time [s]")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle("Fine-grid force history and low-frequency amplitude modulation")
    figure.tight_layout()
    figure.savefig(args.output_dir / "mode_a_modulation_history.png", dpi=170)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(frequencies, envelope_spectrum, color="#6a1b9a")
    axis.axvline(modulation_frequency, color="black", linestyle="--",
                 label=f"candidate {modulation_frequency:.4f} Hz")
    axis.set_xlim(0.0, 0.1)
    axis.set_xlabel("Envelope frequency [Hz]")
    axis.set_ylabel("Windowed spectrum amplitude")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "modulation_spectrum.png", dpi=170)
    plt.close(figure)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
