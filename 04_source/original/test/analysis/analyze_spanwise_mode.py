"""Quantify spanwise Fourier modes in sampled 3D cylinder-wake profiles."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COMPONENTS = {
    "Ux": 3,
    "Uy": 4,
    "Uz": 5,
    "omega_x": 6,
    "omega_y": 7,
    "omega_z": 8,
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--span", type=float, default=0.04)
    parser.add_argument("--diameter", type=float, default=0.01)
    parser.add_argument("--max-physical-mode", type=int, default=13)
    return parser.parse_args()


def periodic_resample(z: np.ndarray, values: np.ndarray, span: float) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(z)
    z, values = z[order], values[order]
    count = 108
    target = np.linspace(-0.5 * span, 0.5 * span, count, endpoint=False)
    extended_z = np.concatenate([z - span, z, z + span])
    extended_values = np.concatenate([values, values, values])
    return target, np.interp(target, extended_z, extended_values)


def mode_metrics(values: np.ndarray, span: float, max_mode: int) -> dict[str, float | int]:
    fluctuation = values - np.mean(values)
    spectrum = np.fft.rfft(fluctuation)
    energy = np.abs(spectrum) ** 2
    upper = min(max_mode, len(energy) - 1)
    physical_energy = energy[1 : upper + 1]
    total = float(np.sum(physical_energy))
    dominant = int(np.argmax(physical_energy)) + 1 if total > 0 else 0
    return {
        "spanwise_mean": float(np.mean(values)),
        "spanwise_rms_fluctuation": float(np.sqrt(np.mean(fluctuation ** 2))),
        "dominant_mode": dominant,
        "dominant_wavelength_m": float(span / dominant) if dominant else float("nan"),
        "mode1_energy_fraction": float(energy[1] / total) if total > 0 else 0.0,
        "mode2_energy_fraction": float(energy[2] / total) if total > 0 else 0.0,
        "mode3_energy_fraction": float(energy[3] / total) if total > 0 else 0.0,
    }


def main() -> None:
    args = arguments()
    files = sorted(args.samples_dir.glob("*_U_vorticity.xy"))
    if not files:
        raise FileNotFoundError(f"No sampled profiles in {args.samples_dir}")

    rows: list[dict[str, float | int | str]] = []
    omega_profiles: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for path in files:
        table = np.loadtxt(path)
        if table.shape[1] != 9:
            raise ValueError(f"Expected 9 columns in {path}, found {table.shape[1]}")
        for component, column in COMPONENTS.items():
            z, values = periodic_resample(table[:, 2], table[:, column], args.span)
            metrics = mode_metrics(values, args.span, args.max_physical_mode)
            rows.append({"line": path.stem.removesuffix("_U_vorticity"),
                         "component": component, **metrics})
            if component == "omega_x":
                fluctuation = values - np.mean(values)
                energy = np.abs(np.fft.rfft(fluctuation)) ** 2
                omega_profiles.append((path.stem.removesuffix("_U_vorticity"), z,
                                       values, energy))

    omega_rows = [row for row in rows if row["component"] == "omega_x"]
    mode1_fractions = np.array([row["mode1_energy_fraction"] for row in omega_rows])
    dominant_modes = [int(row["dominant_mode"]) for row in omega_rows]
    summary = {
        "status": "SPANWISE_MODE_ANALYSIS_OK",
        "sample_files": len(files),
        "span_m": args.span,
        "diameter_m": args.diameter,
        "span_over_diameter": args.span / args.diameter,
        "expected_mode_a_fourier_mode": 1,
        "expected_mode_a_wavelength_over_diameter": args.span / args.diameter,
        "omega_x_lines_dominated_by_mode1": sum(mode == 1 for mode in dominant_modes),
        "omega_x_total_lines": len(omega_rows),
        "omega_x_mode1_energy_fraction_mean": float(np.mean(mode1_fractions)),
        "omega_x_mode1_energy_fraction_min": float(np.min(mode1_fractions)),
        "omega_x_mode1_energy_fraction_max": float(np.max(mode1_fractions)),
        "interpretation": (
            "mode1_is_consistent_with_a_4D_spanwise_structure_but_single_time_"
            "and_line_samples_do_not_alone_prove_Mode_A"
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "spanwise_mode_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (args.output_dir / "spanwise_mode_metrics.csv").open(
        "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    for axis, (name, z, values, _) in zip(axes.flat, omega_profiles):
        axis.plot(z / args.diameter, values, color="#1565c0")
        axis.set_title(name)
        axis.set_ylabel(r"$\omega_x$ [1/s]")
        axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("z/D")
    figure.suptitle("Streamwise-vorticity profiles across the 4D periodic span")
    figure.tight_layout()
    figure.savefig(args.output_dir / "omega_x_spanwise_profiles.png", dpi=170)
    plt.close(figure)

    figure, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    modes = np.arange(args.max_physical_mode + 1)
    for axis, (name, _, _, energy) in zip(axes.flat, omega_profiles):
        normalized = energy[: args.max_physical_mode + 1]
        normalized = normalized / np.sum(normalized[1:])
        axis.bar(modes[1:], normalized[1:], color="#6a1b9a")
        axis.set_title(name)
        axis.set_ylabel("Energy fraction")
        axis.grid(axis="y", alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("Spanwise Fourier mode k")
    figure.suptitle("Fourier energy of streamwise-vorticity fluctuations")
    figure.tight_layout()
    figure.savefig(args.output_dir / "omega_x_spanwise_spectra.png", dpi=170)
    plt.close(figure)

    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
