"""Plot per-channel rollout diagnostics produced by evaluate_fno_channels.py."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = {"physics": "FNO + physics", "control": "FNO control", "persistence": "Persistence"}
COLORS = {"physics": "#0072B2", "control": "#D55E00", "persistence": "#666666"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with args.csv.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    variants = ["physics", "control", "persistence"]
    channels = [
        ("u_normalized_mse", "Ux"),
        ("v_normalized_mse", "Uy"),
        ("w_normalized_mse", "Uz"),
        ("p_normalized_mse", "p"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, constrained_layout=True)
    for axis, (column, title) in zip(axes.flat, channels):
        for variant in variants:
            means = []
            for step in range(1, 201):
                values = [
                    float(record[column])
                    for record in records
                    if record["variant"] == variant and int(record["step"]) == step
                ]
                means.append(np.mean(values))
            axis.plot(np.arange(1, 201) * 0.2, means, label=LABELS[variant], color=COLORS[variant])
        axis.set_yscale("log")
        axis.set_title(f"Normalized MSE — {title}")
        axis.set_xlabel("Rollout horizon [s]")
        axis.set_ylabel("MSE")
        axis.grid(True, which="both", alpha=0.25)
    axes[0, 0].legend()
    figure.savefig(args.output_dir / "per_channel_rollout.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    diagnostics = [
        ("velocity_relative_l2", "Velocity relative L2"),
        ("pressure_gauge_relative_l2", "Gauge pressure relative L2"),
        ("normalized_divergence_rms", "Normalized divergence RMS"),
    ]
    for axis, (column, title) in zip(axes, diagnostics):
        for variant in variants:
            means = []
            for step in range(1, 201):
                values = [
                    float(record[column])
                    for record in records
                    if record["variant"] == variant and int(record["step"]) == step
                ]
                means.append(np.mean(values))
            axis.plot(np.arange(1, 201) * 0.2, means, label=LABELS[variant], color=COLORS[variant])
        axis.set_title(title)
        axis.set_xlabel("Rollout horizon [s]")
        axis.grid(True, alpha=0.25)
    axes[0].legend()
    figure.savefig(args.output_dir / "physical_rollout_diagnostics.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
