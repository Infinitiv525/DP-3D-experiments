"""Create a compact quality-control preview of an exported uniform snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = np.load(args.npz)
    velocity, pressure, valid = data["U"], data["p"], data["valid"].astype(bool)
    x, y, z = data["x"], data["y"], data["z"]
    middle = len(z) // 2
    speed = np.linalg.norm(velocity, axis=-1)
    speed_slice = np.ma.masked_where(~valid[:, :, middle], speed[:, :, middle])
    pressure_slice = np.ma.masked_where(~valid[:, :, middle], pressure[:, :, middle])
    extent = [float(x[0]), float(x[-1]), float(y[0]), float(y[-1])]

    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, sharey=True)
    first = axes[0].imshow(speed_slice.T, origin="lower", extent=extent,
                           aspect="equal", cmap="viridis")
    figure.colorbar(first, ax=axes[0], label="|U| [m/s]")
    axes[0].set_title(f"Velocity magnitude, z={z[middle]:.4g} m")
    second = axes[1].imshow(pressure_slice.T, origin="lower", extent=extent,
                            aspect="equal", cmap="coolwarm")
    figure.colorbar(second, ax=axes[1], label="kinematic p [m²/s²]")
    axes[1].set_title("Pressure")
    for axis in axes:
        axis.set_ylabel("y [m]")
        axis.grid(False)
    axes[1].set_xlabel("x [m]")
    figure.suptitle(args.npz.stem)
    figure.tight_layout()
    figure.savefig(args.output, dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    main()
