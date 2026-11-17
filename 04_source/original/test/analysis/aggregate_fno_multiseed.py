"""Aggregate the paired three-seed FNO physics pilot without opening test data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = {
    "selection_score": lambda data: data["best_selection_score"],
    "mean_mse": lambda data: data["validation"]["aggregate"]["mean_mse"],
    "tail_50_mean_mse": lambda data: data["validation"]["aggregate"]["tail_50_mean_mse"],
    "normalized_divergence_rms": lambda data: data["validation"]["aggregate"][
        "mean_normalized_divergence_rms"
    ],
    "latency_p99_ms": lambda data: data["latency"]["p99_ms"],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, object]] = []
    for seed, variants in manifest.items():
        for variant, path_value in variants.items():
            data = json.loads(Path(path_value).read_text(encoding="utf-8"))
            records.append(
                {
                    "seed": int(seed),
                    "variant": variant,
                    "best_epoch": data["best_epoch"],
                    **{name: float(extractor(data)) for name, extractor in METRICS.items()},
                }
            )
    csv_path = args.output_dir / "multiseed_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    summary: dict[str, object] = {
        "status": "FNO_MULTISEED_PILOT_OK",
        "seed_count": len(manifest),
        "seeds": [int(seed) for seed in manifest],
        "test_splits_read": False,
        "descriptive_only_no_significance_claim": True,
        "variants": {},
        "paired_physics_vs_control": {},
    }
    for variant in ("base", "control", "physics"):
        subset = [record for record in records if record["variant"] == variant]
        summary["variants"][variant] = {
            metric: {
                "mean": float(np.mean([record[metric] for record in subset])),
                "sample_std": float(np.std([record[metric] for record in subset], ddof=1)),
                "median": float(np.median([record[metric] for record in subset])),
            }
            for metric in METRICS
        }
    for metric in METRICS:
        differences = []
        percent_differences = []
        wins = 0
        for seed in manifest:
            control = next(
                record[metric]
                for record in records
                if record["seed"] == int(seed) and record["variant"] == "control"
            )
            physics = next(
                record[metric]
                for record in records
                if record["seed"] == int(seed) and record["variant"] == "physics"
            )
            differences.append(physics - control)
            percent_differences.append(100.0 * (physics - control) / control)
            if physics < control:
                wins += 1
        summary["paired_physics_vs_control"][metric] = {
            "mean_difference_physics_minus_control": float(np.mean(differences)),
            "mean_percent_difference": float(np.mean(percent_differences)),
            "median_percent_difference": float(np.median(percent_differences)),
            "physics_wins_lower_is_better": wins,
            "pairs": len(differences),
            "per_seed_percent_difference": percent_differences,
        }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 4, figsize=(16, 4.5), constrained_layout=True)
    plot_metrics = [
        ("selection_score", "Selection score"),
        ("mean_mse", "Mean MSE"),
        ("tail_50_mean_mse", "Tail-50 MSE"),
        ("normalized_divergence_rms", "Normalized divergence RMS"),
    ]
    for axis, (metric, title) in zip(axes, plot_metrics):
        for seed in sorted(int(value) for value in manifest):
            control = next(
                record[metric]
                for record in records
                if record["seed"] == seed and record["variant"] == "control"
            )
            physics = next(
                record[metric]
                for record in records
                if record["seed"] == seed and record["variant"] == "physics"
            )
            axis.plot([0, 1], [control, physics], marker="o", label=str(seed))
        axis.set_xticks([0, 1], ["Control", "Physics"])
        axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.25)
    axes[0].legend(title="Seed", fontsize=8)
    figure.suptitle("Paired three-seed FNO pilot (lower is better)", fontweight="bold")
    figure.savefig(args.output_dir / "paired_multiseed_ablation.png", dpi=180)
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
