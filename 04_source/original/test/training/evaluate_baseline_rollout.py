"""Autoregressive rollout check on validation scenarios only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from train_baseline_smoke import OneStepCNN, load_snapshot, scenario_files


def build_static_features(
    valid: np.ndarray, coordinates: tuple[np.ndarray, ...], device: torch.device
) -> torch.Tensor:
    coordinate_channels = np.meshgrid(
        *[
            2.0 * (axis - axis.min()) / (axis.max() - axis.min()) - 1.0
            for axis in coordinates
        ],
        indexing="ij",
    )
    static = np.concatenate(
        (
            valid[..., None].astype(np.float32),
            np.stack(coordinate_channels, axis=-1).astype(np.float32),
        ),
        axis=-1,
    )
    return torch.from_numpy(np.moveaxis(static, -1, 0)[None].copy()).to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    config = yaml.safe_load((args.run_dir / "config.yaml").read_text(encoding="utf-8"))
    stats = json.loads((args.run_dir / "training_statistics.json").read_text(encoding="utf-8"))
    mean = np.asarray(stats["mean"], dtype=np.float32).reshape(1, 1, 1, 4)
    std = np.asarray(stats["std"], dtype=np.float32).reshape(1, 1, 1, 4)
    device = torch.device("cuda:0")
    model = OneStepCNN(int(config["model_width"]), int(config["residual_blocks"])).to(device)
    model.load_state_dict(torch.load(args.run_dir / "best_model.pt", map_location=device, weights_only=True))
    model.eval()

    dataset_root = Path(config["dataset_root"])
    validation_scenarios = ["va001", "va002"]
    all_records: list[dict[str, float | str | int]] = []
    scenario_summaries: list[dict[str, float | str | bool]] = []
    for scenario in validation_scenarios:
        files = scenario_files(dataset_root, scenario)
        initial, valid, coordinates = load_snapshot(files[0])
        current_normalized = (initial - mean) / std
        current = torch.from_numpy(np.moveaxis(current_normalized, -1, 0)[None].copy()).to(device)
        initial_tensor = current.clone()
        static = build_static_features(valid, coordinates, device)
        mask = torch.from_numpy(valid[None, None].astype(np.float32)).to(device)
        records: list[dict[str, float | str | int]] = []

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for step, target_path in enumerate(files[1:], start=1):
                features = torch.cat((current, static), dim=1)
                prediction = model(features, current).float()
                target_np, target_valid, _ = load_snapshot(target_path)
                if not np.array_equal(valid, target_valid):
                    raise RuntimeError(f"Mask changed in {scenario}")
                target_normalized = (target_np - mean) / std
                target = torch.from_numpy(
                    np.moveaxis(target_normalized, -1, 0)[None].copy()
                ).to(device)
                squared = (prediction - target) ** 2 * mask
                persistence_squared = (initial_tensor - target) ** 2 * mask
                denomin = float(mask.sum())
                channel_mse = squared.sum(dim=(0, 2, 3, 4)) / denomin
                physical_rmse = torch.sqrt(channel_mse) * torch.from_numpy(
                    std.reshape(4)
                ).to(device)
                record: dict[str, float | str | int] = {
                    "scenario_id": scenario,
                    "step": step,
                    "time_s": 160.0 + 0.2 * step,
                    "rollout_mse_normalized": float(squared.sum() / (mask.sum() * 4)),
                    "persistence_mse_normalized": float(
                        persistence_squared.sum() / (mask.sum() * 4)
                    ),
                    "u_rmse_m_s": float(physical_rmse[0]),
                    "v_rmse_m_s": float(physical_rmse[1]),
                    "w_rmse_m_s": float(physical_rmse[2]),
                    "p_rmse_m2_s2": float(physical_rmse[3]),
                }
                records.append(record)
                current = prediction

        finite = all(
            np.isfinite(float(record["rollout_mse_normalized"])) for record in records
        )
        scenario_summaries.append(
            {
                "scenario_id": scenario,
                "finite_all_steps": finite,
                "steps": len(records),
                "mean_rollout_mse_normalized": float(
                    np.mean([record["rollout_mse_normalized"] for record in records])
                ),
                "final_rollout_mse_normalized": float(records[-1]["rollout_mse_normalized"]),
                "mean_persistence_mse_normalized": float(
                    np.mean([record["persistence_mse_normalized"] for record in records])
                ),
                "final_persistence_mse_normalized": float(
                    records[-1]["persistence_mse_normalized"]
                ),
            }
        )
        all_records.extend(records)

    with (args.output_dir / "rollout_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_records[0]))
        writer.writeheader()
        writer.writerows(all_records)
    summary = {
        "status": "BASELINE_VALIDATION_ROLLOUT_OK",
        "validation_only": True,
        "test_scenarios_read": False,
        "rollout_steps": 200,
        "physical_horizon_s": 40.0,
        "scenario_summaries": scenario_summaries,
        "scope_note": "Smoke-model stability diagnostic, not final test evaluation.",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
