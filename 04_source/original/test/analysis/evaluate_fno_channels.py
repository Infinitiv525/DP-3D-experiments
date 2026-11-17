"""Per-channel validation diagnostics for frozen FNO pilot checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from fno_model import FNO3dStep, normalized_divergence_loss
from train_fno_multistep import (
    load_scenario_table,
    model_step,
    normalized_state,
    scenario_files,
    static_channels,
)


METRIC_COLUMNS = [
    "u_normalized_mse",
    "v_normalized_mse",
    "w_normalized_mse",
    "p_normalized_mse",
    "u_rmse_m_s",
    "v_rmse_m_s",
    "w_rmse_m_s",
    "p_rmse_m2_s2",
    "velocity_relative_l2",
    "pressure_gauge_relative_l2",
    "wake_normalized_mse",
    "normalized_divergence_rms",
]


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Checkpoint must use label=/absolute/path.pt")
    label, path = value.split("=", 1)
    return label, Path(path)


def metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    wake: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    spacing: torch.Tensor,
    divergence_scale: torch.Tensor,
) -> dict[str, float]:
    count = mask.sum().clamp_min(1.0)
    normalized_mse = (((prediction - target) ** 2) * mask).sum(dim=(0, 2, 3, 4)) / count
    prediction_physical = prediction * std + mean
    target_physical = target * std + mean
    physical_rmse = torch.sqrt(
        (((prediction_physical - target_physical) ** 2) * mask).sum(dim=(0, 2, 3, 4))
        / count
    )
    velocity_numerator = (
        ((prediction_physical[:, :3] - target_physical[:, :3]) ** 2) * mask
    ).sum()
    velocity_denominator = ((target_physical[:, :3] ** 2) * mask).sum().clamp_min(1e-20)
    pressure_prediction = prediction_physical[:, 3:4]
    pressure_target = target_physical[:, 3:4]
    pressure_prediction = pressure_prediction - (pressure_prediction * mask).sum() / count
    pressure_target = pressure_target - (pressure_target * mask).sum() / count
    pressure_numerator = (((pressure_prediction - pressure_target) ** 2) * mask).sum()
    pressure_denominator = ((pressure_target**2) * mask).sum().clamp_min(1e-20)
    wake_count = wake.sum().clamp_min(1.0)
    wake_mse = (((prediction - target) ** 2) * wake).sum() / (wake_count * 4.0)
    divergence = normalized_divergence_loss(
        prediction, mask, mean, std, spacing, divergence_scale
    )
    return {
        "u_normalized_mse": float(normalized_mse[0]),
        "v_normalized_mse": float(normalized_mse[1]),
        "w_normalized_mse": float(normalized_mse[2]),
        "p_normalized_mse": float(normalized_mse[3]),
        "u_rmse_m_s": float(physical_rmse[0]),
        "v_rmse_m_s": float(physical_rmse[1]),
        "w_rmse_m_s": float(physical_rmse[2]),
        "p_rmse_m2_s2": float(physical_rmse[3]),
        "velocity_relative_l2": float(torch.sqrt(velocity_numerator / velocity_denominator)),
        "pressure_gauge_relative_l2": float(
            torch.sqrt(pressure_numerator / pressure_denominator)
        ),
        "wake_normalized_mse": float(wake_mse),
        "normalized_divergence_rms": math.sqrt(max(0.0, float(divergence))),
    }


def aggregate(records: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    variants = sorted({str(record["variant"]) for record in records})
    scenarios = sorted({str(record["scenario_id"]) for record in records})
    for variant in variants:
        result[variant] = {}
        for scenario in scenarios + ["all"]:
            subset = [
                record
                for record in records
                if record["variant"] == variant
                and (scenario == "all" or record["scenario_id"] == scenario)
            ]
            tail = [record for record in subset if int(record["step"]) >= 151]
            result[variant][scenario] = {
                "mean": {
                    column: float(np.mean([float(record[column]) for record in subset]))
                    for column in METRIC_COLUMNS
                },
                "tail_50_mean": {
                    column: float(np.mean([float(record[column]) for record in tail]))
                    for column in METRIC_COLUMNS
                },
                "final": {
                    column: float(np.mean([float(record[column]) for record in subset if int(record["step"]) == 200]))
                    for column in METRIC_COLUMNS
                },
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=parse_checkpoint, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")

    table = load_scenario_table(Path(config["scenario_table"]))
    validation = [item for item in table.values() if item.split == config["validation_split"]]
    protected = [
        item.scenario_id for item in table.values() if item.split in config["protected_splits"]
    ]
    statistics = json.loads(Path(config["training_statistics"]).read_text(encoding="utf-8"))
    mean_np = np.asarray(statistics["mean"], dtype=np.float32)
    std_np = np.asarray(statistics["std"], dtype=np.float32)
    parameter_mean = np.asarray(statistics["scenario_parameter_mean"], dtype=np.float32)
    parameter_std = np.asarray(statistics["scenario_parameter_std"], dtype=np.float32)
    mean = torch.from_numpy(mean_np.reshape(1, 4, 1, 1, 1)).to(device)
    std = torch.from_numpy(std_np.reshape(1, 4, 1, 1, 1)).to(device)
    model_config = config["model"]
    dataset_root = Path(config["dataset_root"])
    records: list[dict[str, object]] = []

    for scenario in validation:
        paths = scenario_files(dataset_root, scenario.scenario_id)
        initial_np, valid, axes = normalized_state(paths[0], mean_np, std_np)
        static_np, wake_np, spacing_np = static_channels(
            axes, valid, scenario, parameter_mean, parameter_std
        )
        initial = torch.from_numpy(np.moveaxis(initial_np, -1, 0)[None].copy()).to(device)
        mask = torch.from_numpy(valid[None, None].astype(np.float32)).to(device)
        static = torch.from_numpy(static_np[None]).to(device)
        wake = torch.from_numpy(wake_np[None]).to(device)
        spacing = torch.from_numpy(spacing_np[None]).to(device)
        divergence_scale = torch.tensor([scenario.velocity / scenario.diameter], device=device)
        states = {"persistence": initial.clone()}
        models: dict[str, FNO3dStep] = {}
        for label, checkpoint_path in args.checkpoint:
            model = FNO3dStep(
                input_channels=int(model_config["input_channels"]),
                width=int(model_config["width"]),
                layers=int(model_config["layers"]),
                modes=(
                    int(model_config["modes_x"]),
                    int(model_config["modes_y"]),
                    int(model_config["modes_z"]),
                ),
                padding=(int(model_config["padding_x"]), int(model_config["padding_y"])),
            ).to(device)
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            models[label] = model
            states[label] = initial.clone()
        with torch.inference_mode():
            for step in range(1, 201):
                target_np, target_valid, _ = normalized_state(paths[step], mean_np, std_np)
                if not np.array_equal(valid, target_valid):
                    raise RuntimeError(f"Mask changed in {scenario.scenario_id}")
                target = torch.from_numpy(np.moveaxis(target_np, -1, 0)[None].copy()).to(device)
                for label, model in models.items():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        states[label] = model_step(model, states[label], static, mask, initial)
                for label, prediction in states.items():
                    values = metrics(
                        prediction,
                        target,
                        mask,
                        wake,
                        mean,
                        std,
                        spacing,
                        divergence_scale,
                    )
                    records.append(
                        {
                            "variant": label,
                            "scenario_id": scenario.scenario_id,
                            "step": step,
                            "time_s": 160.0 + 0.2 * step,
                            **values,
                        }
                    )
    csv_path = args.output_dir / "per_channel_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["variant", "scenario_id", "step", "time_s", *METRIC_COLUMNS],
        )
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "status": "FNO_CHANNEL_DIAGNOSTICS_OK",
        "validation_scenarios_read": [item.scenario_id for item in validation],
        "protected_scenarios_not_read": protected,
        "checkpoints": {label: str(path) for label, path in args.checkpoint},
        "records": len(records),
        "metrics": aggregate(records),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
