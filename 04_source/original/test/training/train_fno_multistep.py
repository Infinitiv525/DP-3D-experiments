"""Train and validate a closed-loop multistep 3D FNO stability pilot.

Only train and validation scenarios are opened. The protected test splits remain
untouched until the experimental protocol and model configuration are frozen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from fno_model import FNO3dStep, masked_channel_mse, normalized_divergence_loss


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    split: str
    velocity: float
    diameter: float
    center_x: float
    center_y: float

    @property
    def parameters(self) -> np.ndarray:
        return np.asarray(
            [self.velocity, self.diameter, self.center_x, self.center_y], dtype=np.float32
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_scenario_table(path: Path) -> dict[str, Scenario]:
    result: dict[str, Scenario] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            scenario = Scenario(
                scenario_id=row["scenario_id"],
                split=row["split"],
                velocity=float(row["u_m_s"]),
                diameter=float(row["diameter_m"]),
                center_x=float(row["center_x_m"]),
                center_y=float(row["center_y_m"]),
            )
            result[scenario.scenario_id] = scenario
    return result


def scenario_files(dataset_root: Path, scenario_id: str) -> list[Path]:
    index_path = dataset_root / scenario_id / "series_index.csv"
    with index_path.open(newline="", encoding="utf-8") as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: float(row["time_s"]))
    files = [dataset_root / scenario_id / row["filename"] for row in rows]
    if len(files) != 201 or any(not path.is_file() for path in files):
        raise RuntimeError(f"Incomplete scenario: {scenario_id}")
    return files


def load_snapshot(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    with np.load(path, allow_pickle=False) as data:
        fields = np.concatenate((data["U"], data["p"][..., None]), axis=-1).astype(np.float32)
        valid = data["valid"].astype(bool)
        axes = (data["x"].copy(), data["y"].copy(), data["z"].copy())
    return fields, valid, axes


def normalized_state(
    path: Path, mean: np.ndarray, std: np.ndarray
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    fields, valid, axes = load_snapshot(path)
    fields = (fields - mean.reshape(1, 1, 1, 4)) / std.reshape(1, 1, 1, 4)
    fields[~valid] = 0.0
    return fields, valid, axes


def static_channels(
    axes: tuple[np.ndarray, ...],
    valid: np.ndarray,
    scenario: Scenario,
    parameter_mean: np.ndarray,
    parameter_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, z = axes
    x_normalized = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
    y_normalized = 2.0 * (y - y.min()) / (y.max() - y.min()) - 1.0
    shape = (len(x), len(y), len(z))
    channels = [valid.astype(np.float32)]
    channels.append(np.broadcast_to(x_normalized[:, None, None], shape).astype(np.float32))
    channels.append(np.broadcast_to(y_normalized[None, :, None], shape).astype(np.float32))
    normalized_parameters = (scenario.parameters - parameter_mean) / parameter_std
    channels.extend(np.full(shape, value, dtype=np.float32) for value in normalized_parameters)
    static = np.stack(channels, axis=0)
    x_grid = np.broadcast_to(x[:, None, None], shape)
    y_grid = np.broadcast_to(y[None, :, None], shape)
    wake = (
        valid
        & (x_grid >= scenario.center_x)
        & (x_grid <= scenario.center_x + 10.0 * scenario.diameter)
        & (np.abs(y_grid - scenario.center_y) <= 1.5 * scenario.diameter)
    )
    spacing = np.asarray(
        [np.mean(np.diff(x)), np.mean(np.diff(y)), np.mean(np.diff(z))], dtype=np.float32
    )
    return static, wake[None].astype(np.float32), spacing


class SequenceDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        scenarios: list[Scenario],
        mean: np.ndarray,
        std: np.ndarray,
        parameter_mean: np.ndarray,
        parameter_std: np.ndarray,
        horizon: int,
        maximum_sequences: int,
        seed: int,
    ) -> None:
        self.mean = mean
        self.std = std
        self.parameter_mean = parameter_mean
        self.parameter_std = parameter_std
        self.horizon = horizon
        self.files = {
            scenario.scenario_id: scenario_files(dataset_root, scenario.scenario_id)
            for scenario in scenarios
        }
        self.scenarios = {scenario.scenario_id: scenario for scenario in scenarios}
        candidates = [
            (scenario.scenario_id, start)
            for scenario in scenarios
            for start in range(201 - horizon)
        ]
        generator = np.random.default_rng(seed)
        if len(candidates) > maximum_sequences:
            selected = generator.choice(len(candidates), size=maximum_sequences, replace=False)
            self.samples = [candidates[int(index)] for index in selected]
        else:
            self.samples = candidates

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        scenario_id, start = self.samples[index]
        scenario = self.scenarios[scenario_id]
        paths = self.files[scenario_id]
        current, valid, axes = normalized_state(paths[start], self.mean, self.std)
        targets: list[np.ndarray] = []
        for step in range(1, self.horizon + 1):
            target, target_valid, _ = normalized_state(paths[start + step], self.mean, self.std)
            if not np.array_equal(valid, target_valid):
                raise RuntimeError(f"Mask changed in scenario {scenario_id}")
            targets.append(np.moveaxis(target, -1, 0))
        static, wake, spacing = static_channels(
            axes, valid, scenario, self.parameter_mean, self.parameter_std
        )
        scale = np.asarray(scenario.velocity / scenario.diameter, dtype=np.float32)
        return (
            torch.from_numpy(np.moveaxis(current, -1, 0).copy()),
            torch.from_numpy(np.stack(targets).copy()),
            torch.from_numpy(valid[None].astype(np.float32)),
            torch.from_numpy(static.copy()),
            torch.from_numpy(wake.copy()),
            torch.from_numpy(spacing),
            torch.from_numpy(scale),
        )


def loss_weights(horizon: int) -> tuple[float, float, float]:
    if horizon == 1:
        return 1.0, 0.0, 0.0
    if horizon < 16:
        return 0.5, 0.4, 0.1
    return 0.2, 0.6, 0.2


def rollout_loss(losses: list[torch.Tensor]) -> torch.Tensor:
    first_weight, mean_weight, tail_weight = loss_weights(len(losses))
    stacked = torch.stack(losses)
    tail_start = max(0, len(losses) - max(1, math.ceil(len(losses) / 4)))
    return (
        first_weight * stacked[0]
        + mean_weight * stacked.mean()
        + tail_weight * stacked[tail_start:].mean()
    )


def data_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    fluid_mask: torch.Tensor,
    wake_mask: torch.Tensor,
) -> torch.Tensor:
    return 0.75 * masked_channel_mse(
        prediction, target, fluid_mask
    ) + 0.25 * masked_channel_mse(prediction, target, wake_mask)


def model_step(
    model: FNO3dStep,
    current: torch.Tensor,
    static: torch.Tensor,
    mask: torch.Tensor,
    inlet_reference: torch.Tensor,
) -> torch.Tensor:
    return model(torch.cat((current, static), dim=1), current, mask, inlet_reference)


def train_epoch(
    model: FNO3dStep,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    field_mean: torch.Tensor,
    field_std: torch.Tensor,
    divergence_weight: float,
    divergence_floor_rms: float,
    gradient_clip: float,
) -> dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0,
        "data_loss": 0.0,
        "divergence_raw_loss": 0.0,
        "divergence_excess_loss": 0.0,
        "sequences": 0,
    }
    for current, targets, mask, static, wake, spacing, divergence_scale in loader:
        tensors = [current, targets, mask, static, wake, spacing, divergence_scale]
        current, targets, mask, static, wake, spacing, divergence_scale = [
            value.to(device, non_blocking=True) for value in tensors
        ]
        inlet_reference = current
        optimizer.zero_grad(set_to_none=True)
        step_data: list[torch.Tensor] = []
        step_divergence: list[torch.Tensor] = []
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = current
            for step in range(targets.shape[1]):
                prediction = model_step(model, prediction, static, mask, inlet_reference)
                step_data.append(data_loss(prediction, targets[:, step], mask, wake))
                if divergence_weight > 0.0:
                    step_divergence.append(
                        normalized_divergence_loss(
                            prediction,
                            mask,
                            field_mean,
                            field_std,
                            spacing,
                            divergence_scale,
                        )
                    )
            data_value = rollout_loss(step_data)
            divergence_raw = (
                torch.stack(step_divergence).mean()
                if step_divergence
                else torch.zeros((), device=device)
            )
            divergence_excess = torch.clamp(
                divergence_raw - divergence_floor_rms**2, min=0.0
            )
            loss = data_value + divergence_weight * divergence_excess
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        batch = current.shape[0]
        totals["loss"] += float(loss.detach()) * batch
        totals["data_loss"] += float(data_value.detach()) * batch
        totals["divergence_raw_loss"] += float(divergence_raw.detach()) * batch
        totals["divergence_excess_loss"] += float(divergence_excess.detach()) * batch
        totals["sequences"] += batch
    count = max(1, int(totals["sequences"]))
    return {key: value / count for key, value in totals.items() if key != "sequences"} | {
        "sequences": count
    }


def validate_rollout(
    model: FNO3dStep,
    dataset_root: Path,
    scenarios: list[Scenario],
    mean: np.ndarray,
    std: np.ndarray,
    parameter_mean: np.ndarray,
    parameter_std: np.ndarray,
    device: torch.device,
    field_mean: torch.Tensor,
    field_std: torch.Tensor,
) -> dict[str, object]:
    model.eval()
    result: dict[str, object] = {"scenarios": {}}
    all_model_errors: list[float] = []
    all_tail_errors: list[float] = []
    all_divergence: list[float] = []
    with torch.inference_mode():
        for scenario in scenarios:
            files = scenario_files(dataset_root, scenario.scenario_id)
            initial, valid, axes = normalized_state(files[0], mean, std)
            static_np, wake_np, spacing_np = static_channels(
                axes, valid, scenario, parameter_mean, parameter_std
            )
            current = torch.from_numpy(np.moveaxis(initial, -1, 0)[None].copy()).to(device)
            initial_tensor = current.clone()
            inlet_reference = current.clone()
            mask = torch.from_numpy(valid[None, None].astype(np.float32)).to(device)
            static = torch.from_numpy(static_np[None]).to(device)
            wake = torch.from_numpy(wake_np[None]).to(device)
            spacing = torch.from_numpy(spacing_np[None]).to(device)
            scale = torch.tensor([scenario.velocity / scenario.diameter], device=device)
            model_errors: list[float] = []
            persistence_errors: list[float] = []
            divergence_rms: list[float] = []
            for step in range(1, 201):
                target_np, target_valid, _ = normalized_state(files[step], mean, std)
                if not np.array_equal(valid, target_valid):
                    raise RuntimeError(f"Mask changed in scenario {scenario.scenario_id}")
                target = torch.from_numpy(np.moveaxis(target_np, -1, 0)[None].copy()).to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    current = model_step(model, current, static, mask, inlet_reference)
                    error = data_loss(current, target, mask, wake)
                    persistence = data_loss(initial_tensor, target, mask, wake)
                    divergence = normalized_divergence_loss(
                        current, mask, field_mean, field_std, spacing, scale
                    )
                model_errors.append(float(error))
                persistence_errors.append(float(persistence))
                divergence_rms.append(math.sqrt(max(0.0, float(divergence))))
            first_worse = next(
                (index + 1 for index, pair in enumerate(zip(model_errors, persistence_errors)) if pair[0] > pair[1]),
                None,
            )
            finite = bool(np.all(np.isfinite(model_errors)))
            scenario_result = {
                "model_mse_curve": model_errors,
                "persistence_mse_curve": persistence_errors,
                "normalized_divergence_rms_curve": divergence_rms,
                "model_mean_mse": float(np.mean(model_errors)),
                "model_final_mse": model_errors[-1],
                "persistence_mean_mse": float(np.mean(persistence_errors)),
                "persistence_final_mse": persistence_errors[-1],
                "first_step_worse_than_persistence": first_worse,
                "finite_all_steps": finite,
                "mean_normalized_divergence_rms": float(np.mean(divergence_rms)),
            }
            result["scenarios"][scenario.scenario_id] = scenario_result
            all_model_errors.extend(model_errors)
            all_tail_errors.extend(model_errors[-50:])
            all_divergence.extend(divergence_rms)
    result["aggregate"] = {
        "mean_mse": float(np.mean(all_model_errors)),
        "tail_50_mean_mse": float(np.mean(all_tail_errors)),
        "mean_normalized_divergence_rms": float(np.mean(all_divergence)),
        "finite_all_steps": bool(np.all(np.isfinite(all_model_errors))),
    }
    result["selection_score"] = float(
        math.log1p(result["aggregate"]["mean_mse"])
        + 0.5 * math.log1p(result["aggregate"]["tail_50_mean_mse"])
    )
    return result


def benchmark_latency(
    model: FNO3dStep,
    dataset_root: Path,
    scenario: Scenario,
    mean: np.ndarray,
    std: np.ndarray,
    parameter_mean: np.ndarray,
    parameter_std: np.ndarray,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    files = scenario_files(dataset_root, scenario.scenario_id)
    initial, valid, axes = normalized_state(files[0], mean, std)
    static_np, _, _ = static_channels(axes, valid, scenario, parameter_mean, parameter_std)
    current = torch.from_numpy(np.moveaxis(initial, -1, 0)[None].copy()).to(device)
    mask = torch.from_numpy(valid[None, None].astype(np.float32)).to(device)
    static = torch.from_numpy(static_np[None]).to(device)
    model.eval()
    values: list[float] = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(warmup):
            model_step(model, current, static, mask, current)
        torch.cuda.synchronize()
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model_step(model, current, static, mask, current)
            end.record()
            end.synchronize()
            values.append(float(start.elapsed_time(end)))
    array = np.asarray(values)
    return {
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "mean_ms": float(np.mean(array)),
        "iterations": iterations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-override", type=int)
    parser.add_argument("--initial-checkpoint-override", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.seed_override is not None:
        config["seed"] = args.seed_override
    if args.initial_checkpoint_override is not None:
        config["initial_checkpoint"] = str(args.initial_checkpoint_override)
    effective_config_text = yaml.safe_dump(config, sort_keys=False)
    effective_config_sha256 = hashlib.sha256(effective_config_text.encode("utf-8")).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True

    scenarios = load_scenario_table(Path(config["scenario_table"]))
    train_scenarios = [value for value in scenarios.values() if value.split == config["training_split"]]
    validation_scenarios = [
        value for value in scenarios.values() if value.split == config["validation_split"]
    ]
    protected = [
        value.scenario_id for value in scenarios.values() if value.split in config["protected_splits"]
    ]
    used = {value.scenario_id for value in train_scenarios + validation_scenarios}
    if used & set(protected):
        raise RuntimeError("Protected split leakage detected")

    statistics_path = Path(config["training_statistics"])
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    expected_train = {value.scenario_id for value in train_scenarios}
    if set(statistics["source_scenarios"]) != expected_train:
        raise RuntimeError("Training statistics do not match the training split")
    mean = np.asarray(statistics["mean"], dtype=np.float32)
    std = np.asarray(statistics["std"], dtype=np.float32)
    parameter_mean = np.asarray(statistics["scenario_parameter_mean"], dtype=np.float32)
    parameter_std = np.asarray(statistics["scenario_parameter_std"], dtype=np.float32)
    field_mean = torch.from_numpy(mean.reshape(1, 4, 1, 1, 1)).to(device)
    field_std = torch.from_numpy(std.reshape(1, 4, 1, 1, 1)).to(device)

    model_config = config["model"]
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
    training = config["training"]
    initial_checkpoint = config.get("initial_checkpoint")
    initial_checkpoint_sha256 = None
    if initial_checkpoint:
        initial_checkpoint_path = Path(initial_checkpoint)
        checkpoint = torch.load(initial_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        initial_checkpoint_sha256 = sha256(initial_checkpoint_path)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    schedule = list(training["rollout_schedule"])
    if len(schedule) != int(training["epochs"]):
        raise RuntimeError("Rollout schedule length must equal epoch count")

    dataset_root = Path(config["dataset_root"])
    metrics_path = args.output_dir / "epoch_metrics.jsonl"
    best_score = float("inf")
    best_epoch = 0
    start_time = time.perf_counter()
    for epoch, horizon in enumerate(schedule, start=1):
        maximum_sequences = min(
            int(training["max_train_sequences_per_epoch"]),
            max(1, int(training["max_rollout_frames_per_epoch"]) // int(horizon)),
        )
        dataset = SequenceDataset(
            dataset_root,
            train_scenarios,
            mean,
            std,
            parameter_mean,
            parameter_std,
            int(horizon),
            maximum_sequences,
            seed + epoch,
        )
        loader = DataLoader(
            dataset,
            batch_size=int(training["batch_size"]),
            shuffle=True,
            num_workers=int(training["num_workers"]),
            pin_memory=True,
            persistent_workers=False,
            generator=torch.Generator().manual_seed(seed + epoch),
        )
        epoch_start = time.perf_counter()
        train_metrics = train_epoch(
            model,
            loader,
            optimizer,
            device,
            field_mean,
            field_std,
            float(training["divergence_weight"]),
            float(training.get("divergence_floor_rms", 0.0)),
            float(training["gradient_clip"]),
        )
        validation = validate_rollout(
            model,
            dataset_root,
            validation_scenarios,
            mean,
            std,
            parameter_mean,
            parameter_std,
            device,
            field_mean,
            field_std,
        )
        record = {
            "epoch": epoch,
            "horizon": int(horizon),
            "train": train_metrics,
            "validation": validation,
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps({
            "epoch": epoch,
            "horizon": horizon,
            "train_loss": train_metrics["loss"],
            "validation_mean_mse": validation["aggregate"]["mean_mse"],
            "validation_tail_mse": validation["aggregate"]["tail_50_mean_mse"],
            "selection_score": validation["selection_score"],
        }), flush=True)
        if validation["aggregate"]["finite_all_steps"] and validation["selection_score"] < best_score:
            best_score = float(validation["selection_score"])
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "selection_score": best_score,
                    "config": config,
                },
                args.output_dir / "best_model.pt",
            )

    if best_epoch == 0:
        raise RuntimeError("No finite validation checkpoint was produced")
    checkpoint = torch.load(args.output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    final_validation = validate_rollout(
        model,
        dataset_root,
        validation_scenarios,
        mean,
        std,
        parameter_mean,
        parameter_std,
        device,
        field_mean,
        field_std,
    )
    evaluation = config["evaluation"]
    latency = benchmark_latency(
        model,
        dataset_root,
        validation_scenarios[0],
        mean,
        std,
        parameter_mean,
        parameter_std,
        device,
        int(evaluation["latency_warmup"]),
        int(evaluation["latency_iterations"]),
    )
    summary = {
        "status": "FNO_MULTISTEP_PILOT_OK",
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_seconds": time.perf_counter() - start_time,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "validation": final_validation,
        "latency": latency,
        "latency_target_p99_ms": float(evaluation["target_p99_ms"]),
        "latency_target_pass": latency["p99_ms"] <= float(evaluation["target_p99_ms"]),
        "train_scenarios_read": [value.scenario_id for value in train_scenarios],
        "validation_scenarios_read": [value.scenario_id for value in validation_scenarios],
        "protected_scenarios_not_read": protected,
        "statistics_sha256": sha256(statistics_path),
        "source_config_sha256": sha256(args.config),
        "effective_config_sha256": effective_config_sha256,
        "seed_override": args.seed_override,
        "initial_checkpoint_override": (
            str(args.initial_checkpoint_override)
            if args.initial_checkpoint_override is not None
            else None
        ),
        "initial_checkpoint": initial_checkpoint,
        "initial_checkpoint_sha256": initial_checkpoint_sha256,
        "divergence_floor_rms": float(training.get("divergence_floor_rms", 0.0)),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "config.yaml").write_text(effective_config_text, encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
