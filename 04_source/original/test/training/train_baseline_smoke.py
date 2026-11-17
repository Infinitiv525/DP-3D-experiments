"""End-to-end smoke training for one-step 3D flow prediction.

This is a pipeline baseline, not the final PINO used in the dissertation study.
It deliberately reads only train and validation scenarios.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset


FIELD_NAMES = ("u", "v", "w", "p")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_scenarios(table: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    with table.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            result.setdefault(row["split"], []).append(row["scenario_id"])
    return result


def load_snapshot(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    with np.load(path, allow_pickle=False) as data:
        fields = np.concatenate((data["U"], data["p"][..., None]), axis=-1)
        valid = data["valid"].astype(bool)
        coordinates = (data["x"].copy(), data["y"].copy(), data["z"].copy())
    return fields.astype(np.float32), valid, coordinates


def scenario_files(dataset_root: Path, scenario: str) -> list[Path]:
    index_path = dataset_root / scenario / "series_index.csv"
    with index_path.open(newline="", encoding="utf-8") as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: float(row["time_s"]))
    files = [dataset_root / scenario / row["filename"] for row in rows]
    if len(files) != 201 or any(not path.is_file() for path in files):
        raise RuntimeError(f"Incomplete scenario {scenario}")
    return files


def calculate_training_statistics(dataset_root: Path, scenarios: list[str]) -> dict[str, list[float]]:
    sums = np.zeros(4, dtype=np.float64)
    square_sums = np.zeros(4, dtype=np.float64)
    count = 0
    for scenario in scenarios:
        for path in scenario_files(dataset_root, scenario):
            fields, valid, _ = load_snapshot(path)
            values = fields[valid]
            sums += values.sum(axis=0, dtype=np.float64)
            square_sums += np.square(values, dtype=np.float64).sum(axis=0)
            count += values.shape[0]
    mean = sums / count
    variance = np.maximum(square_sums / count - mean**2, 1e-12)
    return {
        "field_names": list(FIELD_NAMES),
        "mean": mean.tolist(),
        "std": np.sqrt(variance).tolist(),
        "valid_value_count_per_channel": count,
        "source_scenarios": scenarios,
    }


@dataclass(frozen=True)
class Pair:
    scenario: str
    current: Path
    target: Path


class FlowPairDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        scenarios: list[str],
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        self.mean = mean.reshape(1, 1, 1, 4).astype(np.float32)
        self.std = std.reshape(1, 1, 1, 4).astype(np.float32)
        self.pairs: list[Pair] = []
        for scenario in scenarios:
            files = scenario_files(dataset_root, scenario)
            self.pairs.extend(
                Pair(scenario, files[index], files[index + 1])
                for index in range(len(files) - 1)
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        pair = self.pairs[index]
        current, valid, coordinates = load_snapshot(pair.current)
        target, target_valid, _ = load_snapshot(pair.target)
        if not np.array_equal(valid, target_valid):
            raise RuntimeError(f"Mask changed within scenario {pair.scenario}")
        current = (current - self.mean) / self.std
        target = (target - self.mean) / self.std
        coordinate_channels = np.meshgrid(
            *[
                2.0 * (axis - axis.min()) / (axis.max() - axis.min()) - 1.0
                for axis in coordinates
            ],
            indexing="ij",
        )
        features = np.concatenate(
            (
                current,
                valid[..., None].astype(np.float32),
                np.stack(coordinate_channels, axis=-1).astype(np.float32),
            ),
            axis=-1,
        )
        return (
            torch.from_numpy(np.moveaxis(features, -1, 0).copy()),
            torch.from_numpy(np.moveaxis(current, -1, 0).copy()),
            torch.from_numpy(np.moveaxis(target, -1, 0).copy()),
            torch.from_numpy(valid[None, ...].astype(np.float32)),
        )


class ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(width, width, kernel_size=3, padding=1),
        )
        self.activation = nn.GELU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(value + self.layers(value))


class OneStepCNN(nn.Module):
    def __init__(self, width: int, residual_blocks: int) -> None:
        super().__init__()
        self.stem = nn.Conv3d(8, width, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(residual_blocks)])
        self.head = nn.Conv3d(width, 4, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, features: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        return current + self.head(self.blocks(torch.nn.functional.gelu(self.stem(features))))


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (((prediction - target) ** 2) * mask).sum() / (mask.sum() * prediction.shape[1])


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, use_bf16: bool
) -> tuple[float, float]:
    model.eval()
    model_loss = 0.0
    persistence_loss = 0.0
    samples = 0
    with torch.inference_mode():
        for features, current, target, mask in loader:
            features, current, target, mask = [item.to(device, non_blocking=True) for item in (features, current, target, mask)]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                prediction = model(features, current)
                loss = masked_mse(prediction, target, mask)
                baseline = masked_mse(current, target, mask)
            batch = features.shape[0]
            model_loss += float(loss) * batch
            persistence_loss += float(baseline) * batch
            samples += batch
    return model_loss / samples, persistence_loss / samples


def latency(model: nn.Module, sample: tuple[torch.Tensor, ...], device: torch.device) -> dict[str, float]:
    features, current = sample[0][:1].to(device), sample[1][:1].to(device)
    model.eval()
    values: list[float] = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(50):
            model(features, current)
        torch.cuda.synchronize()
        for _ in range(300):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(features, current)
            end.record()
            end.synchronize()
            values.append(float(start.elapsed_time(end)))
    array = np.asarray(values)
    return {
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "mean_ms": float(np.mean(array)),
        "iterations": len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=False)

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")

    dataset_root = Path(config["dataset_root"])
    splits = load_scenarios(Path(config["scenario_table"]))
    train_scenarios = splits[config["training_split"]]
    validation_scenarios = splits[config["validation_split"]]
    protected = [scenario for split in config["protected_splits"] for scenario in splits[split]]
    if set(train_scenarios) & set(validation_scenarios + protected):
        raise RuntimeError("Scenario split leakage detected")

    stats = calculate_training_statistics(dataset_root, train_scenarios)
    (args.output_dir / "training_statistics.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    train_dataset = FlowPairDataset(dataset_root, train_scenarios, mean, std)
    validation_dataset = FlowPairDataset(dataset_root, validation_scenarios, mean, std)
    generator = torch.Generator().manual_seed(seed)
    loader_options = dict(
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        pin_memory=True,
        persistent_workers=int(config["num_workers"]) > 0,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)

    model = OneStepCNN(int(config["model_width"]), int(config["residual_blocks"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    use_bf16 = config["mixed_precision"] == "bf16"
    metrics: list[dict[str, float | int]] = []
    best_validation = float("inf")
    start_training = time.perf_counter()
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        total = 0.0
        samples = 0
        epoch_start = time.perf_counter()
        for features, current, target, mask in train_loader:
            features, current, target, mask = [item.to(device, non_blocking=True) for item in (features, current, target, mask)]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                prediction = model(features, current)
                loss = masked_mse(prediction, target, mask)
            loss.backward()
            optimizer.step()
            batch = features.shape[0]
            total += float(loss.detach()) * batch
            samples += batch
        validation_loss, persistence_loss = evaluate(model, validation_loader, device, use_bf16)
        record = {
            "epoch": epoch,
            "train_mse_normalized": total / samples,
            "validation_mse_normalized": validation_loss,
            "persistence_mse_normalized": persistence_loss,
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        metrics.append(record)
        print(json.dumps(record), flush=True)
        if validation_loss < best_validation:
            best_validation = validation_loss
            torch.save(model.state_dict(), args.output_dir / "best_model.pt")

    model.load_state_dict(torch.load(args.output_dir / "best_model.pt", map_location=device, weights_only=True))
    first_validation_batch = next(iter(validation_loader))
    result = {
        "status": "BASELINE_SMOKE_TRAINING_OK",
        "config_sha256": file_sha256(args.config),
        "seed": seed,
        "device": torch.cuda.get_device_name(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train_scenarios": train_scenarios,
        "validation_scenarios": validation_scenarios,
        "protected_scenarios_not_read": protected,
        "train_pairs": len(train_dataset),
        "validation_pairs": len(validation_dataset),
        "best_validation_mse_normalized": best_validation,
        "persistence_validation_mse_normalized": metrics[-1]["persistence_mse_normalized"],
        "improvement_over_persistence_percent": 100.0 * (1.0 - best_validation / float(metrics[-1]["persistence_mse_normalized"])),
        "training_seconds": time.perf_counter() - start_training,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
        "latency_batch1_bf16": latency(model, first_validation_batch, device),
        "metrics": metrics,
        "scope_note": "Pipeline smoke test; not the final PINO or dissertation comparison.",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "config.yaml").write_text(
        args.config.read_text(encoding="utf-8"), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
