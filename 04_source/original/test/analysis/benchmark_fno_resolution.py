"""Benchmark the candidate FNO step at two grid resolutions on one GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from fno_model import FNO3dStep


def benchmark(
    config: dict,
    dimensions: tuple[int, int, int],
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    device = torch.device("cuda:0")
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
    nx, ny, nz = dimensions
    features = torch.randn(
        1, int(model_config["input_channels"]), nx, ny, nz, device=device
    )
    current = torch.randn(1, 4, nx, ny, nz, device=device)
    mask = torch.ones(1, 1, nx, ny, nz, device=device)
    inlet = current.clone()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        prediction = model(features, current, mask, inlet)
        loss = torch.mean(prediction**2)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    if not torch.isfinite(prediction).all():
        raise RuntimeError("Non-finite FNO output")

    model.eval()
    timings: list[float] = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(warmup):
            model(features, current, mask, inlet)
        torch.cuda.synchronize()
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(features, current, mask, inlet)
            end.record()
            end.synchronize()
            timings.append(float(start.elapsed_time(end)))
    values = np.asarray(timings, dtype=np.float64)
    return {
        "dimensions": list(dimensions),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_probe_loss": float(loss.detach()),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "iterations": iterations,
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "mean_ms": float(np.mean(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    torch.manual_seed(int(config["seed"]))
    results = [
        benchmark(config, (64, 32, 16), warmup=50, iterations=300),
        benchmark(config, (128, 64, 32), warmup=50, iterations=300),
    ]
    target = float(config["evaluation"]["target_p99_ms"])
    summary = {
        "status": "FNO_RESOLUTION_BENCHMARK_OK",
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "target_p99_ms": target,
        "results": results,
        "passes_target": {
            "64x32x16": bool(results[0]["p99_ms"] < target),
            "128x64x32": bool(results[1]["p99_ms"] < target),
        },
        "scope_note": "Synthetic architecture benchmark, not trained-model accuracy.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
