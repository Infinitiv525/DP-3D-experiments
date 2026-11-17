"""Verify the training environment on a GPU with a small 3D workload."""

from __future__ import annotations

import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


def percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "mean_ms": float(statistics.fmean(values)),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA")

    torch.manual_seed(20261014)
    torch.cuda.manual_seed_all(20261014)
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    model = nn.Sequential(
        nn.Conv3d(16, 32, kernel_size=3, padding=1),
        nn.GELU(),
        nn.Conv3d(32, 32, kernel_size=3, padding=1),
        nn.GELU(),
        nn.Conv3d(32, 4, kernel_size=1),
    ).to(device)
    sample = torch.randn(1, 16, 64, 32, 16, device=device)
    target = torch.randn(1, 4, 64, 32, 16, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    prediction = model(sample)
    loss = torch.mean((prediction - target) ** 2)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite training loss")

    model.eval()
    with torch.inference_mode():
        for _ in range(30):
            model(sample)
        torch.cuda.synchronize()
        timings: list[float] = []
        for _ in range(200):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = model(sample)
            end.record()
            end.synchronize()
            timings.append(float(start.elapsed_time(end)))
        if output.shape != target.shape or not torch.isfinite(output).all():
            raise RuntimeError("Invalid inference output")

    result = {
        "status": "GPU_TORCH_PROBE_OK",
        "timestamp_unix": time.time(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "gpu_name": properties.name,
        "gpu_compute_capability": list(torch.cuda.get_device_capability(device)),
        "gpu_total_memory_bytes": properties.total_memory,
        "input_shape": list(sample.shape),
        "output_shape": list(output.shape),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_step_loss": float(loss.detach().cpu()),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "inference_iterations": len(timings),
        "inference_note": "Environment probe only; not final model latency.",
        **percentiles(timings),
    }
    output_path = Path("gpu_torch_probe.json")
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
