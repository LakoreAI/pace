"""Parameter, MAC and latency measurement for the efficiency tables.

MACs are counted with forward hooks for the layers that dominate audio CNNs
(Conv1d, Conv2d, Linear). This is a static multiply-accumulate count for one
forward pass; it does not include BatchNorm/activation arithmetic. Latency is
wall-clock and hardware-dependent: report it together with the device.
"""

import time
from typing import Dict

import torch


def count_macs(model: torch.nn.Module, sample: torch.Tensor) -> int:
    """Multiply-accumulates for a single forward pass of `sample` (no grad)."""
    total = 0
    handles = []

    def conv_hook(module, inputs, output):
        nonlocal total
        spatial = 1
        for k in module.kernel_size:
            spatial *= int(k)
        total += (
            int(output.numel())
            * (int(module.in_channels) // int(module.groups))
            * spatial
        )

    def linear_hook(module, inputs, output):
        nonlocal total
        total += int(output.numel()) * int(module.in_features)

    for module in model.modules():
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d)):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(sample)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return total


@torch.no_grad()
def measure_latency(
    model: torch.nn.Module,
    sample: torch.Tensor,
    warmup: int = 5,
    repeats: int = 20,
) -> Dict[str, float]:
    """Median forward latency in milliseconds (syncs CUDA before each timing)."""
    device = sample.device
    was_training = model.training
    model.eval()
    try:
        for _ in range(warmup):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000.0)
    finally:
        model.train(was_training)
    times.sort()
    return {
        "median_ms": times[len(times) // 2],
        "mean_ms": sum(times) / len(times),
        "min_ms": times[0],
        "batch_size": int(sample.shape[0]),
    }
