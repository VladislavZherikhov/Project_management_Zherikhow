"""
Task 1 - quick CPU benchmark of candidate architectures (no training).

Weights are not loaded: latency, size and parameter count do not depend on
the weight values, so random initialisation gives the same cost numbers.
The classifier head is replaced with a single output (binary good/bad).

Latency protocol: batch size 1, 224x224, 10 warm-up runs, then the mean of
100 timed runs, torch.inference_mode(), fp32.

Usage:
    python benchmark_candidates.py
"""

import io
import json
import platform
import statistics
import time
from pathlib import Path

import torch
from torch import nn
from torchvision import models

WARMUP_RUNS = 10
TIMED_RUNS = 100
INPUT_SIZE = 224


def binary_head(model, name):
    """Replace the ImageNet head with a 1-unit output, like the real task."""
    if name == "resnet50":
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif name.startswith("yolo"):
        # Ultralytics uses a softmax over classes: 2 outputs for good/bad
        head = model.model[-1]
        head.linear = nn.Linear(head.linear.in_features, 2)
    else:
        last = model.classifier[-1]
        model.classifier[-1] = nn.Linear(last.in_features, 1)
    return model


def yolo_cls(cfg):
    """YOLOv8 classification model built from its yaml (no weight download)."""
    from ultralytics import YOLO
    return YOLO(cfg).model


CANDIDATES = {
    "mobilenet_v2":    lambda: models.mobilenet_v2(weights=None),
    "efficientnet_b0": lambda: models.efficientnet_b0(weights=None),
    "resnet50":        lambda: models.resnet50(weights=None),
    "yolov8n_cls":     lambda: yolo_cls("yolov8n-cls.yaml"),
    "yolov8s_cls":     lambda: yolo_cls("yolov8s-cls.yaml"),
}


def size_mb(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell() / 1024 ** 2


def latency_ms(model, runs, warmup):
    x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - start) * 1000)
    return statistics.mean(times), statistics.stdev(times)


def main():
    torch.manual_seed(0)
    results = {
        "device": platform.processor() or "cpu",
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "protocol": f"batch 1, {INPUT_SIZE}x{INPUT_SIZE}, {WARMUP_RUNS} warm-up, mean of {TIMED_RUNS} runs",
        "models": {},
    }
    for name, build in CANDIDATES.items():
        model = binary_head(build(), name).eval()
        params = sum(p.numel() for p in model.parameters())
        mean, std = latency_ms(model, TIMED_RUNS, WARMUP_RUNS)
        results["models"][name] = {
            "params_m": round(params / 1e6, 2),
            "size_mb_fp32": round(size_mb(model), 1),
            "cpu_latency_ms_mean": round(mean, 1),
            "cpu_latency_ms_std": round(std, 1),
        }
        print(f"{name:<16} params {params / 1e6:6.2f} M  size {size_mb(model):6.1f} MB  "
              f"latency {mean:6.1f} ± {std:4.1f} ms")

    out = Path(__file__).with_name("benchmark_results.json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
