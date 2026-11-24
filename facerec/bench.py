"""Model size and latency (ms / image at batch 1)."""
from __future__ import annotations

import copy
import time

import torch


def n_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def latency_ms(model, device="cuda", n=100, warmup=20, threads=None):
    m = copy.deepcopy(model).to(device).eval()
    if device == "cpu" and threads:
        torch.set_num_threads(threads)
    x = torch.randn(1, 3, 112, 112, device=device)
    for _ in range(warmup):
        m(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        m(x)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000
