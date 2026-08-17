"""Speed at long N: CR O(N log N) vs flash O(N^2) compute.

Memory is O(N) for both flash (bf16) and CR (fp32); the durable CR win over
flash is *compute* — O(N log N) vs O(N^2).  Measure fwd+bwd time at long N.
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from crnn.layers import PiecewiseCRAttention


def time_flash(B, N, d, H, dev, repeat=5):
    q = torch.randn(B, H, N, d // H, device=dev, dtype=torch.bfloat16,
                    requires_grad=True)
    k = torch.randn(B, H, N, d // H, device=dev, dtype=torch.bfloat16,
                    requires_grad=True)
    v = torch.randn(B, H, N, d // H, device=dev, dtype=torch.bfloat16,
                    requires_grad=True)
    for _ in range(2):
        F.scaled_dot_product_attention(q, k, v).sum().backward()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        F.scaled_dot_product_attention(q, k, v).sum().backward()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeat * 1000


def time_cr(B, N, d, p, dev, repeat=5):
    z = torch.randn(B, N, d, dtype=torch.complex64, device=dev)
    a = PiecewiseCRAttention(d, p=p, n_flow=1, gate=False, mix=False).to(dev)
    for _ in range(2):
        a(z).real.sum().backward()
        a.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        a(z).real.sum().backward()
        a.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeat * 1000


def main():
    dev = torch.device("cuda")
    B, d, H = 8, 128, 8
    print(f"attention fwd+bwd time at long N  (B={B} d={d})")
    print(f"{'N':>7} | {'flash-bf16':>11} | {'CR-fp32':>11} | {'CR speedup':>10}")
    for p in (23, 29, 31, 37):
        N = p ** 3
        tf = time_flash(B, N, d, H, dev)
        tc = time_cr(B, N, d, p, dev)
        print(f"{N:>7} | {tf:>7.1f} ms | {tc:>7.1f} ms | {tf/tc:>7.1f}x",
              flush=True)


if __name__ == "__main__":
    main()
