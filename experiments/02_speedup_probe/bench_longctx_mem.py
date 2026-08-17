"""Long-sequence memory: CR vs flash vs naive, at a FIXED memory budget.

Answers: for a fixed memory budget, how many tokens can each attention hold?
Extrapolation uses the correct scaling: O(N^2) for naive, O(N) for flash/CR.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from crnn.layers import PiecewiseCRAttention


def mem_naive(B, N, d, dev):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    q = torch.randn(B, N, d, device=dev, dtype=torch.float32, requires_grad=True)
    k = torch.randn(B, N, d, device=dev, dtype=torch.float32, requires_grad=True)
    v = torch.randn(B, N, d, device=dev, dtype=torch.float32, requires_grad=True)
    try:
        out = (q @ k.transpose(-2, -1) / (d ** 0.5)).softmax(-1) @ v
        out.sum().backward()
    except RuntimeError:
        return None
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e9


def mem_flash(B, N, d, H, dev):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    q = torch.randn(B, H, N, d // H, device=dev, dtype=torch.bfloat16,
                    requires_grad=True)
    k = torch.randn(B, H, N, d // H, device=dev, dtype=torch.bfloat16,
                    requires_grad=True)
    v = torch.randn(B, H, N, d // H, device=dev, dtype=torch.bfloat16,
                    requires_grad=True)
    out = F.scaled_dot_product_attention(q, k, v)
    out.sum().backward()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e9


def mem_cr(B, N, d, p, dev):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    z = torch.randn(B, N, d, dtype=torch.complex64, device=dev)
    a = PiecewiseCRAttention(d, p=p, n_flow=1, gate=False, mix=False).to(dev)
    out = a(z)
    out.real.sum().backward()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e9


def main():
    dev = torch.device("cuda")
    B, d, H = 8, 128, 8
    budget = 40.0
    print(f"attention fwd+bwd memory at long N  (B={B} d={d}, budget={budget}GB)")
    print(f"{'N':>7} | {'naive-fp32':>14} | {'flash-bf16':>14} | {'CR-fp32':>14}")
    print("       " + " |   GB   max@8GB |   GB  max@8GB |   GB  max@8GB")
    for p in (23, 29, 31, 37):
        N = p ** 3
        mn = mem_naive(B, N, d, dev)
        mf = mem_flash(B, N, d, H, dev)
        mc = mem_cr(B, N, d, p, dev)

        def show_quad(m):   # naive: O(N^2)
            if m is None:
                return "OOM     --"
            nk = N * (budget / m) ** 0.5
            return f"{m:>5.2f} {nk/1e3:>6.0f}K"

        def show_lin(m):    # flash/CR: O(N)
            if m is None:
                return "OOM     --"
            nk = N * (budget / m)
            return f"{m:>5.2f} {nk/1e3:>6.0f}K"

        print(f"{N:>7} | {show_quad(mn):>14} | {show_lin(mf):>14} | "
              f"{show_lin(mc):>14}", flush=True)


if __name__ == "__main__":
    main()
