"""CR attention vs flash attention (bf16 SDPA) at the attention level.

DeepSeek-class models use bf16 flash attention (O(N) memory via tiling, O(N^2)
compute).  The CR attention is O(N log N) compute with no N^2 anywhere.  This
probe measures fwd+bwd memory/time of

    flash-bf16 : F.scaled_dot_product_attention in bf16 (flash backend)
    cr-c64     : PiecewiseCRAttention n_flow=1 gate=False (complex64)
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from crnn.layers import PiecewiseCRAttention


def main():
    dev = torch.device("cuda")
    B, d = 8, 64
    print(f"{'p':>4} {'N':>6} | {'flash-bf16':>18} | {'cr-c64':>18}")
    print("      " + "      |  time(ms)  mem(GB)  |  time(ms)  mem(GB)")
    for p in (11, 13, 17, 19, 23):
        N = p ** 3
        # flash attention, bf16 (triggers the flash backend)
        q = torch.randn(B, N, d, device=dev, dtype=torch.bfloat16,
                        requires_grad=True)
        k = torch.randn(B, N, d, device=dev, dtype=torch.bfloat16,
                        requires_grad=True)
        v = torch.randn(B, N, d, device=dev, dtype=torch.bfloat16,
                        requires_grad=True)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        for _ in range(2):
            out = F.scaled_dot_product_attention(q, k, v); out.sum().backward()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            out = F.scaled_dot_product_attention(q, k, v); out.sum().backward()
        torch.cuda.synchronize()
        t0 = (time.perf_counter() - t0) / 5 * 1000
        m0 = torch.cuda.max_memory_allocated() / 1e9

        # CR attention, complex64
        z = torch.randn(B, N, d, dtype=torch.complex64, device=dev)
        a = PiecewiseCRAttention(d, p=p, n_flow=1, gate=False).to(dev)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        for _ in range(2):
            out = a(z); out.real.sum().backward(); a.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        for _ in range(5):
            out = a(z); out.real.sum().backward(); a.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t1 = (time.perf_counter() - t1) / 5 * 1000
        m1 = torch.cuda.max_memory_allocated() / 1e9
        print(f"{p:>4} {N:>6} | {t0:>7.1f}  {m0:>6.3f}  | {t1:>8.1f}  {m1:>6.3f}")


if __name__ == "__main__":
    main()
