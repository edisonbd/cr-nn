"""Unified-precision (fp32) comparison: CR vs softmax, everything fp32.

The user's fair point: bf16-flash vs fp32-CR is apples-to-oranges.  Unify at
fp32.  At fp32, F.scaled_dot_product_attention has NO fused kernel and falls
back to the O(N^2) math backend, i.e. "flash at fp32" == "naive softmax".  So
the only fp32 softmax is O(N^2), and the fair question is: CR (fp32) vs
softmax (fp32), both precision-matched.

Attention-level AND full-model, p = 11..23.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from crnn.layers import PiecewiseCRAttention


def bench(fn, dev, repeat=5):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return ((time.perf_counter() - t0) / repeat * 1000,
            torch.cuda.max_memory_allocated() / 1e9)


def main():
    dev = torch.device("cuda")
    B, d = 8, 128
    print(f"UNIFIED fp32  (B={B} d={d})")
    print(f"{'p':>4} {'N':>6} | {'softmax-fp32':>14} | {'CR-fp32':>14} | "
          f"{'speedup':>8} | {'mem_ratio':>9}")
    print("      " + "      |  ms     GB     |  ms     GB     |  CR faster |  CR/mem")
    for p in (11, 13, 17, 19, 23):
        N = p ** 3

        # softmax at fp32: naive matmul (== SDPA at fp32, which falls to math)
        q = torch.randn(B, N, d, device=dev, dtype=torch.float32,
                        requires_grad=True)
        k = torch.randn(B, N, d, device=dev, dtype=torch.float32,
                        requires_grad=True)
        v = torch.randn(B, N, d, device=dev, dtype=torch.float32,
                        requires_grad=True)

        def sm():
            out = (q @ k.transpose(-2, -1) / (d ** 0.5)).softmax(-1) @ v
            out.sum().backward()
        t_sm, m_sm = bench(sm, dev)

        # CR at fp32 (complex64 = fp32 real + fp32 imag)
        z = torch.randn(B, N, d, dtype=torch.complex64, device=dev)
        a = PiecewiseCRAttention(d, p=p, n_flow=1, gate=False, mix=False).to(dev)

        def cr():
            out = a(z)
            out.real.sum().backward()
            a.zero_grad(set_to_none=True)
        t_cr, m_cr = bench(cr, dev)

        print(f"{p:>4} {N:>6} | {t_sm:>5.1f} {m_sm:>7.3f} | {t_cr:>5.1f} "
              f"{m_cr:>7.3f} | {t_sm/t_cr:>7.1f}x | {m_cr/m_sm:>8.2f}x",
              flush=True)


if __name__ == "__main__":
    main()
