"""Attention-level 3-way (corrected): naive softmax vs flash (4-D) vs CR.

The earlier bench_vs_flash.py used 3-D single-head q,k,v, which silently fell
back to the O(N^2) math kernel; that 'flash' column was really naive softmax.
Fused flash/mem-efficient kernels require a 4-D (B,H,N,D) layout.  This probe
measures the honest three-way at the attention operator only.

  naive : fp32, explicit QK^T  -> O(N^2) compute AND memory
  flash : bf16, 4-D SDPA       -> O(N) memory, O(N^2) compute, half precision
  CR    : fp32 complex64       -> O(N log N) compute, O(N) memory, full precision
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from crnn.layers import PiecewiseCRAttention


def main():
    dev = torch.device("cuda")
    B, d, H, D = 8, 128, 8, 16
    print(f"attention fwd+bwd  (B={B} d={d} H={H} D={D})")
    print(f"{'p':>4} {'N':>6} | {'naive-fp32':>12} | {'flash-bf16':>12} | "
          f"{'CR-c64':>12} | {'CR-half':>12}")
    print("      " + "      |  ms     GB    |  ms     GB    |  ms     GB    |  ms     GB")
    for p in (11, 13, 17, 19, 23):
        N = p ** 3
        row = f"{p:>4} {N:>6} |"

        # naive softmax, fp32 (3-D single-head materializes B x N x N)
        q = torch.randn(B, N, d, device=dev, dtype=torch.float32,
                        requires_grad=True)
        k = torch.randn(B, N, d, device=dev, dtype=torch.float32,
                        requires_grad=True)
        v = torch.randn(B, N, d, device=dev, dtype=torch.float32,
                        requires_grad=True)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        for _ in range(2):
            out = (q @ k.transpose(-2, -1) / (d ** 0.5)).softmax(-1) @ v
            out.sum().backward()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(5):
            out = (q @ k.transpose(-2, -1) / (d ** 0.5)).softmax(-1) @ v
            out.sum().backward()
        torch.cuda.synchronize()
        t_naive = (time.perf_counter() - t0) / 5 * 1000
        m_naive = torch.cuda.max_memory_allocated() / 1e9

        # flash, bf16, 4-D multi-head (triggers the fused kernel)
        q4 = torch.randn(B, H, N, D, device=dev, dtype=torch.bfloat16,
                         requires_grad=True)
        k4 = torch.randn(B, H, N, D, device=dev, dtype=torch.bfloat16,
                         requires_grad=True)
        v4 = torch.randn(B, H, N, D, device=dev, dtype=torch.bfloat16,
                         requires_grad=True)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        for _ in range(2):
            out = F.scaled_dot_product_attention(q4, k4, v4)
            out.sum().backward()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(5):
            out = F.scaled_dot_product_attention(q4, k4, v4)
            out.sum().backward()
        torch.cuda.synchronize()
        t_flash = (time.perf_counter() - t0) / 5 * 1000
        m_flash = torch.cuda.max_memory_allocated() / 1e9

        # CR, fp32 complex64 (full precision)
        z = torch.randn(B, N, d, dtype=torch.complex64, device=dev)
        a = PiecewiseCRAttention(d, p=p, n_flow=1, gate=False, mix=False).to(dev)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        for _ in range(2):
            out = a(z); out.real.sum().backward(); a.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(5):
            out = a(z); out.real.sum().backward(); a.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t_cr = (time.perf_counter() - t0) / 5 * 1000
        m_cr = torch.cuda.max_memory_allocated() / 1e9

        # CR, complex32 storage + fp32 FFT (half=True): CR's own half precision
        zh = z.to(torch.complex32)
        ah = PiecewiseCRAttention(d, p=p, n_flow=1, gate=False, mix=False,
                                  half=True).to(dev)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        for _ in range(2):
            out = ah(zh); out.real.sum().backward(); ah.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(5):
            out = ah(zh); out.real.sum().backward(); ah.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t_crh = (time.perf_counter() - t0) / 5 * 1000
        m_crh = torch.cuda.max_memory_allocated() / 1e9

        print(f"{p:>4} {N:>6} | {t_naive:>5.1f} {m_naive:>7.3f} | "
              f"{t_flash:>5.1f} {m_flash:>7.3f} | {t_cr:>5.1f} {m_cr:>7.3f} | "
              f"{t_crh:>5.1f} {m_crh:>7.3f}", flush=True)


if __name__ == "__main__":
    main()
