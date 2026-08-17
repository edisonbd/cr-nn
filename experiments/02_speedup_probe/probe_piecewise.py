"""Piecewise-CR speed + memory probe (forward+backward, the training cost).

Compares four attention operators at the same (B, N, d), N = p^3 (prime p):

    softmax   : multi-head scaled dot-product attention   O(N^2 d) time/mem
    szego     : matrix-valued H_p group convolution        O(p^4) matmuls
    fluid     : diagonal spectral flow (single stage)      O(N log N)
    piecewise : activation-segmented spectral flow         O(K N log N)

The headline the user cares about is *training* memory and *speed*: we report
forward+backward wall time and peak allocated VRAM for each operator.  The
piecewise layer must come in far below softmax (no N^2 attention matrix) and
below szego (no p x p batched matmuls / einsum intermediates), while staying
O(K p^3 log p).

Run on the A800:
    /root/miniconda3/bin/python experiments/02_speedup_probe/probe_piecewise.py
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from crnn.layers import FluidCRAttention, PiecewiseCRAttention, VecCRAttention


class SoftmaxAttn(nn.Module):
    """Multi-head scaled dot-product attention with learnable projections."""

    def __init__(self, d: int, nhead: int = 4):
        super().__init__()
        self.d = d
        self.nhead = nhead
        self.hd = d // nhead
        self.wq = nn.Linear(d, d)
        self.wk = nn.Linear(d, d)
        self.wv = nn.Linear(d, d)
        self.wo = nn.Linear(d, d)

    def forward(self, x):
        B, N, d = x.shape
        q = self.wq(x).view(B, N, self.nhead, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, N, self.nhead, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, N, self.nhead, self.hd).transpose(1, 2)
        a = torch.softmax(q @ k.transpose(-1, -2) / (self.hd ** 0.5), dim=-1)
        o = (a @ v).transpose(1, 2).contiguous().view(B, N, d)
        return self.wo(o)


def _bench(module, z, is_complex, device, repeat=5):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(2):                       # warmup
        out = module(z)
        loss = out.real.sum() if is_complex else out.sum()
        loss.backward()
        module.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        out = module(z)
        loss = out.real.sum() if is_complex else out.sum()
        loss.backward()
        module.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) / repeat * 1000.0
    mem_gb = torch.cuda.max_memory_allocated() / 1e9
    return dt_ms, mem_gb


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"torch: {torch.__version__}")
    d, B = 64, 2
    primes = [5, 7, 11, 13, 17, 19, 23]
    header = (f"{'p':>4} {'N=p^3':>8} | "
              f"{'softmax':>8} {'szego':>8} {'fluid':>8} {'piecewise':>8} | "
              f"{'softmax':>8} {'szego':>8} {'fluid':>8} {'piecewise':>8}")
    print(header)
    print("  " + " " * 12 + "|---- time fwd+bwd (ms) ----|"
          "---- peak VRAM (GB) ----")
    print("-" * len(header))

    results = []
    for p in primes:
        N = p ** 3
        row = []
        # softmax (real) — skip if the N^2 matrix is too large
        try:
            sm = SoftmaxAttn(d).to(device)
            xr = torch.randn(B, N, d, device=device)
            t, m = _bench(sm, xr, False, device)
            row.extend([t, m])
        except RuntimeError as e:
            row.extend([float("nan"), float("nan")])
            print(f"  softmax OOM at p={p}: {str(e)[:60]}", flush=True)

        # szego (matrix-valued group conv)
        try:
            sz = VecCRAttention(d, p=p, mix=True, gate=False).to(device)
            zc = torch.randn(B, N, d, dtype=torch.complex64, device=device)
            t, m = _bench(sz, zc, True, device)
            row.extend([t, m])
        except RuntimeError as e:
            row.extend([float("nan"), float("nan")])
            print(f"  szego OOM at p={p}: {str(e)[:60]}", flush=True)

        # fluid (single diagonal stage)
        fl = FluidCRAttention(d, p=p, mix=True, gate=False, spectrum="full").to(device)
        t, m = _bench(fl, zc, True, device)
        row.extend([t, m])

        # piecewise (K=3 segmented stages)
        pw = PiecewiseCRAttention(d, p=p, mix=True, gate=False,
                                  n_flow=3, spectrum="full", nl="modrelu").to(device)
        t, m = _bench(pw, zc, True, device)
        row.extend([t, m])

        print(f"{p:>4} {N:>8} | "
              f"{row[0]:>8.1f} {row[2]:>8.1f} {row[4]:>8.1f} {row[6]:>8.1f} | "
              f"{row[1]:>8.3f} {row[3]:>8.3f} {row[5]:>8.3f} {row[7]:>8.3f}")
        results.append((p, N, row))

    print()
    print("speedups (vs softmax) at the largest N measured:")
    if results:
        p, N, row = results[-1]
        print(f"  szego    {row[0]/row[2]:.2f}x  (time),  mem {row[1]/row[3]:.2f}x")
        print(f"  fluid    {row[0]/row[4]:.2f}x  (time),  mem {row[1]/row[5]:.2f}x")
        print(f"  piecewise {row[0]/row[6]:.2f}x (time),  mem {row[1]/row[7]:.2f}x")


if __name__ == "__main__":
    main()
