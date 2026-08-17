"""Memory + speed of the CR attention at TRAINING sizes vs softmax.

The asymptotic O(N) vs O(N^2) gap only appears at N>10K; at p=7/11/13 the
constant factor decides.  This probe isolates the constant: it compares, for
p = 7..19, forward+backward peak memory and wall time of

    softmax  : multi-head scaled dot-product attention
    piecewise n_flow=3 gate=True  (current)
    piecewise n_flow=1 gate=False (optimised: single FFT round, no dbar gate)

at batch=16, d=64, so we see exactly where the CR constant comes from.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from crnn.layers import PiecewiseCRAttention, FluidCRAttention


class SoftmaxAttn(nn.Module):
    def __init__(self, d, nhead=4):
        super().__init__()
        self.d, self.nhead, self.hd = d, nhead, d // nhead
        self.wq = nn.Linear(d, d); self.wk = nn.Linear(d, d)
        self.wv = nn.Linear(d, d); self.wo = nn.Linear(d, d)

    def forward(self, x):
        B, N, d = x.shape
        q = self.wq(x).view(B, N, self.nhead, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, N, self.nhead, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, N, self.nhead, self.hd).transpose(1, 2)
        a = torch.softmax(q @ k.transpose(-1, -2) / (self.hd ** 0.5), -1)
        return self.wo((a @ v).transpose(1, 2).contiguous().view(B, N, d))


def bench(mod, z, is_cx, dev, repeat=5):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(2):
        out = mod(z); (out.real if is_cx else out).sum().backward()
        mod.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        out = mod(z); (out.real if is_cx else out).sum().backward()
        mod.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return ((time.perf_counter() - t0) / repeat * 1000,
            torch.cuda.max_memory_allocated() / 1e9)


def main():
    dev = torch.device("cuda")
    B, d = 16, 64
    print(f"{'p':>4} {'N':>6} | {'softmax':>16} {'pw3+g':>16} {'pw1':>16}")
    print("      " + "      | " + "  time(ms)  mem(GB)" * 3)
    for p in (7, 11, 13, 17, 19):
        N = p ** 3
        sm = SoftmaxAttn(d).to(dev)
        xr = torch.randn(B, N, d, device=dev)
        zc = torch.randn(B, N, d, dtype=torch.complex64, device=dev)
        t0, m0 = bench(sm, xr, False, dev)
        try:
            a3 = PiecewiseCRAttention(d, p=p, n_flow=3, gate=True).to(dev)
            t3, m3 = bench(a3, zc, True, dev)
        except RuntimeError as e:
            t3, m3 = float("nan"), float("nan")
        a1 = PiecewiseCRAttention(d, p=p, n_flow=1, gate=False).to(dev)
        t1, m1 = bench(a1, zc, True, dev)
        print(f"{p:>4} {N:>6} | {t0:>8.1f} {m0:>7.3f} {t3:>8.1f} {m3:>7.3f} "
              f"{t1:>8.1f} {m1:>7.3f}")


if __name__ == "__main__":
    main()
