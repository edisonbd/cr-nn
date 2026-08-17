"""GeoFFN vs Euclidean ComplexFFN: memory/speed/params as d scales.

The user's point: at small d the embedding+head dominate, so GeoFFN's FFN-level
win is diluted.  The FFN is O(d^2), so the advantage should GROW with d.  Sweep
d and report FFN-only time/memory/params + ratios at fixed N.
"""
from __future__ import annotations

import time

import torch

from crnn.layers.complex_nn import ComplexFFN, GeoFFN


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
            torch.cuda.max_memory_allocated() / 1e6)


def main():
    dev = torch.device("cuda")
    N, B = 4913, 4
    print(f"FFN-only scaling  (N={N}, B={B}, field (B,N,d) complex)")
    print(f"{'d':>4} | {'params euclid':>13} {'geo':>10} | "
          f"{'time euclid':>11} {'geo':>8} {'speedup':>8} | "
          f"{'mem euclid':>10} {'geo':>8} {'mem_ratio':>9}")
    for d in (128, 256, 512, 1024):
        z = torch.randn(B, N, d, dtype=torch.complex64, device=dev)
        eu = ComplexFFN(d, expansion=4).to(dev)
        pe = sum(p.numel() for p in eu.parameters())
        te, me = bench(lambda: eu(z).real.sum().backward(), dev)

        geo = GeoFFN(d, rounds=2, nl="softmodrelu").to(dev)
        pg = sum(p.numel() for p in geo.parameters())
        tg, mg = bench(lambda: geo(z).real.sum().backward(), dev)

        print(f"{d:>4} | {pe/1e6:>8.2f}M {pg/1e6:>7.2f}M | "
              f"{te:>6.1f} {tg:>6.1f} {te/tg:>6.2f}x | "
              f"{me:>6.1f} {mg:>6.1f} {mg/me:>7.2f}x", flush=True)


if __name__ == "__main__":
    main()
