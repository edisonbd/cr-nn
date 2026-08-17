"""GeoFFN (complex hypersurfaces + collapse) vs Euclidean ComplexFFN.

Measures (1) intermediate memory of the no-expansion collapse FFN vs the 4x
Euclidean FFN, and (2) the actual collapse ratio (fraction of channels
annihilated per round by the phase-preserving radial threshold).
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
            torch.cuda.max_memory_allocated() / 1e6)  # MB


def main():
    dev = torch.device("cuda")
    d = 128
    B = 4
    print(f"complex FFN memory  (d={d}, B={B}, field (B,N,d) complex)")
    print(f"{'N':>6} | {'ComplexFFN 4x':>14} | {'GeoFFN K=2':>14} | {'mem ratio':>9}")
    print("      " + " |  ms    MB     |  ms     MB     |  geo/euclid")
    for N in (1331, 2197, 4913, 12167):
        z = torch.randn(B, N, d, dtype=torch.complex64, device=dev)

        ffn = ComplexFFN(d, expansion=4).to(dev)
        t0, m0 = bench(lambda: ffn(z).real.sum().backward(), dev)

        geo = GeoFFN(d, rounds=2, nl="softmodrelu").to(dev)
        t1, m1 = bench(lambda: geo(z).real.sum().backward(), dev)

        print(f"{N:>6} | {t0:>5.1f} {m0:>7.1f} | {t1:>5.1f} {m1:>7.1f} | "
              f"{m1/m0:>7.2f}x", flush=True)

    # collapse ratio: how much does the radial threshold actually annihilate?
    z = torch.randn(B, N, d, dtype=torch.complex64, device=dev)
    geo = GeoFFN(d, rounds=3, nl="modrelu").to(dev)
    # negative bias => threshold at -bias > 0, annihilating |z| < -bias
    geo.bias.data.fill_(-0.5)
    ratios = geo.collapse_ratio(z)
    print(f"\ncollapse ratio (bias=-0.5, modrelu): "
          + " ".join(f"round{k}={r:.1%}" for k, r in enumerate(ratios)))


if __name__ == "__main__":
    main()
