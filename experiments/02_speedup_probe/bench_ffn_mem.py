"""Memory breakdown: Euclidean FFN vs CR-geometric channel mixer.

Confirms (a) the Euclidean FFN is the dominant intermediate-memory consumer in
the current CRBlock, and (b) how much a matrix-free CR-geometric channel mixer
(cyclic-DFT, no 4x expansion) saves.  Position-domain FFT is identical in both,
so the difference isolates the channel-mixing (FFN) geometry.

Models:
  euclid_ffn  : nn.Linear(2d->8d) + GELU + nn.Linear(8d->2d)   (current CRBlock)
  geo_mixer   : K rounds of channel-DFT + pointwise modReLU     (cr-geo)
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from crnn.layers.geo_cr import GeoChannelMix


def euclid_ffn(d):
    return nn.Sequential(nn.Linear(2 * d, 8 * d), nn.GELU(),
                         nn.Linear(8 * d, 2 * d))


def geo_mixer(d, rounds):
    mix = nn.ModuleList([GeoChannelMix(d) for _ in range(rounds)])
    return mix


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
    print(f"channel-mixing memory  (d={d}, B=4, real field (B,N,2d))")
    print(f"{'N':>6} | {'euclid FFN':>14} | {'geo mixer K=3':>14} | {'mem ratio':>9}")
    print("      " + " |  ms    MB     |  ms     MB     |  geo/euclid")
    for N in (1331, 2197, 4913, 6859, 12167):
        x = torch.randn(4, N, 2 * d, device=dev)   # the (B,N,2d) real field

        ffn = euclid_ffn(d).to(dev)
        t0, m0 = bench(lambda: ffn(x).sum().backward(), dev)

        mix = geo_mixer(d, 3)
        mix = mix.to(dev)
        z = torch.complex(x[..., :d], x[..., d:])

        def gm():
            h = z
            for m in mix:
                h = m(h)
                h = torch.nn.functional.relu(h.real) + 1j * h.imag
            h.real.sum().backward()
            for m in mix:
                m.zero_grad(set_to_none=True)
        t1, m1 = bench(gm, dev)

        print(f"{N:>6} | {t0:>5.1f} {m0:>7.1f} | {t1:>5.1f} {m1:>7.1f} | "
              f"{m1/m0:>7.2f}x", flush=True)


if __name__ == "__main__":
    main()
