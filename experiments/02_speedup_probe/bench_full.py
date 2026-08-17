"""Full-model (ToyLM) forward+backward throughput + peak memory: CR vs transformer.

This is the *deployment-relevant* number (the whole model, not just the
attention): steps/s and peak VRAM for the matrix-free CR model vs a
same-parameter transformer, as the window N = p^3 grows.  The attention-level
win (bench_mem.py) shows up here once N is large enough that attention, not the
FFN, dominates the memory.
"""

from __future__ import annotations

import time

import torch

from crnn.models import ToyLM


def bench(model, x, dev, repeat=5):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(2):
        out = model(x)
        out.float().mean().backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        out = model(x)
        out.float().mean().backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return ((time.perf_counter() - t0) / repeat * 1000,
            torch.cuda.max_memory_allocated() / 1e9)


def main():
    dev = torch.device("cuda")
    B, d, vocab = 16, 64, 32
    print(f"{'p':>4} {'N':>6} | {'transformer(10L)':>20} | {'CR(nf1,nogate)':>20}")
    print("      " + "      |  time(ms)  mem(GB)   |  time(ms)  mem(GB)")
    for p in (7, 11, 13, 17):
        N = p ** 3
        x = torch.randint(0, vocab, (B, N), device=dev)
        trf = ToyLM(vocab=vocab, d_model=d, n_layers=10, p=p,
                    block_type="transformer").to(dev)
        t0, m0 = bench(trf, x, dev)
        cr = ToyLM(vocab=vocab, d_model=d, n_layers=3, p=p, block_type="cr-vec",
                   attn_type="piecewise", n_flow=1, gate=False, nl="gelu").to(dev)
        t1, m1 = bench(cr, x, dev)
        print(f"{p:>4} {N:>6} | {t0:>8.1f} {m0:>7.3f}   | {t1:>8.1f} {m1:>7.3f}")


if __name__ == "__main__":
    main()
