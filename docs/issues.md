# GitHub issues to create

Copy each block below into a separate GitHub issue when the repo goes live.
Each issue is self-contained: background, what was tried, the exact blocker, how
to reproduce, and what "help" concretely means. Keep titles as written so they
are searchable.

---

## Issue 1 — Close the precision gap: matrix-free sub-quadratic attention at 1.12×, can it reach 1.0×?

**Background.** This repo replaces softmax attention with matrix-free operators
(no $N\times N$ matrix). The best so far is an **LDR attention** — a Toeplitz
cross-correlation $Q\star K$ (two FFTs) plus a Linformer-style low-rank residual
($r=64$) — reaching **1.12×** of a Qwen2-style baseline (GQA+RoPE+flash) and
beating a same-scale GPT-2.

**What was tried (measured, 80M-token corpus, d=512, 4 layers, blockwise):**

| attention | eval ppl | rel. Qwen2 |
|---|---:|---:|
| complex Szegő (twist) | 1833 | 1.72× |
| real spectral (drop complex) | 1460 | 1.37× |
| $Q\star K$ cross-correlation | 1381 | 1.29× |
| **LDR ($Q\star K$ + low-rank $r{=}64$)** | **1201** | **1.12×** |
| low-rank alone (fixed pooling) | 1468 | 1.37× |
| $r{=}256$ low-rank | 1476 | overfits |
| stacked rounds $n_{flow}{=}2$ | 1819 | overfits |
| triple convolution $Q\star K\star V$ | explodes | pathological |
| gating $\sigma(W_g x)\odot v$ | no change | — |
| segmented per-segment kernels | 1681 | loses global interaction |

**The blocker.** The remaining gap is Toeplitz vs full matrix. Position-specific
capacity must be *small* ($r=64$ helps, $r=256$ overfits), so naively increasing
it fails.

**Reproduce.**
```bash
PYTHONPATH=. python experiments/03_toy_seq/ldr_attn.py --d 512 --layers 4 --p 17 --r 64
```

**Help wanted.** Any *matrix-free* sub-quadratic interaction that gets below
1.12×. Candidates not yet explored: learned (not strided) low-rank projections,
multi-scale cross-correlation, input-adaptive kernel selection (Mamba-style
selectivity).

---

## Issue 2 — Escape the fp32 lock: a half-precision transform on a prime-order group?

**Background.** The attention is built on an FFT over a finite Heisenberg group
$H_p$ with **prime** order $p$ (prime is required for Plancherel — see
`docs/math.md` §7.1). But cuFFT's fp16 kernel supports only power-of-2 lengths,
and torch has no complex-bf16 dtype, so the FFT is **locked to fp32**, making
attention memory ~2× that of bf16 flash.

**What was tried.**

1. fp16 direct FFT — rejected for prime $p$.
2. fp16 via Bluestein (chirp-z reduces a prime-length FFT to power-of-2) —
   works at small N, degrades to ~1% at N=101, **overflows (NaN) at N=1331**;
   the power-of-2 padding also makes buffers *larger* than the direct fp32 FFT.
3. bf16 — no complex-bf16 dtype, no cuFFT bf16 kernel.

**Reproduce.** `experiments/02_speedup_probe/bluestein_probe.py`.

**Help wanted.** A half-precision (fp16/bf16) transform on a *prime*-order group,
or a different matrix-free operator that is bf16-compatible.

---

## Issue 3 — Validate the O(1) unbounded context on real models

**Background.** This is the part we believe is novel and want validated at scale:
a **block-recurrent decoder that carries a fixed $O(1)$ running state** instead
of an $O(N)$ KV cache. Measured: **0.015 GB** processing 1.36M tokens of blocks,
vs a Qwen2.5-0.5B KV cache of 12.3 GB at 1M tokens.

**Why publish now.** The idea (a fixed running state across blocks) is simple
enough to be independently rediscovered; publishing establishes priority and —
more importantly — is the only way to find out if it holds on real models.

**Reproduce.** `experiments/03_toy_seq/block_recurrent.py`.

**Help wanted.** If you work on long-context decoding, please swap a KV cache for
a fixed running state and report whether quality holds. Questions we cannot
answer alone:
- does the fixed state retain enough information for retrieval/grounding?
- does it degrade on copy tasks or code?
- what is the right state update (linear mean vs a learned/Szegő-structured
  update)?

---

## Issue 4 — (optional) A Szegő-structured state update for the O(1) context

**Background.** The current O(1) state update is a linear projection of the block
mean (`block_recurrent.py`). A natural improvement is to make the state update
itself a CR/Szegő projection, so the running summary is CR-structured rather than
a Euclidean mean.

**Help wanted.** A matrix-free, $O(1)$ state update with better retention than a
linear mean, ideally evaluated on a long-context retrieval or copy benchmark.
