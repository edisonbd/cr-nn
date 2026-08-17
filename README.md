# CR-NN: matrix-free attention + O(1) unbounded context

Two mechanisms, both measured on an A800, that attack the two limits of
long-context Transformers:

| | flash attention / KV-cache | **this repo** |
|---|---|---|
| attention compute | $O(N^2)$ | **$O(N\log N)$, 16× faster at $N{=}50$K** |
| context memory | $O(N)$ KV cache (12.3 GB @ 1M tokens, Qwen2.5-0.5B) | **$O(1)$ running state (0.015 GB @ 1.36M tokens)** |

The headline is the second row: a **block-recurrent decoder that keeps a fixed
$O(1)$ complex running state**, so context memory is *constant* instead of
growing linearly — no KV cache, no window, no summarization heuristics.

> **Status**: research code. The honest exploration below shows both what works
> and what does not; treat the negative results as a map, not a claim of SOTA.

---

## 🤝 Call for collaboration

This is an open problem, not a finished product. **We are actively looking for
collaborators** on two specific, well-scoped questions:

1. **CR-attention precision** — the LDR attention reaches 1.12× of a Qwen2-style
   baseline but has not closed the gap. Is there a matrix-free sub-quadratic
   interaction that beats it?
2. **CR-attention memory** — the FFT is locked to fp32 (cuFFT fp16 is power-of-2
   only, incompatible with prime group order), so it is 2× the memory of bf16
   flash. Is there a half-precision path (or a different transform) that
   recovers the memory?

These are documented precisely in [`docs/open_problems.md`](docs/open_problems.md).

**The one thing we believe is already solved — and want validated at scale — is
the O(1) unbounded context** (`block_recurrent.py`): a fixed-size running state
replacing the KV cache. It is easy to describe, easy to re-implement, and —
because it is $O(1)$ — the most likely part to be independently rediscovered.
If you work on long-context decoding, please try it and tell us if it holds up
on real models (see the call-out below).

### Why we are open-sourcing before a paper

Replacing the Transformer's attention turned out to be harder than expected; the
negative results below are the map. Keeping them private would not protect the
ideas (the O(1) state especially is simple enough to be rediscovered), and
would forfeit the one thing that matters — **validation by real use**. So:
priority by public release, validation by adoption, then a paper if warranted.

---

## The two ideas

### 1. Matrix-free attention (LDR): $O(N\log N) + O(Nr)$

Starting from the Szegő projection on the finite Heisenberg group, we ended at a
**low-displacement-rank (LDR) attention**: a Toeplitz part (content-dependent
cross-correlation $Q\star K$, two FFTs) plus a Linformer-style low-rank residual
(position-specific part). It never materializes an $N\times N$ matrix.

### 2. O(1) unbounded context

`block_recurrent.py` — a decoder that carries a fixed-size complex state across
blocks. Context memory is $O(p^3 + d)$, independent of total length.

---

## Quick start

```bash
pip install torch transformers tokenizers
git clone <this repo>
cd cr-nn
export PYTHONPATH=.

# O(1) unbounded context demo (constant memory while streaming blocks)
python experiments/03_toy_seq/block_recurrent.py

# LDR attention vs Qwen2-style / GPT-2 (blockwise, same corpus/budget)
python experiments/03_toy_seq/ldr_attn.py --d 512 --layers 4 --p 17 --r 64
```

---

## Results (all measured, not theoretical)

**Precision** — blockwise next-block prediction, 80M-token book corpus, d=512,
4 layers, matched budget:

| attention | eval ppl | rel. Qwen2 |
|---|---:|---:|
| complex Szegő (twist) | 1833 | 1.72× |
| real spectral (drop complex) | 1460 | 1.37× |
| $Q\star K$ cross-correlation (content-dependent) | 1381 | 1.29× |
| **LDR ($Q\star K$ + low-rank, $r{=}64$)** | **1201** | **1.12×** |
| GPT-2 (MHA, flash) | 1301 | 1.22× |
| Qwen2-style (GQA+RoPE, flash) | 1068 | 1.00× |

**Speed** — attention fwd+bwd, CR vs flash:

| N | CR speedup over flash |
|---|---:|
| 12,167 | 4.4× |
| 24,389 | 8.8× |
| 29,791 | 10.8× |
| 50,653 | **16.2×** |

**Context** — running state vs KV cache:

| approach | memory @ 1.36M tokens |
|---|---|
| CR block-recurrent | **0.015 GB (constant)** |
| Qwen2.5-0.5B KV cache | 12.3 GB ($O(N)$) |

---

## What we learned (the negative results are the map)

A full exploration is logged in `HANDOFF.md` (§1–39). The failures are recorded
because they save others the same detours:

- **The complex dimension is pure overhead.** Dropping it *improves* ppl
  (1833→1460), halves memory, and gives 4.5× speed — the abelian scalar flow
  gains nothing from complex structure.
- **The wall is content-independence, not smoothness.** A fixed kernel (Szegő or
  otherwise) treats every input the same; the fix is a content-dependent
  interaction ($Q\star K$), not a sharper activation.
- **Triple convolution $Q\star K\star V$ explodes** (eval ppl $>10^8$): $O(N^2)$
  accumulation, not fixable by regularisation.
- **Exact fractional-Fourier factorization is impossible** — the non-commutative
  convolution is fundamentally $O(p^4)$; there is no scalar null space.
- **Position-specific capacity must be small**: $r{=}64$ low-rank residual helps,
  $r{=}256$ and stacked rounds overfit.

The precise boundary (matrix-free sub-quadratic attention at ~1.12× of a causal
Transformer) is the honest deliverable.

---

## Repository layout

```
crnn/                 # layers: piecewise CR attention, complex blocks, GeoFFN
experiments/
  01_unit_tests/      # operator correctness (analytic comparisons)
  02_speedup_probe/   # CR vs naive/flash/Qwen memory+speed probes
  03_toy_seq/         # blockwise decoders, LDR attention, ablations
docs/
  paper.md            # technical write-up (top-conference style)
  math.md, assumptions.md
HANDOFF.md            # full project log §1-39
```

## Acknowledgements

This project was developed and completed with the assistance of **DeepSeek Pro**.

## License

This project is released under the [Apache License 2.0](LICENSE).
