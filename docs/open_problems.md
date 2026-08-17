# Open problems & what we tried

This documents the two unsolved problems we are asking for help with, the
blockers hit while trying to replace the Transformer's attention, and the one
contribution we believe is already solved and want validated.

All numbers are measured on an A800; the corpus is an 80M-token public-domain
book corpus (8K BPE), task is blockwise next-block prediction, d=512, 4 layers,
matched budget.

---

## Open problem 1 — CR-attention precision

**Current best:** LDR attention = Toeplitz cross-correlation $Q\star K$ (two
FFTs) + a Linformer-style low-rank residual ($r=64$), reaching **1.12×** of a
Qwen2-style (GQA+RoPE+flash) baseline and beating a same-scale GPT-2.

**The gap:** 1.12× remains, vs the full $O(N^2)$ softmax at 1.00×.

**What we tried (and the outcome):**

| attempt | eval ppl | verdict |
|---|---:|---|
| complex Szegő projection | 1833 | the complex dimension is pure overhead |
| real spectral (drop complex) | 1460 | +: memory halved, 4.5× faster |
| content-dependent $Q\star K$ | 1381 | +: breaks content-independence |
| **LDR ($Q\star K$ + low-rank, $r=64$)** | **1201** | **+ best; position-specific residual** |
| low-rank alone (fixed pooling) | 1468 | −: needs the Toeplitz part |
| $r=256$ low-rank | 1476 | −: overfits |
| stacked rounds $n_{flow}=2$ | 1819 | −: overfits (memorisation channel) |
| triple convolution $Q\star K\star V$ | explodes | −: $O(N^2)$ accumulation, pathological |
| gating $\sigma(W_g x)\odot v$ | no change | −: modulates amplitude, not the kernel |
| segmented per-segment kernels | 1681 | −: loses global interaction |

**The precise question.** Is 1.12× the ceiling for *matrix-free* sub-quadratic
attention (no $N\times N$, no large projection), or is there an interaction we
missed? Candidates we have *not* fully explored: learned (not strided) low-rank
projections, multi-scale cross-correlation, and input-adaptive kernel selection
(Mamba-style selectivity).

---

## Open problem 2 — CR-attention memory

The FFT is **locked to fp32**: cuFFT's fp16 kernel supports only power-of-2
lengths, incompatible with prime group order $p$; and there is no complex-bf16
dtype. So the field is fp32, 2× the memory of bf16 flash.

**What we tried:**

- `half=True` (complex32 storage + fp32 FFT): no memory win — the fp32 FFT
  buffers dominate.
- fp16 FFT via Bluestein (chirp-z reduces a prime-length FFT to a power-of-2
  one): works at small N, loses precision to ~1% at N=101, and **overflows (NaN)
  at N=1331**; the power-of-2 padding also makes the buffers *larger* than the
  direct fp32 FFT.

**The precise question.** Is there a half-precision transform on a *prime*-order
group, or a different matrix-free operator that is bf16-compatible?

---

## Blockers hit while replacing the Transformer

Three theorem-level or empirical boundaries, each with a pointer into the log
(`HANDOFF.md`):

1. **Exact fractional-Fourier factorization is impossible.** The non-commutative
   convolution $\hat f_\lambda \cdot \hat s_\lambda$ is fundamentally $O(p^4)$;
   there is no scalar null space (the $\lambda$-squeeze dilation by $\sqrt\lambda$
   is ill-posed on $\mathbb{Z}_p$). Seven numerical attempts failed (§30).
2. **The wall is content-independence, not smoothness or equivariance.** A fixed
   kernel (Szegő or otherwise) treats every input the same; Mamba/Hyena are also
   translation-equivariant but *content-dependent*, which is what matters. The
   fix is a content-dependent interaction, not a sharper activation (§39).
3. **"Learnable approximation beats fixed exactness."** The matrix Szegő (true
   projection, $O(p^4)$, fixed kernel) loses to the learnable scalar spectral
   weight on ppl (10.98 vs 8.03), speed (5.7 vs 5.0 ms), and memory (1.01 vs
   0.35 GB) simultaneously (§39).

---

## The part we believe is solved — please validate it

**O(1) unbounded context** (`block_recurrent.py`): a decoder keeps a fixed-size
complex running state across blocks, so context memory is constant
($O(p^3+d)$) instead of the $O(N)$ KV cache.

- Measured: **0.015 GB** processing 1.36M tokens of blocks, vs a Qwen2.5-0.5B
  KV cache of 12.3 GB at 1M tokens.
- It is simple enough to be independently rediscovered, so we are publishing it
  now to establish priority **and** to find out whether it holds on real models.

**If you work on long-context decoding:** please try swapping the KV cache for a
fixed running state and report whether quality holds. The questions we cannot
answer alone: does the fixed state retain enough information for
retrieval/grounding? does it degrade on copy tasks or code? what is the right
state-update (linear mean vs a learned/Szegő-structured update)?
