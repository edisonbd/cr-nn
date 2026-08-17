# HANDOFF — CR-NN Project

> Last updated: 2026-08-16
> Status: M1–M4 complete + formal-scale subword validation + precision exploration to LDR
> Tests: 54/54 passing

This is the canonical handoff for the CR-NN project: what it is, what was
tried, what works, what does not, and where it stands. Companion documents:
`docs/math.md` (mathematical foundation), `docs/assumptions.md` (assumptions +
risk register), `docs/paper.md` (technical write-up), `docs/open_problems.md`
(unsolved questions we are asking for help with).

---

## 1. What this project is

**Original goal**: replace softmax attention with the Szegő projection of a CR
(Cauchy–Riemann) manifold, to remove the $O(N^2)$ matrix multiply and reduce
memory while matching Transformer precision.

**Where it actually landed**: the complex CR geometry turned out to be overhead
rather than the point. The valuable, validated contributions are two:

1. **LDR attention** — a matrix-free, sub-quadratic attention: a Toeplitz
   cross-correlation $Q\star K$ plus a Linformer-style low-rank residual. It
   reaches **1.12×** of a Qwen2-style baseline (and beats a same-scale GPT-2),
   with $O(N\log N)+O(Nr)$ compute and no $N\times N$ matrix.
2. **CR-context** — a block-recurrent decoder carrying a fixed $O(1)$ running
   state, giving *constant* context memory: **0.015 GB** at 1.36M tokens, versus
   a Qwen2.5-0.5B KV cache of 12.3 GB ($O(N)$).

The honest conclusion of the exploration: **CR is not a drop-in Transformer
replacement; it is a matrix-free sub-quadratic attention operator with $O(1)$
unbounded context.** See §9 for the full precision evolution and §10 for the
negative results.

---

## 2. Key constraints (hard-won pitfalls)

### p must be prime
The matrix-valued Fourier transform on the finite Heisenberg group $H_p$ is
correct **only for prime $p$**. Composite $p$ breaks the Schrödinger
representation family (low-dimensional irreps go missing), Plancherel fails, and
~5% of energy leaks. Grid resolutions are therefore neighbouring primes
(3, 5, 7, 11, 13, 17, 19, 23, 29, …) with sequence length $N = p^3$.

### Group law uses the one-sided form
$c = c_1 + c_2 + a_1 b_2$ (coefficient 1, one-sided), **not** the symmetric
continuous form $2(x_1 y_2 - y_1 x_2)$ (coefficient 2). They are isomorphic but
differently parametrised; the continuous kernel is evaluated on the lattice
under this law.

### FFT sign conventions
`omega = e^{+2\pi i/p}`, but numpy/torch `fft` uses `e^{-2\pi i/p}`. The forward
transform of the positive-exponent 2-D transform is therefore
`conj(fft2(conj(x)))`.

### Do not claim softmax = Bargmann kernel
"Softmax attention equals a discrete truncation of the Bargmann coherent-state
kernel" has no literature support; it is a structural analogy only.

### cuFFT locks the FFT to fp32
cuFFT's fp16 kernel supports only power-of-2 lengths, incompatible with prime
$p$; and torch has no complex-bf16 dtype. So the FFT runs in fp32 (complex64).
All attempts to reach half precision fail (see §8).

### Exact FrFT is impossible
The non-commutative convolution $\hat f_\lambda \cdot \hat s_\lambda$ is
fundamentally $O(p^4)$; there is no scalar null space. Seven numerical attempts
(DFT eigenbasis, rotation family, symplectic FFT, chirp-z FrFT, holo/anti-holo
phases, un-shear, 625 metaplectic combos) all fail. The "dimension-collapse"
mask for causal masking fails for the same reason: the Szegő projection
amplifies $\bar\partial_b^*$-exact fields rather than annihilating them. The
correct causal realization is blockwise autoregression (§6).

---

## 3. The mathematics (summary)

The finite Heisenberg group $H_p = \mathbb{Z}_p^3$ has group law
$(a,b,c)(a',b',c') = (a+a', b+b', c+c'+ab')$. Its irreducible representations
are the $p-1$ Schrödinger representations $\sigma_\lambda$ ($\lambda\in
\mathbb{Z}_p\setminus\{0\}$, acting on $\mathbb{C}^p$) plus the $p^2$
one-dimensional characters (at $\lambda=0$).

The **Szegő projection** $\mathcal{S}$ projects $L^2(H_p)$ onto the
CR-holomorphic (boundary-holomorphic) subspace. In the group Fourier domain it
is: $\hat{\mathcal{S}}_0 = 0$ (centre modes annihilated) and
$\hat{\mathcal{S}}_\lambda = |v_0\rangle\langle v_0|$ (rank-1 projection onto
the vacuum / lowest-weight vector) for $\lambda\ne 0$. The kernel has closed
form $S(w) = \overline{w}^{\,n+1}/|w|^{2(n+1)}$ with $w = x^2+y^2-it$.

**Two realizations**:

| | matrix Szegő | scalar piecewise (used) |
|---|---|---|
| complexity | $O(p^4)$ | $O(N\log N)$ |
| structure | true non-commutative convolution | diagonal spectral weight (abelian approx.) |
| learnable | no (fixed kernel) | yes (learned spectrum + breakpoints) |
| ppl (toy) | 10.98 | **8.03** |
| speed / memory | 5.7 ms / 1.01 GB | **5.0 ms / 0.35 GB** |

The scalar spectral flow is: chirp (twist) → FFT3 → diagonal weight → activation
→ IFFT3 → chirp → activation. The twist $e^{2\pi i\alpha\, ab/p}$ carries the
non-commutative $ab$-shear; it is a *pointwise* phase, so the whole thing is
matrix-free. The exact version (metaplectic / fractional-Fourier) is impossible
(§2).

---

## 4. Implementation

- `crnn/layers/piecewise_cr_attention.py` — `PiecewiseCRAttention` (the
  matrix-free spectral flow), `szego=True` option (explicit $\lambda=0$
  annihilation), complex activations (modReLU / softmodrelu / radial / GELU),
  `cr_prune` (CR-structured dropout).
- `crnn/layers/complex_nn.py` — `ComplexLinear`, `ComplexFFN`, `ComplexRMSNorm`,
  `GeoFFN` (complex-hypersurface FFN with phase-preserving collapse).
- `crnn/layers/geo_cr.py` — `GeoCRBlock` (fully geometric, no `nn.Linear`).
- `experiments/03_toy_seq/block_recurrent.py` — the $O(1)$ block-recurrent
  decoder (CR-context).
- `experiments/03_toy_seq/ldr_attn.py` — the LDR attention (final best).
- `experiments/03_toy_seq/{real_attn,real_attn2,fluid_attn,lowrank_attn,
  selective_attn,segmented_attn}.py` — the precision-exploration variants.

54/54 unit tests pass (`experiments/01_unit_tests/`, `tests/`).

---

## 5. Speed and memory (measured)

All numbers from test runs on an A800.

**Attention operator, long context** (unified fp32, $B{=}8$, $d{=}128$):

| N | softmax-fp32 (ms / GB) | CR-fp32 (ms / GB) | speedup |
|---:|---:|---:|---:|
| 12167 | 144.9 / 19.37 | 8.7 / 1.62 | 16.7× |
| 50653 | OOM | 38.5 / 5.42 | — |

**CR vs flash (bf16), long context** — CR's $O(N\log N)$ beats flash's $O(N^2)$
tiling:

| N | CR speedup over flash |
|---:|---:|
| 12167 | 4.4× |
| 24389 | 8.8× |
| 29791 | 10.8× |
| 50653 | **16.2×** |

**Memory honesty.** flash already achieves $O(N)$ memory in bf16; CR's fp32
field is ~2× flash's memory. CR's memory advantage is (a) vs *naive* softmax
($O(N^2)$, OOM at 24K) and (b) the $O(1)$ running state vs the $O(N)$ KV cache
(§6) — not vs flash.

---

## 6. CR-context: O(1) unbounded context

`block_recurrent.py` keeps a **fixed-size complex running state** $S\in\mathbb{C}^d$
summarising all previous blocks; each block is processed with global CR attention
and the state is updated. Peak memory is $O(p^3 + d)$, independent of total
length.

| approach | memory @ 1.36M tokens |
|---|---|
| CR block-recurrent | **0.015 GB (constant)** |
| Qwen2.5-0.5B KV cache | 12.3 GB ($O(N)$) |

This is the part we believe is genuinely novel and want validated at scale
(`docs/open_problems.md`). The current state update is a linear projection of
the block mean; a Szegő-structured update is an open improvement.

---

## 7. Precision evolution (the main thread)

Formal subword language modeling, blockwise next-block prediction, 80M-token
book corpus, d=512, 4 layers, matched budget. Held-out ppl:

| step | eval ppl | rel. Qwen2 | mechanism |
|---|---:|---:|---|
| complex Szegő (twist) | 1833 | 1.72× | fixed complex kernel |
| real spectral (drop complex) | 1460 | 1.37× | drop complex overhead |
| $Q\star K$ cross-correlation | 1381 | 1.29× | break content-independence |
| **LDR ($Q\star K$ + low-rank $r{=}64$)** | **1201** | **1.12×** | break position-independence |
| GPT-2 (MHA, flash) | 1301 | 1.22× | — |
| Qwen2-style (GQA+RoPE) | 1068 | 1.00× | — |

**Root cause (corrected):** the binding constraint is **content-independence**,
not smoothness or translation-equivariance. A fixed kernel (Szegő or otherwise)
treats every input the same. The three fixes — drop complex, content-dependent
$Q\star K$, low-rank residual — each close part of the gap, 1.72× → 1.12×.

---

## 8. Memory: the fp32 lock and half-precision attempts

The FFT is locked to fp32 (cuFFT fp16 power-of-2 only; no complex-bf16). Three
paths to half precision all fail:

1. **fp16 direct FFT** — rejected for prime $p$.
2. **fp16 via Bluestein** (chirp-z reduces a prime-length FFT to power-of-2) —
   works at small N, degrades to ~1% at N=101, overflows (NaN) at N=1331; the
   power-of-2 padding also makes buffers *larger* than the direct fp32 FFT.
3. **bf16** — no complex-bf16 dtype, no cuFFT bf16 kernel.

So CR-attention memory is 2× flash bf16. This is `docs/open_problems.md` #2.

---

## 9. Precision ablation — the full table

| attempt | eval ppl | verdict |
|---|---:|---|
| complex Szegő | 1833 | complex is pure overhead |
| real spectral | 1460 | + |
| $Q\star K$ (content-dependent) | 1381 | + |
| **LDR ($Q\star K$ + low-rank $r{=}64$)** | **1201** | **+ best** |
| low-rank alone (fixed pooling) | 1468 | − |
| $r{=}256$ low-rank | 1476 | − overfits |
| stacked rounds $n_{flow}{=}2$ | 1819 | − overfits |
| triple convolution $Q\star K\star V$ | explodes | − $O(N^2)$ accumulation |
| random channel collapse | no rescue | − |
| gating $\sigma(W_g x)\odot v$ | no change | − |
| segmented per-segment kernels | 1681 | − loses global interaction |

Also tested and neutral: modReLU vs softmodrelu (the "smoothness" hypothesis is
refuted — activation sharpness is not the bottleneck), GeoFFN (memory −45% at the
FFN level, but full-model dilution), `mix=True` channel projection (no help; the
FFN already provides channel mixing).

---

## 10. Negative results (the map)

These are recorded as *boundaries*, not failures — they save others the detours:

1. **Exact FrFT impossible** (§2, §30 of the log).
2. **The wall is content-independence** (§7).
3. **Learnable approximation beats fixed exactness** — the matrix Szegő (true
   projection) loses to the learnable scalar spectrum on ppl, speed, *and*
   memory simultaneously (§3 table).
4. **Triple convolution explodes** — $O(N^2)$ accumulation, pathological on
   held-out data, not fixable by regularisation.
5. **Position-specific capacity must be small** — $r=64$ helps, $r=256$ and
   stacked rounds overfit (the content-dependent residual becomes a
   memorisation channel).

---

## 11. The three core lessons

1. **The complex geometry was overhead, not the point.** Dropping the complex
   dimension improved ppl (1833→1460), halved memory, and gave 4.5× speed. The
   abelian scalar flow gains nothing from complex structure.
2. **Content-dependence > position-dependence** — breaking content-independence
   ($Q\star K$) gave −5.4%, breaking position-independence (low-rank residual)
   gave −13%; both are needed (LDR), but the position-specific part is the
   larger contribution.
3. **Within the "matrix-free" constraint, LDR is the convergence point.**
   Without an $N\times N$ matrix or a large projection matrix, the Toeplitz +
   low-rank LDR attention (1.12× Qwen2) is the best interaction found.

---

## 12. Final positioning

**CR-NN is not a Transformer replacement. It is a matrix-free, sub-quadratic
attention operator ($O(N\log N)+O(Nr)$, 1.12× Qwen2) with $O(1)$ unbounded
context (0.015 GB vs 12.3 GB KV cache).**

The two things that are *unmatched* by flash attention and KV-cache decoders are
the $O(N\log N)$ compute and the $O(1)$ state. The precision (1.12×) is honest
and close, not SOTA. Open problems: precision (§7) and memory (§8).

---

## 13. What to do next

1. **Validate CR-context at scale** — the top priority (`docs/open_problems.md`).
2. **Precision** — try learned (not strided) low-rank projections, multi-scale
   cross-correlation, or Mamba-style input-adaptive kernel selection.
3. **Memory** — find a half-precision transform on a prime-order group.
4. **Paper** — only if/when the open-sourced repo gains traction; a workshop
   paper built around the honest negative results is the realistic target.

The full chronological log (39 sections) with all intermediate numbers lives in
the git history of `HANDOFF.md` before this English rewrite; the authoritative
content is here and in `docs/paper.md`.
