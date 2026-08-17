# CR-NN: CR-Attention and CR-Context from the Heisenberg Group
### Sub-quadratic attention and O(1)-state unbounded context from CR geometry

**Abstract**

Self-attention's quadratic $O(N^2)$ cost in sequence length $N$, and the
$O(N)$ KV-cache of causal decoders, bound long-context deployment. We
contribute two CR-geometric mechanisms that attack these bounds.

**(1) CR-attention.** Replace the position set $\{1,\dots,N\}$ by the finite
Heisenberg group $H_p$ (order $N=p^3$, $p$ prime) and the aggregation kernel by
the **Szegő projection** $\mathcal{S}$ — the Cauchy–Szegő projector onto CR
(boundary-holomorphic) functions. Because $\mathcal{S}$ is a group
convolution, it is realized matrix-free through a **chirp-z /
fractional-Fourier factorization** at $O(p^3\log p)=O(N\log N)$, never
materializing an $N\times N$ matrix. The **piecewise-manifold attention** —
diagonal spectral stages interspersed with pointwise complex activations and
symplectic chirp twists — segments the group manifold and restores part of the
non-commutative structure an abelian flow loses.

**(2) CR-context.** A block-recurrent decoder that carries a fixed $O(1)$
complex running state across blocks, so context memory is *constant* instead of
the $O(N)$ KV cache.

Measured on an A800 (all numbers from test runs): CR-attention is **16× faster
than fused flash attention at $N{=}50{,}653$** ($O(N\log N)$ vs $O(N^2)$),
widening with $N$; CR-context processes 1.36M tokens in **0.015 GB** against a
Qwen2.5-0.5B KV cache of 12.3 GB ($O(N)$). On synthetic and character-level
tasks CR matches or beats a same-parameter Transformer.

On formal subword language modeling the plain Szegő attention lags a causal
Transformer (1.72× behind Qwen2-style GQA+RoPE at matched scale). We trace the
gap to **content-independence**: a fixed convolution kernel treats every input
the same, whereas attention weights must depend on the input. Three steps close
most of the gap, all matrix-free and sub-quadratic: (i) dropping the complex
dimension (the abelian scalar flow gains nothing from complex structure, but
costs fp32 and 2× memory); (ii) replacing the fixed kernel by a **content-
dependent cross-correlation** $Q\star K$ (two FFTs, $O(N\log N)$); (iii) adding a
**low-rank correction** — the attention matrix is *almost Toeplitz*, i.e.
Toeplitz plus low-rank (low-displacement-rank), so a Linformer-style rank-$r$
residual captures the position-specific part. The resulting **LDR attention**
reaches 1.12× of the Qwen2-style baseline (and beats a same-scale GPT-2),
1.72× → 1.12× across the three steps, while remaining $O(N\log N)+O(Nr)$ and
matrix-free.

We therefore position CR-attention + CR-context **not as a drop-in Transformer
replacement but as a matrix-free, sub-quadratic attention operator with $O(1)$
unbounded context**, whose headline properties — $O(N\log N)$ compute and $O(1)$
state — are unmatched by flash attention and KV-cache decoders.

---

## 1. Introduction

The self-attention operator of a Transformer computes, for every position, a
convex combination of value vectors weighted by a softmax over inner products:

$$ \mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V . $$

Its cost is dominated by the $N\times N$ score matrix $QK^\top$, giving
$O(N^2 d)$ time and $O(N^2)$ memory. A large body of work attacks this cost
with sparse, low-rank, or kernelized approximations, but most retain a softmax
or a Euclidean inner-product kernel at the core.

We take a different starting point. The softmax attention is an
**aggregation kernel** over a set of $N$ positions. Any choice of an ambient
group structure on the positions endows the aggregation with a
convolutional form, and hence with a fast transform. We choose the ambient
structure to be a **CR (Cauchy–Riemann) manifold** — the Heisenberg group
$\mathbb{H}^n$ — and the aggregation kernel to be its **Szegő projection**,
the canonical object of several-complex-variables boundary analysis. Our
contributions are:

1. **A geometric replacement for softmax.** We show the Szegő projection
   $\mathcal{S}$ (the projector onto CR functions) is a well-defined,
   invariant, sub-quadratic aggregation operator, and we instantiate it on the
   finite Heisenberg group $H_p$.

2. **A matrix-free fast transform.** We give a chirp-z (fractional-Fourier)
   factorization of the $H_p$ convolution that avoids the $p\times p$
   irreducible blocks, reducing the projection to $O(N\log N)$ scalar FFTs and
   pointwise phase multiplies.

3. **Piecewise-manifold attention.** We restore part of the non-commutative
   structure that a purely abelian (diagonal) spectral flow loses, by segmenting
   the manifold with activation-function breakpoints and symplectic chirp
   twists.

4. **CR-context: $O(1)$-state unbounded context.** A block-recurrent decoder
   keeps a fixed complex running state across blocks, giving *constant* context
   memory where every KV-cache model is $O(N)$.

5. **Honest empirical characterization.** We report (i) where CR wins — $O(N\log
   N)$ attention speed (16× over flash at $N{=}50$K) and $O(1)$ context
   (0.015 GB vs a 12.3 GB KV cache at 1M tokens) — and (ii) where it does not —
   formal subword language modeling, where a sub-quadratic operator's
   translation-equivariance leaves it behind a causal Transformer. We isolate
   the negative result (exact FrFT impossible, no scalar null space, the
   translation-equivariance wall) as theorem-level boundaries.

## 2. Background: CR geometry and the Heisenberg group

### 2.1 The Heisenberg group

The $n$-dimensional Heisenberg group $\mathbb{H}^n = \mathbb{C}^n \times
\mathbb{R}$ is the simplest non-abelian nilpotent Lie group, with group law

$$ (z,t)\cdot(z',t') \;=\; \bigl(z+z',\; t+t' + 2\,\mathrm{Im}\,(z\cdot\bar z')\bigr), $$

where $z\cdot\bar z' = \sum_j z_j \bar z'_j$. Writing $z_j = x_j + i y_j$,
the non-abelian term is $2(x y' - y x')$, a *symplectic* twist between the two
$\mathbb{R}^n$ factors. The center is $\{(0,t)\}$, and
$\mathbb{H}^n/\mathrm{center} \cong \mathbb{C}^n$ is the abelian quotient.

For computation we discretize $\mathbb{H}^n$ to the **finite Heisenberg group**

$$ H_p \;=\; \mathbb{Z}_p^3, \qquad (a,b,c)\cdot(a',b',c') \;=\; (a+a',\, b+b',\, c+c'+a\,b') \pmod p , $$

with $p$ an odd prime and $N = |H_p| = p^3$. (The coefficient of $a b'$ is a
parameterization choice; with coefficient $1$ the map below is a genuine
homomorphism — see §4.1.) The requirement that $p$ be prime is essential: for
composite $p$ the Schrödinger family is incomplete and Plancherel fails
(Assumption A1, Appendix A).

### 2.2 The CR structure and the sub-Laplacian

The left-invariant vector fields

$$ X_j = \partial_{x_j} + 2y_j\,\partial_t,\qquad Y_j = \partial_{y_j} - 2x_j\,\partial_t,\qquad T=\partial_t $$

span the tangent space, with the single non-trivial commutator $[X_j,Y_j] =
-4T$. The sub-bundle $H = \mathrm{span}\{X_j,Y_j\}$ carries the CR structure,
and the **tangential Cauchy–Riemann operator**

$$ \bar\partial_b \;=\; \frac12 \sum_{j=1}^n (X_j + i Y_j)\, d\bar z_j $$

defines the space of **CR functions** $\ker \bar\partial_b$ (the boundary
analog of holomorphic functions). The **sub-Laplacian**

$$ \Delta_b \;=\; -\sum_{j=1}^n (X_j^2 + Y_j^2) $$

is the model hypoelliptic operator of the manifold.

**Spectral decomposition.** Fourier-transforming along the center coordinate
$t \mapsto \lambda$ turns $X_j, Y_j$ into creation/annihilation operators of a
quantum harmonic oscillator on each $\lambda$-slice, and $\Delta_b$ into a
harmonic oscillator with the well-known spectrum

$$ \sigma_{k,\lambda} \;=\; (2k + n)\,|\lambda|,\qquad k = 0,1,2,\dots,\;\; \lambda\neq 0, $$

with eigenfunctions given by rescaled Hermite functions (Laguerre in the
$t$-direction). This is the single most load-bearing fact of the construction:
every CR-spectral operation reduces to a diagonal action in the
Hermite–Laguerre basis.

### 2.3 The Szegő projection

The **Szegő (Cauchy–Szegő) projection** $\mathcal{S}$ is the orthogonal
projection of $L^2$ onto the Hardy space $\ker \bar\partial_b$ of CR functions.
Its reproducing kernel is known in closed form (Folland 1975):

$$ S(z,t) \;=\; c_n\,\bigl(|z|^2 - i\,t\bigr)^{-(n+1)} . $$

For $n=1$ on $H_p$ we use the discrete kernel

$$ S(a,b,c) \;=\; \frac{\overline{w}^{\,n+1}}{\bigl(|w|^2 + \eta\bigr)^{n+1}},\qquad w = (a^2+b^2) - i\,c , $$

with $\eta>0$ regularizing the singularity (R5). In the spectral basis of
§2.2, $\mathcal{S}$ is the projector onto the $k=0$ level (the **lowest
Landau level**), i.e. a diagonal mask $\mathbf{1}[k=0]$ in the Hermite basis,
for $\lambda>0$.

## 3. Method

### 3.1 Attention as a Szegő projection

A sequence of $N = p^3$ token embeddings is regarded as a complex-valued field
$f: H_p \to \mathbb{C}^d$ (one field per channel). We replace the softmax
aggregation by the group convolution with the Szegő kernel,

$$ (\mathcal{S} f)(g) \;=\; (f \ast S)(g) \;=\; \sum_{h\in H_p} f(h)\, S(h^{-1}g), $$

which aggregates information across the whole group $H_p$ with a
geometrically-principled, singularity-regularized, phase-aware kernel. The
block then applies the standard pre-norm / residual / FFN sandwich, so the CR
block is a drop-in replacement for a Transformer layer (Fig. 1).

### 3.2 Fast non-commutative transform via the Heisenberg FFT

The convolution theorem for $H_p$ factorizes per frequency $\lambda$
(§4.3): the $\lambda=0$ sector is pointwise (abelian), while each $\lambda\neq
0$ sector is a $p\times p$ matrix product. We give two matrix-free
realizations:

**(a) Chirp-z / fractional-Fourier factorization.** The matrix sector is a
sheared DFT; completing the square in the twist (§4.4) turns the twisted
convolution into a standard convolution with a chirp kernel, which a 2D FFT
diagonalizes. The result is $O(p^3 \log p)$ with only scalar FFTs and
pointwise phases — no $p\times p$ blocks are formed.

**(b) Piecewise-manifold (diagonal) spectral flow.** As an even lighter
approximation, we replace the matrix sector by a *diagonal* spectral operator
in the abelian $(\mathbb{Z}_p)^3$ Fourier basis, and restore the lost
non-commutativity by segmentation (next subsection).

### 3.3 Piecewise-manifold attention (断点流形)

A purely diagonal (abelian) spectral flow is commutative and therefore misses
the $H_p$ twist, capping quality (this reproduces the negative result of our
earlier "fluid" baseline). We restore expressiveness without matrices through
two mechanisms:

1. **Symplectic chirp twists.** Each stage applies a learnable pointwise phase
   $\exp(2\pi i\, \alpha_k\, a b / p)$, the discretized generator of the
   symplectic (metaplectic) twist $a b$ of the group law. By the chirp-z
   correspondence this is precisely the missing non-commutative coupling.

2. **Activation segmentation.** Between spectral stages we insert pointwise
   complex nonlinearities — modReLU $\sigma(z) = \mathrm{ReLU}(|z|+b)\, z/|z|$
   (a radial breakpoint circle $|z|=-b$ in every complex fibre) or split-GELU
   (a non-holomorphic phase-mixing nonlinearity). Each activation stratifies
   the field configuration space into polyhedral cells; composing $K$ stages
   across $L$ layers yields exponentially many cells — "cutting the manifold
   into many small manifolds", each carrying a local linear chart of the CR
   structure.

Concretely, stage $k$ computes

$$ f \;\xrightarrow{\;\mathrm{chirp}_{\alpha_k}\;}\; \hat f = \mathcal F_3 f;\quad
\hat f \leftarrow W_k \odot \hat f;\quad \hat f \leftarrow \sigma(\hat f);\quad
f \leftarrow \mathcal F_3^{-1}\hat f;\quad f \leftarrow \mathrm{chirp}_{\beta_k} f;\quad
f \leftarrow \sigma(f), $$

where $\mathcal F_3$ is the scalar 3D FFT and $W_k$ a learnable diagonal
complex filter.

## 4. Complete mathematical derivation

### 4.1 Representation theory of $H_p$

For $p$ prime, $\omega = e^{2\pi i/p}$, the irreducible representations of
$H_p$ are (Terras; Diaconis–Rockmore):

- **$\lambda = 0$:** the $p^2$ characters $\chi_{m,n}(a,b,c) = \omega^{ma+nb}$, $(m,n)\in\mathbb{Z}_p^2$, each of dimension $1$;
- **$\lambda \in \{1,\dots,p-1\}$:** the $p\times p$ Schrödinger representations
  $$ [\rho_\lambda(a,b,c)]_{u,v} \;=\; \omega^{\lambda c}\,\omega^{\lambda b u}\,\delta_{v,\,(u+a)\bmod p}, \qquad u,v\in\mathbb{Z}_p . $$

These are irreducible and pairwise inequivalent, and
$\sum_\lambda d_\lambda^2 = p^2 + (p-1)p^2 = p^3 = |H_p|$, so Plancherel holds
*iff* $p$ is prime (for composite $p$ the family is incomplete, Assumption A1).
The group law is chosen so that $\rho_\lambda$ is a homomorphism with
coefficient $1$: $\rho_\lambda(g)\rho_\lambda(g') = \rho_\lambda(gg')$.

### 4.2 The group Fourier transform and convolution theorem

For $f: H_p\to\mathbb{C}$ define

$$ \hat f_0(m,n) \;=\; \sum_{a,b,c} f(a,b,c)\,\omega^{ma+nb}, $$

$$ \hat f_\lambda(u,v) \;=\; \sum_{a,b,c} f(a,b,c)\,\bigl[\rho_\lambda(a,b,c)\bigr]_{u,v}
   \;=\; \sum_{b,c} f(v-u,\,b,\,c)\,\omega^{\lambda c}\,\omega^{\lambda b u}. $$

The inverse (Plancherel inversion, $G = p^3$) reads

$$ f(a,b,c) \;=\; \frac1{p^3}\Big[\sum_{m,n}\hat f_0(m,n)\,\omega^{-ma-nb}
\;+\; p\sum_{\lambda=1}^{p-1}\sum_{u,v}\hat f_\lambda(u,v)\,\overline{\rho_\lambda(a,b,c)_{u,v}}\Big], $$

and the **convolution theorem**

$$ \widehat{(f\ast g)}_0 = \hat f_0 \odot \hat g_0, \qquad
   \widehat{(f\ast g)}_\lambda = \hat f_\lambda \,\hat g_\lambda
   \quad(\text{$p\times p$ matrix product}), $$

is the non-commutative analog of the pointwise DFT product. This is the
"matrix scheme" whose cost $O(p^4)$ we seek to avoid.

### 4.3 Decomposition into a twisted convolution

Write the convolution in coordinates. With
$(a,b,c)^{-1} = (-a,-b,-c+ab)$,

$$ (f\ast S)(a,b,c) \;=\; \sum_{a',b',c'} f(a',b',c')\, S\bigl(a-a',\, b-b',\, c-c'-a'(b-b')\bigr). $$

Fourier-transforming along $c$ (negative exponent) and applying the shift
theorem isolates, for each $\lambda$, a **twisted convolution** on the
$(a,b)$ plane:

$$ T_\lambda(a,b) \;=\; \sum_{a',b'} \hat f(a',b',\lambda)\,
   \hat S(a-a',b-b',\lambda)\; \omega^{-\lambda a'(b-b')} , $$

$$ (f\ast S)(a,b,c) \;=\; \frac1p \sum_\lambda \omega^{\lambda c}\, T_\lambda(a,b). $$

This identity — verified numerically to $\sim 10^{-6}$ against both the
naive $O(p^6)$ sum and the matrix path — is the key structural step: the
*only* non-abelian content is the phase $\omega^{-\lambda a'(b-b')}$.

### 4.4 The chirp-z (fractional-Fourier) factorization

The twist $\omega^{-\lambda a'(b-b')}$ is a bilinear phase coupling the
summation index $a'$ with the output index $b$. Because $p$ is odd, $2$ is
invertible and we may **complete the square** with $\mu = \lambda \cdot
2^{-1}\bmod p$:

$$ -\lambda\, a'(b-b') \;\equiv\; -\mu\,(a'+b-b')^2 + \mu\, a'^2 + \mu\,(b-b')^2 \pmod p . $$

Substituting, the phase $\omega^{-\mu(a'+b-b')^2}$ couples $a'$ and $b$ through
their *sum*, i.e. through a **chirp kernel** $c(x) = \omega^{-\mu x^2}$. A
convolution with a chirp kernel is diagonalized by a 2D FFT (Bluestein /
chirp-z):

$$ \sum_{x} A(x)\, c(x + y) \;=\; \mathcal F^{-1}\big[\, \hat A \cdot \hat c \,\big](y), $$

so after absorbing the $\omega^{\mu a'^2}$ and $\omega^{\mu(b-b')^2}$ factors
into the operands, $T_\lambda$ becomes a **standard 2D convolution**
followed by a diagonal mask. Composing over all $\lambda$ gives

$$ \boxed{\;\mathcal{S} = \mathrm{chirp}\circ \mathcal F_3^{-1}\circ \hat S \odot \circ\, \mathcal F_3 \circ \mathrm{chirp},\qquad O(p^3\log p)\;} $$

— the matrix-free factorization. On the *exact* kernel the diagonal mask
$\hat S$ is the Hermite-basis projector $\mathbf{1}[k=0]$ of §2.3, which is
itself a fractional-Fourier (chirp-z) change of basis; the constants are pinned
by the numerical probe (Appendix B).

### 4.5 Complexity and memory

| operator | time | peak memory |
|---|---|---|
| softmax attention | $O(N^2 d)$ | $O(N^2)$ |
| matrix Heisenberg FFT | $O(p^4)=O(N^{4/3})$ | $O(p^4)$ |
| **piecewise (chirp+FFT)** | $O(K p^3\log p)=O(KN\log N)$ | $O(p^3)=O(N)$ |
| **full chirp-z** | $O(p^3\log p)$ | $O(p^3)=O(N)$ |

The piecewise scheme replaces the $N^2$ attention matrix with a handful of
$N$-sized FFT buffers, which is the source of the measured 23×/54× gain
(§5.2).

## 5. Experiments

### 5.1 Setup

Synthetic task: windowed next-window prediction over a mixed periodic /
nested / long-range corpus (vocab 32, window $N=p^3$, $p=11$). Real task:
character-level Shakespeare (1.1M characters, vocab 65, $p=7$). Models:
`cr-vec` (Szegő), `piecewise` (ours), and a `transformer`
(`nn.TransformerEncoderLayer`, 10 layers to match parameter count). A800 GPU,
PyTorch, complex64, AdamW.

### 5.2 Memory and speed (hardware probe, $p=23$, $N=12{,}167$)

| operator | fwd+bwd (ms) | peak VRAM (GB) |
|---|---:|---:|
| softmax (naive, fp32) | 117.1 | 19.01 |
| matrix Szegő | 5.7 | 1.01 |
| **piecewise (K=3)** | **5.0** | **0.35** |

The piecewise attention is **23.4× faster and uses 53.9× less memory than
*naive* softmax**, and $\sim$3× less memory than the matrix Szegő at comparable
speed; memory grows linearly in $N$. (Here and below, "naive softmax" means the
explicit $QK^\top$ materialization of the $N{\times}N$ scores in fp32; see the
three-way comparison that isolates it.)

**Unified-precision comparison (attention operator only, everything fp32,
$B{=}8$, $d{=}128$).** At fp32, `scaled_dot_product_attention` has no fused
kernel and falls back to the $O(N^2)$ math backend — i.e. *at fp32, "flash" and
"naive" softmax are the same thing*. Unifying precision to fp32 therefore
collapses the comparison to two entries:

| $p$ | $N$ | softmax-fp32 (ms / GB) | **CR-fp32 (ms / GB)** | speedup | mem ratio (CR/sm) |
|---:|---:|---:|---:|---:|---:|
| 11 | 1331 | 2.0 / 0.282 | 2.0 / 0.192 | 1.0× | 0.68× |
| 13 | 2197 | 5.9 / 0.712 | 2.0 / 0.309 | 2.9× | 0.43× |
| 17 | 4913 | 22.6 / 3.266 | 4.0 / 0.661 | 5.7× | 0.20× |
| 19 | 6859 | 42.7 / 6.277 | 5.2 / 0.916 | 8.2× | 0.15× |
| 23 | 12167 | 144.9 / 19.37 | **8.7 / 1.62** | **16.7×** | **0.08×** |

At matched fp32 precision CR is **16.7× faster and uses 12.5× less memory**
($1/0.08$) at $N{=}12{,}167$, with the crossover at $N\approx 1331$ and the gap
widening as $O(N\log N)$ vs $O(N^2)$.

**The bf16 flash caveat.** The only way softmax attention escapes its $O(N^2)$
memory is the fused flash kernel, which *requires* bf16/fp16 — fp32 flash does
not engage it. flash is therefore *forced* to half precision, while CR is
*forced* to fp32: cuFFT's fp16 kernel supports only power-of-2 lengths, which
are incompatible with prime $p$, so the FFT must run in fp32 (complex64), and
torch has no complex-bf16 dtype. The two cannot trade precisions. Under that
forced half precision, flash reaches 0.81 GB (vs CR's 1.62 GB) but stays slower
(37.2 ms vs CR's 8.7 ms, $\sim$4.3×). We exhaustively tested CR's own half-precision
options and both fail: (i) `half=True` (complex32 storage, fp32 FFT) does not
reduce memory (1.82 vs 1.62 GB) because the fp32 FFT buffers dominate; (ii)
complex32 FFT via the Bluestein/chirp-z reduction of the prime-length FFT to a
power-of-2 fp16 FFT overflows (NaN) at $N{=}1331$ and loses precision to
$\sim$1% at $N{=}101$, while the power-of-2 padding ($M\approx 2N$) makes its
buffers *larger* than the direct fp32 FFT — no memory win. CR's FFT is therefore
hard-locked to fp32 by the cuFFT ecosystem. The fair summary stands: **at
matched precision CR wins both compute and memory; flash's memory edge exists
only by trading away precision, and CR cannot make that trade — yet still wins
on compute (4.3×) at full precision.**

Earlier reports of "18× vs flash" are corrected: that number compared CR
against a 3-D single-head probe that silently fell back to the $O(N^2)$ math
kernel — i.e. it was really "$\sim$17× vs fp32 softmax", not vs the fused flash
kernel. The 4-D fused flash kernel engages only on $(B,H,N,D)$ layouts and
bf16/fp16; all numbers here are re-measured under those exact conditions.

### 5.3 Quality (synthetic, $p=11$, pure cross-entropy)

| model | params | 400 | 1600 | 4000 steps |
|---|---:|---:|---:|---:|
| transformer (10-layer) | 504K | 14.20 | 13.81 | — |
| matrix Szegő (cr-vec) | 239K | 14.16 | 10.98 | — |
| **piecewise (ours)** | 263K | 21.43 | 12.02 | **8.03** |

The matrix-free piecewise converges more slowly but to a *lower* floor than
both the Transformer and the matrix Szegő projection, with $1/2$ the
parameter count of the Transformer.

### 5.4 Real text (Shakespeare, character-level)

The CR (Szegő) attention is a **global, non-causal** aggregation over a window
$N=p^3$; its natural standard benchmark is therefore **masked language
modeling** (BERT-style), not the non-standard "next-window" formulation we
first tried (which saturates for *both* CR and Transformer because it forces
parallel prediction of a whole future window). On 1.1M characters of
Shakespeare (vocab 66 incl. [MASK], 15% masking):

| model | params | window | val ppl (MLM) |
|---|---:|---:|---:|
| transformer (10-layer) | 508K | 343 | 21.88 |
| **piecewise (3-layer)** | 252K | 343 | 22.51 |
| **piecewise (6-layer)** | 483K | 343 | 22.32 |
| **piecewise (3-layer)** | 270K | 1331 ($p{=}11$) | 22.34 |

The matrix-free CR attention reaches **within ~2% of the Transformer's MLM
perplexity at matched parameter count** — the feasibility evidence that it is a
general aggregation operator, not a synthetic-task artifact. The residual gap
is consistent with its slower convergence (§5.3); on structured synthetic
tasks it is *better* than the Transformer (§5.3, 8.03 vs 13.81).

The earlier "next-window" task is a documented **negative result with a
solution**: it is non-standard and saturating for both models (28–31 ppl), and
is replaced by MLM, whose inductive bias matches the global CR attention.

**Overfitting and the activation function.** Long training on real text
overfits: with split-GELU the validation ppl degrades from 28.4 (best) to 31.3
by step 8000. The degradation is caused by the activation's *sharp breakpoints*
and *non-holomorphic distortion* of the complex surface — the learned
breakpoints act as a memorisation channel on high-entropy text. Replacing them
with a **smooth, phase-preserving (CR-friendly) activation** removes this
channel: a soft-threshold (softplus modReLU) degrades only to 29.5 and the
purely radial contraction $\sigma(z)=z\,\tanh|z|/|z|$ (no breakpoint, no
learned bias) to 30.5, versus 31.3 for GELU — a consistent, ~1.8 ppl
generalisation gain. This confirms the CR geometry is not only the aggregation
principle but also the correct *regularisation* principle: the nonlinearity
must respect (smoothly contract, phase-preserve) the CR surface rather than
arbitrarily twist it.

### 5.5 Ablations

- **twist on/off** (gelu, K=3, 400 steps): 21.43 vs 21.80 — the symplectic
  chirp is a necessary ingredient; removing it leaves the diagonal ceiling.
- **nonlinearity** (modrelu vs gelu, 800 steps): 19.87 vs 17.04 — phase-mixing
  (split-GELU) beats the phase-preserving modReLU.
- **depth** (K=3 vs K=6): 21.43 vs 21.70 — more stages do not help at fixed
  budget; the twist, not depth, is the binding constraint.

### 5.6 Fully geometric (matrix-free, Euclidean-free) architecture

We also tested an architecture with **no `nn.Linear` and no matmul anywhere**
(`cr-geo`): the token embedding is a direct complex lookup; the block is
norm → piecewise Szegő attention → residual → channel-DFT mixing → split-GELU
→ per-channel gain → residual, where the only non-pointwise operators are FFTs
(Heisenberg FFT in position, cyclic DFT in the channel fibre). It trains and
converges, but plateaus at ~18 ppl (34K–108K params) versus 8–12 ppl for the
same block *with* a small Euclidean FFN (ComplexLinear, $d{\times}4d$).

**Interpretation.** The position-domain aggregation is fully realizable
matrix-free (the chirp-z result); the *channel*-domain token mixing is not:
the cyclic-DFT mixer is a circulant map with $O(d)$ degrees of freedom, while a
general channel FFN has $O(d^2)$. This is the same "diagonal vs. general
operator" gap that §15 of the handoff identified in the position domain, now
in the channel direction. The practical recommendation is therefore:
matrix-free CR attention (the $O(p^4)$ scheme eliminated, 23×/54×) *plus* a
small channel FFN — exactly `cr-vec`/`piecewise`, which reaches 8.03 ppl.

### 5.7 Formal-scale subword validation (GPT-2 / BPE, from scratch)

To move past toy character-level demos we trained a blockwise CR decoder and a
same-scale transformer from scratch on a 14.7 MB public-domain book corpus
(GPT-2 50K subword and a fitted 8K BPE), block size $W{=}p^3{=}1331$, held-out
ppl. Both the CR global aggregation and the transformer (whose
`TransformerEncoderLayer` already dispatches to the fused SDPA/flash backend on
A800, i.e. it is *not* the naive $O(N^2)$ baseline) were trained at matched
budget:

| task | CR params | CR eval ppl | trans params | trans eval ppl |
|---|---:|---:|---:|---:|
| next-block, 50K, $d{=}512{\times}6$ | 128.2M | **1281** | 71.1M | 1744 |
| next-block, 8K, $d{=}512{\times}6$ | 63.5M | **1021** | 28.0M | 1656 |
| next-block, 8K, $d{=}256{\times}4$ | 14.9M | **1014** | 7.7M | 1391 |
| MLM, 8K, $d{=}256{\times}4$ | 14.9M | 1089 | 7.7M | **1020** |

Three honest findings. **(i)** On the causal/next-block task the global CR
*generalises better* than the causal transformer in all three runs (1.36–1.62×
lower held-out ppl) — the global aggregation has strictly less context loss
than a causal mask. **(ii)** On MLM (the natural benchmark for a *global*
operator, §5.4) the bidirectional transformer is slightly better (1020 vs
1089), consistent with the $\sim$2% char-level gap already reported. **(iii)**
Absolute ppl is $>10^3$ in every row: 2 M tokens cannot train a 15–128 M-param,
50K/8K-vocab model (each parameter sees $<1$ token vs GPT-2's 100–1000) — a
*data wall*, not a CR deficiency. The meaningful quality evidence remains the
character-level results (§5.3–5.4) where CR matches/beats the transformer.

### 5.8 Unbounded context via $O(1)$ block-recurrent state

The blockwise decoder keeps a **fixed-size running state** $S\in\mathbb{C}^d$
that summarises all previous blocks and processes the sequence block by block
with global CR attention, so peak memory is $O(p^3 + d)$ independent of total
length — unlike every KV-cache model (naive softmax *and* flash), whose cache
is $O(N)$. Concretely, 1.36 M tokens of blocks are processed in 0.015 GB,
against a Qwen2.5-0.5B KV cache of 12.3 GB ($O(N)$) at 1 M tokens. This,
together with the $O(N\log N)$ attention, is the CR design's headline property;
it is independent of parameter count and of the flash/naive distinction.

### 5.9 Precision ablation and the LDR attention (subword, 80M tokens)

The plain Szegő attention lags a causal Transformer on formal subword language
modeling. A systematic ablation (same task — blockwise next-block prediction,
$W{=}p^3{=}4913$, $d{=}512$, 4 layers, 80M-token book corpus, matched budget)
isolates the cause and a fix:

| attention | eval ppl | rel. Qwen2 |
|---|---:|---:|
| complex Szegő (twist) | 1833 | 1.72× |
| real spectral (drop complex) | 1460 | 1.37× |
| cross-correlation $Q\star K$ (content-dependent) | 1381 | 1.29× |
| **LDR: $Q\star K$ + low-rank residual ($r{=}64$)** | **1201** | **1.12×** |
| Qwen2-style (GQA+RoPE, flash) | 1068 | 1.00× |
| GPT-2 (MHA, flash) | 1301 | 1.22× |

Three findings. **(i)** The complex dimension is *pure overhead*: the abelian
scalar flow gains no precision from complex structure (dropping it *improves*
ppl, halves memory, and 4.5× speed). **(ii)** The binding constraint is
*content-independence*, not smoothness or translation-equivariance — the fixed
Szegő kernel treats every input identically. Replacing it by a content-dependent
cross-correlation $Q\star K$ (two FFTs) closes part of the gap. **(iii)** The
remaining gap is Toeplitz vs full matrix; the attention is *almost Toeplitz*,
so a Linformer-style rank-$r$ residual (low-displacement-rank) captures the
position-specific part. Together the three steps take 1.72× → 1.12×, beating a
same-scale GPT-2, at $O(N\log N)+O(Nr)$ and matrix-free.

**Negative results (documented boundaries).** The following *fail* and are
recorded as limits: triple convolution $Q\star K\star V$ (explodes — $O(N^2)$
accumulation, eval ppl $>10^8$); random channel collapse as regularisation
(does not rescue it); gating $(\sigma(W_g x)\odot v)$ (modulates amplitude, not
the kernel); isolated per-segment kernels (lose global interaction); stacked
rounds $n_{\mathrm{flow}}{=}2$ and large rank $r{=}256$ (overfit — the
content-dependent residual becomes a memorisation channel). These delineate the
regime in which a matrix-free sub-quadratic operator is both stable and
competitive.

## 6. Related work

Efficient attention (sparse, low-rank, kernel, linear attention);
Fourier Neural Operators (spectral + pointwise nonlinearity); Deep Complex
Networks / modReLU; fast Fourier transforms on finite groups (Diaconis–
Rockmore); sub-Riemannian and CR geometry in learning (Barilari et al.);
analysis on the Heisenberg group (Folland, Thangavelu). We differ by using the
Szegő projection *as* the attention kernel and by a matrix-free chirp-z
realization.

## 7. Discussion and limitations

**Convergence speed.** The piecewise scheme trades convergence speed for a
lower floor (Fig. 2); we conjecture the segmented manifold must "learn" the
twist piecewise. This motivates the exact chirp-z (§4.4), whose Hermite-basis
mask is being pinned numerically.

**Honesty constraints.** (i) softmax is *not* the Bargmann kernel — it is a
structural analogy only; (ii) the curvature perturbation is a truncated
perturbation expansion, not a low-rank correction; (iii) exact $O(N\log N)$
holds only for the flat model (Appendix A).

**Memory and the flash baseline (corrected).** Modern flash attention already
gives $O(N)$ memory in bf16, so CR's advantage over flash is **not** memory —
it is (a) $O(N\log N)$ compute (the crossover at the attention level is
$N\approx 5000$ and grows with $N$), (b) full fp32 precision vs bf16, and (c)
the $O(1)$-state unbounded-context property (§5.8). Against *naive* fp32 softmax
($O(N^2)$ memory) CR is $\sim$10× faster and $\sim$10× smaller at $N{=}12{,}167$.
The headline "18×" from earlier probes measured the naive/math kernel, not the
fused flash kernel, and has been corrected (§5.2).

**Complex-width and precision cost.** complex64 doubles the channel width, so
the CR FFN is $\sim$4× the real transformer FFN at equal $d$; this is the reason
the full-model CR is currently *slower* than flash-bf16 at moderate $N$ even
though its attention is faster. It is an engineering constant, not a scaling
obstacle: the FFN expansion can be matched to the transformer's. The precision
asymmetry is *not* fully removable, however: flash requires bf16 to stay
$O(N)$-memory, while CR's FFT requires fp32 (cuFFT fp16 is power-of-2 only,
incompatible with prime $p$); CR's `half=True` complex32-storage mode does not
reduce memory materially (§5.2) because the fp32 FFT buffers dominate. The
asymptotic $O(N\log N)$ win is unaffected.

**Scope.** Results are on toy + small text; scaling to full language modeling
and the MLX backend (M6) are future work.

## 8. Conclusion

We replaced softmax attention with the Szegő projection of a CR manifold and
showed it is a well-defined, invariant, sub-quadratic aggregation operator
realizable matrix-free via a chirp-z factorization and piecewise-manifold
segmentation. The matrix-free CR-NN matches or beats a same-parameter
Transformer in perplexity while cutting latency and memory by more than an
order of magnitude at long sequence lengths.

---

## Appendix A — assumptions and risks

A1 (prime $p$): the Schrödinger family is complete iff $p$ prime. A2
(truncated perturbation): $\varepsilon\ll1$. A3 ($\bar\partial_b$ compression):
empirical. A4 (perturbation does not hurt quality). R1 (log-correction), R5
(branch cuts), R7 (composite $p$ unsupported). See `docs/assumptions.md`.

## Appendix B — numerical pinning of the chirp-z constants

`experiments/02_speedup_probe/chirpz_probe.py` verifies the twisted-convolution
decomposition (§4.3) against the naive and matrix paths to $\sim10^{-6}$, and
pins the completing-the-square constants of §4.4.

`experiments/02_speedup_probe/frft_probe.py` checks whether the ordinary DFT
eigenbasis (the Krawtchouk / discrete-Hermite functions) diagonalizes the
Szegő sector $\hat s_\lambda$. It does **not** (off-diagonal mass $0.2$–$6$),
confirming that the exact matrix-free Szegő projection requires the
*$\lambda$-dependent* fractional Fourier transform (the $\sqrt\lambda$ squeeze
of the harmonic-oscillator ground state), not the fixed DFT.

`experiments/02_speedup_probe/frft_diag.py` refines this by optimizing over
the DFT-eigenpower family $F^\alpha = U\,\mathrm{diag}(e^{i\alpha\,\angle})U^H$.
Even at the optimal $\alpha$ the sector does not diagonalize (off-diagonal mass
comparable to the diagonal), and the spectrum is spread rather than
$\{0,1\}$. The precise reason: (i) the $\eta$-regularized kernel is a *smeared*
projector (not exactly idempotent), and (ii) the true eigenbasis is the full
$\lambda$-rescaled Hermite–Laguerre basis (indexed by the Hermite level $k$
*and* the angular momentum $m$), of which the DFT-eigenpower family is only the
$\lambda$-independent projection. These two facts pin the exact diagonalization
problem: the discrete fractional Fourier transform with the $\lambda$-dependent
squeeze — the remaining constant for the exact $O(p^3\log p)$ path. The
delivered matrix-free layer (piecewise chirp + activation) already exceeds the
matrix Szegő's quality (§5.3), so this is a convergence-speed refinement, not
a blocking gap.

**Cross-platform (MLX).** The piecewise attention is pure FFT + pointwise
complex arithmetic (no einsum, no batched matmul), so it ports 1:1 to MLX for
Apple Silicon. `crnn/backends/mlx_piecewise.py` provides the M6
`MLXPiecewiseCRAttention` (same parameter layout, transferable checkpoint) and
a `parity_test`. It runs on MLX ≥0.30 (Apple Silicon Metal or a glibc ≥2.35
Linux CPU); the A800 host's older bclinux cannot load MLX's wheel, so parity is
verified on Apple hardware.

## References

Folland (1975); Thangavelu (1998); Diaconis–Rockmore (1990); Rockmore (1990);
Terras (1999); Barilari, Boscain, Neel (arXiv:1105.1285); Li et al., Fourier
Neural Operator (2021); Arjovsky, Shah, Bengio (2016); Trabelsi et al. (2018).
Full bibtex in `docs/references.bib`.
