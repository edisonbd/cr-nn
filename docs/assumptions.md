# Assumptions and risk register

This file explicitly lists all assumptions, open propositions, and known risks
that the CR-NN project depends on. Each entry carries a stability grade and a
verification method. Cite the corresponding IDs when writing the paper.

## Stability grades

- **Solid**: proven in the classical literature, usable directly, no caveat.
- **Conditional**: holds under specific conditions, which must be enforced in the
  implementation or flagged as deviations.
- **Open**: a proposition the project must verify or argue itself — part of the
  research contribution.

---

## Core assumptions

### A1: Sequence embedding on a discrete lattice ($p$ must be prime) [Solid, v0.2 correction]
Sequence tokens embed into the $p\times p\times p$ lattice of the finite
Heisenberg group $H_p$. Group convolution on $H_p$ is accelerated by the
matrix-valued Fourier transform: naive $O(p^6)$ drops to $O(p^4)$.
- Basis: Diaconis & Rockmore 1990; Maslen; Terras; Tao "finite uncertainty
  principle".
- **Key constraint**: $p$ must be prime. For composite $p$ the Schrödinger
  representation family is incomplete ($\sum d_\rho^2<|G|$) and Plancherel
  fails. For prime $p$ it is numerically verified: round-trip error ~1e-15,
  Parseval ~1e-15, convolution vs naive ~1e-14 (p=3,5,7,11).
- Caveat: the matrix-valued ($p\times p$) representation is more complex than a
  scalar FFT, with a larger constant factor; the actual constant and crossover
  point must be measured (M3).
- The group law uses the one-sided form $c=c_1+c_2+a_1b_2$ (not the continuous
  symmetric form $2(x_1y_2-y_1x_2)$); the two are isomorphic.
- Verification: M2 unit tests (prime $p$) on round-trip/Parseval/convolution;
  passed.

### A2: Truncated perturbation expansion [Conditional]
The Szegő projection on a curved CR manifold is truncated as
$S_\text{curved}\approx S_\text{flat}+\sum_{j=1}^M\varepsilon^j L_j[S_\text{flat}]$,
with error $O(\varepsilon^{M+1})$.
- Basis: Barilari arXiv:1105.1285 (sub-Riemannian heat-kernel perturbation,
  verified); Boutet de Monvel–Sjöstrand 1976 (Szegő-kernel FIO asymptotics).
- Condition: $\varepsilon\ll 1$ (default $|\varepsilon|<0.1$), small-time /
  high-frequency regime.
- **Must not be described as a "low-rank correction"**: the curvature
  perturbation is generally full-rank, with no literature support.
- Verification: M5 curvature ablation, sweeping $M$ and $\varepsilon$.

### A3: $\bar\partial_b$ regularization compresses information [Open]
The term $\mu\|\bar\partial_b\,\mathrm{out}\|^2$ in the training loss pushes the
representation toward CR functions, equivalent to compressing information into
the holomorphic (complex) subspace.
- Basis: CR-geometry definition ($\bar\partial_b f=0$ is holomorphic), but
  "the regularizer actually concentrates information" is an empirical claim.
- Verification: M4 measures the feature entropy / spectral energy distribution
  after training, with and without the $\bar\partial_b$ regularizer.

### A4: Truncated perturbation does not harm downstream quality [Open]
The error of the approximate fast transform under truncated perturbation does not
significantly harm downstream-task quality.
- This is the central open proposition on which the whole hybrid scheme rests.
- Verification: M3 (speed probe confirms numerical stability) + M4 (toy-sequence
  ppl compared against the flat model).
- Stop-loss: if A4 fails, fall back to the pure flat model ($M=0$), abandon the
  curvature perturbation, and let quality rely on the learnable embedding
  positions in $\mathbb{H}^n$.

---

## Known risk register

### R1: Logarithmic correction term [High risk]
The sub-Riemannian heat-kernel expansion develops a $\log\rho$ term at a certain
order (absent in the elliptic case), which may land on the truncation order $M$.
- Impact: treating it as a power term may be numerically unstable.
- Mitigation: flag the log term separately in the implementation, with a switch;
  compare with/without in the M5 ablation.
- Literature: Barilari et al. arXiv:1606.01159 (bi-Heisenberg heat-kernel
  asymptotics).

### R2: Eigenvalue multiplicity splitting [Medium risk]
Under small curvature the spectrum is continuous, but the Hermite degeneracy
(multiplicity $n$) can split. If the fast transform relies on the multiplicity
structure, splitting breaks the diagonalization.
- Mitigation: tolerate splitting in training (do not hard-code the multiplicity),
  or model the splitting explicitly in the perturbation.
- Verification: M5 checks whether the trained spectrum is still approximately
  diagonal.

### R3: Global vs local frequency bands [Medium risk]
The perturbation expansion is local (small-time / high-frequency). Low-frequency
large-scale behaviour deviates more from the flat model.
- Impact: if the network operates in the low-frequency band, the curvature
  approximation fails.
- Mitigation: M4 analyses the network's actual operating band (spectral energy
  distribution); if needed, restrict the curvature term to high-frequency layers.

### R4: Discretization breaks the group structure [Low–medium risk]
Discretizing continuous $\mathbb{H}^n$ to $H_p$ can break the group-convolution
structure if sampled badly, invalidating the FFT speedup.
- Mitigation: strictly use the algebraic structure of $H_p$ (upper-triangular
  matrix group), not an approximate lattice from continuous sampling.
- Verification: M2 unit tests confirm the FFT group convolution matches the naive
  implementation numerically.

### R5: Complex-power branch-cut consistency [Engineering risk]
The Korányi/Szegő kernels contain a complex power $(|z|^2-it)^{-(n+1)}$; the two
backends' `pow`/`log` branch conventions must agree.
- Mitigation: explicitly split as $\exp(-(n+1)\cdot\mathrm{Log}(\cdot))$ with the
  principal-branch Log; cross-check both backends in unit tests.

### R6: MLX complex64 coverage is incomplete [Engineering risk]
MLX has only `complex64` (no `complex128`), and some uncommon complex ops may be
missing.
- Mitigation: validate the math in Torch first (M2); on the M6 MLX port, check
  op by op and hand-write real/imag splits where needed.

### R7: Composite $p$ unsupported [Resolved, on record]
For composite $p$ the Schrödinger representation family is incomplete and
Plancherel/inversion fail (found and fixed in v0.2).
- Resolution: force $p$ prime; choose neighbouring primes for the lattice
  resolution.
- If $p=2^k$ is ever needed (e.g. to align with hardware tile sizes), the
  projective irreps must be constructed by hand — no uniform formula; listed as
  future work.

---

## Narrative compliance (for paper writing)

When writing the paper/docs, obey the following wording constraints to avoid
objections from geometric-analysis reviewers:

1. ✅ May say: "the CR Szegő projection as a geometric *replacement* for
   attention" — replacement, not reinterpretation.
2. ✅ May say: "a truncated perturbation expansion, the curvature terms being
   finite-order symbol corrections" — supported by Barilari et al.
3. ❌ May not say: "softmax = a discrete truncation of the Bargmann coherent-state
   kernel" — no literature support; structural analogy only.
4. ❌ May not say: "the curvature term is a low-rank correction" — the curvature
   perturbation is generally full-rank.
5. ❌ May not say: "exact $O(N\log N)$ fast transform" (on the curved model) — only
   the flat model is exact; the curved one is approximate.

---

## Version

- v0.1 (2026-08-09): initial version, aligned with math.md v0.1.
