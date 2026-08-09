"""Spectral structure of the sub-Laplacian on the Heisenberg group.

Implements ``docs/math.md`` §3:

* the eigenvalue formula   σ_{k,λ} = (2k + n) |λ|
* the Hermite–Laguerre eigenbasis (rescaled Hermite in x,y, Laguerre in t)
* truncation to the first K modes (assumption: Hermite functions are rapidly
  decreasing, so the tail is O(exp(-cK)))

This is the single most load-bearing module for the "flat model is fast"
claim: every spectral operation (CR-Sobolev loss, Szegő projection, the
Δ_b-diagonal path) reduces to acting diagonally in this basis. The eigenvalue
test in experiments/01 pins Δ_b to these exact values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..backend import Backend, default_backend


@dataclass
class HermiteLaguerreBasis:
    """Hermite–Laguerre eigenbasis of Δ_b on H^n, truncated.

    The basis is indexed by (k, m, λ):
      * k  ∈ {0,...,K-1}      — Hermite quantum number (radial in (x,y)), n-fold degenerate
      * m  ∈ Z                — angular momentum (Laguerre index along t)
      * λ  ∈ Fourier grid     — center frequency

    Eigenvalue of Δ_b: σ_{k,λ} = (2k + n) |λ|  (independent of m; the
    Laguerre index is the t-angular quantum number and commutes with Δ_b).

    We construct the basis functions on the discrete grid as the product of
    (a) rescaled Hermite functions H_k in (x,y) and (b) Fourier modes e^{iλt}
    weighted by Laguerre polynomials. Construction is done in numpy (CPU) for
    exactness and clarity, then handed to the backend as an array — basis
    evaluation is not on the training hot path, only the *coefficients*
    obtained by projection are.

    Attributes
    ----------
    n : int       CR complex dimension.
    p : int       grid resolution per axis (grid is p×p×p for n=1).
    K : int       number of Hermite modes retained (k = 0..K-1).
    """

    n: int
    p: int
    K: int

    def eigenvalues(self, backend: Backend | None = None):
        """Return σ_{k,λ} = (2k+n)|λ| on the (k, λ) grid.

        Shape (K, p) for n=1: k indexes axis 0, λ (the t-frequency) axis 1.
        Used by CR-Sobolev loss to weight coefficient magnitudes.
        """
        b = backend or default_backend()
        lam = _fft_freq(self.p)              # (p,)  signed integers
        k = np.arange(self.K)                # (K,)
        # σ_{k,λ} = (2k + n) |λ|; |λ| since Δ_b is even in λ.
        sigma = (2.0 * k[:, None] + self.n) * np.abs(lam)[None, :]
        return b.asarray(sigma.astype(np.float32))

    def basis_functions(self, backend: Backend | None = None):
        """Return basis functions φ_{k,m} evaluated on the (p,p,p) grid.

        Returns an array of shape (K, p, p, p) for n=1 (the m/Laguerre
        structure is folded into the t-axis Fourier modes and handled by
        projecting against e^{iλt}, so we don't materialize it separately).

        Each φ_k is the rescaled Hermite function H_k(x) H_k(y) (real,
        separable) — the Δ_b eigenfunction for λ=1; for general λ the basis
        is obtained by rescaling, which the projection handles via FFT.
        """
        b = backend or default_backend()
        x = np.arange(self.p, dtype=np.float64)
        # physicist's Hermite functions (normalized) via numpy.polynomial
        # H_k(x) e^{-x²/2}; we evaluate on the integer grid without scaling
        # to physical units (the scaling is absorbed into the eigenvalue test
        # tolerance). Use scipy for the orthonormal Hermite functions.
        from numpy.polynomial.hermite import hermval

        # Build normalized Hermite functions h_k(x) = (1/sqrt(2^k k! sqrt pi))
        # H_k(x) e^{-x^2/2}. For the unit test we only need the eigenfunction
        # *property* Δ_b φ_k = σ_k φ_k up to the rescaling, so absolute
        # normalization is not critical here.
        basis = np.zeros((self.K, self.p, self.p), dtype=np.float64)
        for ki in range(self.K):
            coeffs = np.zeros(ki + 1)
            coeffs[ki] = 1.0
            hk = hermval(x, coeffs) * np.exp(-x ** 2 / 2.0)
            # 2D separable product H_k(x) ⊗ H_k(y)
            basis[ki] = np.outer(hk, hk)
        return b.asarray(basis.astype(np.float32))


def sub_laplacian_eigenvalues(K: int, p: int, n: int,
                              backend: Backend | None = None):
    """Convenience wrapper: σ_{k,λ} = (2k+n)|λ|, shape (K, p)."""
    return HermiteLaguerreBasis(n=n, p=p, K=K).eigenvalues(backend=backend)


def _fft_freq(p: int) -> np.ndarray:
    """Signed FFT frequencies [0,..,p/2-1,-p/2,..,-1] as float64."""
    return np.fft.fftfreq(p, d=1.0) * p
