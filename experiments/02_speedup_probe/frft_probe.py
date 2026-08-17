"""Pin down the exact fractional-Fourier (Krawtchouk) diagonalization of the
Szegő projection's matrix sector.

Goal: find the unitary U (the discrete Hermite / Krawtchouk basis) in which the
Szegő kernel's per-lambda matrix sector s_lam is DIAGONAL, and the resulting
spectrum.  If U is a FrFT (chirp-z), the exact Szegő projection is matrix-free.

We test two hypotheses:
  H1: U = the DFT eigenbasis (ordinary DFT; Krawtchouk functions are the DFT
      eigenvectors).  I.e. s_lam is diagonalized by the DFT.
  H2: U = a fractional power F^alpha of the DFT (the lambda-dependent squeeze),
      for some alpha = alpha(lam).

The DFT on Z_p has eigenvalues in {1,-1,i,-i}; the Krawtchouk functions are its
eigenvectors, and the Hermite level k corresponds to the eigenvalue (-i)^k.
"""

from __future__ import annotations

import numpy as np
import torch

from crnn.layers.cr_attention_backend import _fft_forward
from crnn.layers.vec_cr_attention import build_szego_kernel


def dft_matrix(p):
    """Unitary DFT matrix F[m,n] = (1/sqrt p) w^{-mn}."""
    w = np.exp(-2j * np.pi / p)
    F = np.array([[w ** (m * n) for n in range(p)] for m in range(p)]) / np.sqrt(p)
    return F


def main():
    for p in (5, 7, 11):
        print(f"===== p={p} =====")
        n_cr, eta = 1, 1e-6
        S = build_szego_kernel(p, n_cr, eta, "cpu", torch.complex64)  # (p,p,p)
        # matrix sector: s_lam (p-1, p, p)
        _, mat = _fft_forward(S.unsqueeze(0), p, 1, torch.complex64, "cpu",
                              torch.exp(torch.tensor(2j * torch.pi / p,
                                                     dtype=torch.complex64)))
        mat = mat[0].numpy()  # (p-1, p, p)
        F = dft_matrix(p)
        # eigendecomposition of the DFT (eigh since F is unitary but not hermitian;
        # use eig and sort by eigenvalue phase)
        evals, evecs = np.linalg.eig(F)
        # sort by eigenvalue
        order = np.argsort(np.angle(evals))
        evals = evals[order]
        evecs = evecs[:, order]
        U = evecs  # columns = Krawtchouk-ish functions
        # H1: is U^H s_lam U diagonal?
        for lam in (1, 2, (p - 1) // 2, p - 1):
            lam = min(lam, p - 1)
            M = mat[lam - 1]  # p x p
            D = U.conj().T @ M @ U
            offdiag = np.abs(D - np.diag(np.diag(D))).max()
            diag = np.diag(D)
            print(f"  lam={lam}: DFT-eigenbasis offdiag={offdiag:.3e} "
                  f"|diag|={np.abs(diag).round(3).tolist()}")


if __name__ == "__main__":
    main()
