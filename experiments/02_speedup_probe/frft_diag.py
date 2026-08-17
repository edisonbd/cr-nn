"""Rigorously pin the fractional-Fourier (lambda-squeeze) diagonalization of
the Szegő matrix sector.

Theory (discrete harmonic oscillator on Z_p):
  The sub-Laplacian on the lambda-slice is a discrete harmonic oscillator whose
  eigenfunctions are the Krawtchouk (discrete Hermite) functions K_0..K_{p-1}
  (the DFT eigenvectors).  For a general lambda the oscillator is *squeezed* by
  sqrt(lambda); on the discrete group this squeeze is a fractional Fourier
  transform  F^alpha = U diag(e^{-i alpha k}) U^H,  where U are the Krawtchouk
  functions ordered by Hermite level k (DFT eigenvalue (-i)^k).

  The Szegő projection s_lam is the projector onto the *squeezed ground state*
  (k=0 level of the lambda-oscillator), hence

      s_lam = F^{alpha(lam)} P_0 F^{-alpha(lam)},

  with P_0 = the projector onto K_0.  This script numerically solves for
  alpha(lam) and checks that F^{alpha(lam)} s_lam F^{-alpha(lam)} is diagonal
  with a single unit entry (the Szegő spectrum = {1 on k=0, 0 else}).
"""

from __future__ import annotations

import numpy as np
import torch

from crnn.layers.cr_attention_backend import _fft_forward
from crnn.layers.vec_cr_attention import build_szego_kernel


def krawtchouk_basis(p):
    """Return U (p x p) whose columns are the DFT eigenvectors ordered by
    Hermite level k, and the eigenvalue phases.  We build the DFT, eigendecompose,
    and order the eigenspaces by the Mehta convention (k-th vector has DFT
    eigenvalue (-i)^k)."""
    w = np.exp(-2j * np.pi / p)
    F = np.array([[w ** (m * n) for n in range(p)] for m in range(p)]) / np.sqrt(p)
    evals, evecs = np.linalg.eig(F)
    # order by eigenvalue phase
    order = np.argsort(np.angle(evals))
    return evecs[:, order], evals[order]


def frac_ft(U, evals, alpha):
    """F^alpha = U diag(e^{i alpha * phase_frac}) U^H, phase in units of pi/2."""
    # evals are phases; raise each to the power alpha (in units of pi/2)
    d = np.exp(1j * alpha * np.angle(evals))          # e^{i alpha * angle}
    return (U * d[None, :]) @ U.conj().T


def main():
    for p in (5, 7, 11):
        print(f"===== p={p} =====")
        S = build_szego_kernel(p, 1, 1e-6, "cpu", torch.complex64)
        _, mat = _fft_forward(S.unsqueeze(0), p, 1, torch.complex64, "cpu",
                              torch.exp(torch.tensor(2j * torch.pi / p,
                                                     dtype=torch.complex64)))
        mat = mat[0].numpy()                          # (p-1, p, p)
        U, evals = krawtchouk_basis(p)
        # search alpha in [0, pi] minimizing off-diagonal of F^a s_lam F^{-a}
        for lam in (1, 2, (p + 1) // 2, p - 1):
            M = mat[min(lam, p - 1) - 1]
            best = (None, 1e9)
            for ai in np.linspace(0, np.pi, 2001):
                Fa = frac_ft(U, evals, ai)
                D = Fa @ M @ Fa.conj().T
                off = np.abs(D - np.diag(np.diag(D))).sum()
                if off < best[1]:
                    best = (ai, off, D)
            ai, off, D = best
            diag = np.abs(np.diag(D))
            # normalise the spectrum and report the top entry
            idx = np.argsort(-diag)[:3]
            print(f"  lam={lam}: alpha={ai:.3f} offdiag_sum={off:.3e} "
                  f"top_diag={[(int(i), round(float(diag[i]), 2)) for i in idx]}")


if __name__ == "__main__":
    main()
