"""Truncated curvature perturbation for the Szego projection (M5).

Implements docs/math.md section 5:

    S_curved f ~= S_flat f + sum_{j=1..M} eps_j * L_j[S_flat f]

with L_j = Delta_b^j (the truncated perturbation expansion; Barilari
arXiv:1105.1285).  Each eps_j is a learnable amplitude, initialised at 0 so
the model starts flat, and softly constrained to |eps| <= eps_max via
eps_eff = eps_max * tanh(eps).

The R1 log-correction term (a possible log rho factor at specific orders of
the expansion, NOT included in the main perturbation) is exposed as a
separate switch so the ablation can compare with/without it.  Its
engineering form here is a pointwise multiplier log(1 + rho^4) with its own
learnable amplitude -- explicitly a placeholder, flagged R1.
"""

from __future__ import annotations

import numpy as np
import torch

from ..geometry.operators import Delta_b


def delta_b_powers(f, M: int, axis_t: int = -1):
    """Apply Delta_b^j to f for j = 1..M.  Returns a list [L_1 f, ..., L_M f].

    f : (..., p, p, p) complex, spatial domain (x, y, t axes at -3, -2, -1).
    """
    results = []
    acc = f
    for _ in range(M):
        acc = Delta_b(acc, axis_t=axis_t)
        results.append(acc)
    return results


def log_correction_factor(p: int, eta: float, device, dtype) -> torch.Tensor:
    """Pointwise log(1 + rho^4) on the fftshifted p x p x p grid.

    rho^4 = |z|^4 + t^2 is the squared Koranyi distance (math.md 4.1), so
    log(1 + rho^4) is a smooth, growing-in-radius local weight.  This is the
    engineering placeholder for the R1 log term; it is NOT part of the main
    perturbation and is gated by its own switch/amplitude.
    """
    coords = np.fft.fftfreq(p, d=1.0) * p
    xx = coords.reshape(p, 1, 1)
    yy = coords.reshape(1, p, 1)
    tt = coords.reshape(1, 1, p)
    rho4 = (xx ** 2 + yy ** 2) ** 2 + tt ** 2
    factor = np.log1p(rho4).astype(np.float32)
    return torch.from_numpy(factor).to(device=device, dtype=dtype)


def apply_perturbation(out, eps_eff, M: int, axis_t: int = -1,
                       normalize: bool = True):
    """out <- out + sum_j eps_eff[j] * Delta_b^j out.

    eps_eff : (M,) tensor of effective (soft-constrained) amplitudes.
    normalize : if True, RMS-normalise each Delta_b^j term to unit energy.
        Raw Delta_b^j has spectral norm growing like ((2k+n)|lam|)^j, so
        higher orders blow up training (observed: M=2 collapse, M=3
        explosion in the first M5 sweep).  Normalising makes L_j a unit
        direction and lets eps_eff control the amplitude.
    """
    if M <= 0 or eps_eff is None:
        return out
    acc = out
    for j in range(M):
        acc = Delta_b(acc, axis_t=axis_t)
        if normalize:
            rms = acc.abs().square().mean().sqrt()
            acc = acc / (rms + 1e-8)
        out = out + eps_eff[j] * acc
    return out


def apply_log_correction(out, eps_log_eff, p, eta, axis_t: int = -1):
    """out <- out + eps_log * log(1 + rho^4) * out  (R1 placeholder)."""
    if eps_log_eff is None:
        return out
    factor = log_correction_factor(p, eta, out.device, out.dtype)
    return out + eps_log_eff * factor * out
