"""Szegő projection: the CR-NN attention replacement.

Implements math.md §4.2 (flat) and §5 (curved, truncated perturbation):

    flat:     Π_flat f = S_flat * f              (group convolution, O(N log N))
    curved:   Π_curved f ≈ Π_flat f + Σ_{j=1}^M ε^j L_j[Π_flat f]

This is the single object that *replaces* softmax attention (math.md §8). It
is a kernel-regression-style aggregation, but the kernel is the Szegő kernel
of the CR manifold rather than an inner-product softmax. We do not claim
softmax is a special case (assumptions.md — that equivalence is unsupported);
we claim this is a different, geometrically-motivated aggregator.

The flat term is exact and FFT-accelerated. The perturbation terms L_j are
parameterized transport operators (math.md §5.2): each is a learnable linear
combo of low-order symbols built from Δ_b and the coordinates, applied to
the flat output. Their coefficients are the "curvature" parameters ε_j.
"""

from __future__ import annotations

from ..backend import Backend, default_backend
from ..geometry.operators import Delta_b, szego_kernel_flat
from .heisenberg_fft import group_convolve


def szego_projection_flat(f, n: int, eta: float = 1e-6,
                          backend: Backend | None = None):
    """Flat Szegő projection Π_flat f = S_flat * f.

    ``f`` shape (..., p, p, p) over (x, y, t), complex. The kernel S_flat is
    evaluated on the grid and the convolution uses :func:`group_convolve`,
    giving O(N log N).
    """
    b = backend or default_backend()
    p = f.shape[-1]
    # Build S_flat on the grid. The kernel is a function of g = (z,t); on the
    # discrete grid we evaluate it at the group-element coordinates (centered
    # so that the identity is at index 0, i.e. use fftshifted coordinates).
    S = _eval_szego_kernel_on_grid(p, n, eta=eta, backend=b)
    return group_convolve(f, S, backend=b)


def szego_projection_curved(f, n: int, M: int, eps,
                            eta: float = 1e-6, backend: Backend | None = None):
    """Curved Szegő projection via truncated perturbation (math.md §5).

    Π_curved f ≈ Π_flat f + Σ_{j=1}^M ε_j · L_j[Π_flat f]

    ``eps`` is a sequence of length M (the learnable curvature amplitudes).
    Each L_j is a parameterized transport operator: we take L_j = c_j · Δ_b^j
    applied to the flat output, where c_j are fixed symbolic constants (here
    1.0 — the *learnable* part is ε_j itself; the symbol shape is fixed by
    the perturbation theory). This is the engineering realization of
    "learning absorbs the truncation tail" (assumption A2).

    The log correction term (R1) is NOT included here; it has its own flag in
    the ablation experiment (experiments/04).
    """
    b = backend or default_backend()
    flat = szego_projection_flat(f, n, eta=eta, backend=b)
    if M == 0 or not eps:
        return flat
    out = flat
    acc = flat
    for j in range(M):
        # L_j [flat] = Δ_b^j (flat): apply Δ_b once per order, accumulating.
        acc = Delta_b(acc, axis_t=-1, backend=b)
        out = out + eps[j] * acc
    return out


# ---------------------------------------------------------------------------
# kernel evaluation on the grid
# ---------------------------------------------------------------------------

def _eval_szego_kernel_on_grid(p, n, eta, backend):
    """Evaluate S_flat(z,t) = (|z|² - i t)^{-(n+1)} on the p×p×p grid.

    Uses fftshifted coordinates so the identity element (0,0,0) is at the
    center — required for the convolution theorem to hold with the kernel
    as given (group convolution centers the kernel at the identity).
    """
    b = backend
    import numpy as np
    # fftshifted coordinates: [0,1,..,p/2-1, -p/2,..,-1] for each axis.
    coords1d = np.fft.fftfreq(p, d=1.0) * p          # signed, length p
    x = coords1d.reshape(p, 1, 1)
    y = coords1d.reshape(1, p, 1)
    t = coords1d.reshape(1, 1, p)
    z = x + 1j * y                                    # (p,p,p) complex, n=1
    absz2 = (x ** 2 + y ** 2)
    w = absz2 - 1j * t
    # regularize origin
    w = w + eta
    S = w ** (-(n + 1))
    return b.asarray(S.astype(np.complex64))
