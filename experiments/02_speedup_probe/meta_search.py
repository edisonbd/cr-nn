"""Exhaustive metaplectic (chirp-z) search for the diagonalizing transform of
the Heisenberg twisted convolution.

The general metaplectic transform on the (a,b) plane is a product of per-axis
pre-chirp / FFT / post-chirp factors.  We brute-force the 4 chirp constants
(c1..c4) and check whether any makes the twisted convolution pointwise:

    Phi f = chirp_m . FFT_b . chirp_b . FFT_a . chirp_a  (applied per axis)

If a clean diagonalization exists (O(p^3 log p) exact), the residual will drop
to ~0 for the right constants; otherwise the twisted convolution is NOT
scalar-diagonalizable (the non-abelian matrix product is fundamental).
"""

import itertools
import numpy as np


def fft1d(x, sign, p):
    w = np.exp(sign * 2j * np.pi / p)
    n = np.arange(p)
    return np.array([np.sum(x * w ** (k * n % p)) for k in range(p)])


def twisted_conv(F, S, lam, p):
    w = np.exp(2j * np.pi / p)
    T = np.zeros_like(F)
    for a in range(p):
        for b in range(p):
            acc = 0j
            for a1 in range(p):
                for b1 in range(p):
                    acc += F[a1, b1] * S[(a - a1) % p, (b - b1) % p] * w ** ((-lam * a1 * b) % p)
            T[a, b] = acc
    return T


def apply_phi(f, c_a, c_m, c_b, c_k, p):
    """Phi: pre-chirp a (c_a a^2) -> FFT_a -> chirp m (c_m m^2) ->
    pre-chirp b (c_b b^2) -> FFT_b -> chirp k (c_k k^2)."""
    w = np.exp(2j * np.pi / p)
    a = np.arange(p)
    # pre-chirp in a (rows)
    g = f * (w ** (c_a * a * a % p))[:, None]
    # FFT along a (axis 0, sign -1)
    g = np.stack([fft1d(g[:, b], -1, p) for b in range(p)], axis=1)
    # chirp in m (rows)
    m = np.arange(p)
    g = g * (w ** (c_m * m * m % p))[:, None]
    # pre-chirp in b (cols)
    g = g * (w ** (c_b * a * a % p))[None, :]
    # FFT along b (axis 1)
    g = np.stack([fft1d(g[m_, :], -1, p) for m_ in range(p)], axis=0)
    # chirp in k (cols)
    g = g * (w ** (c_k * a * a % p))[None, :]
    return g


def main():
    p = 5
    rng = np.random.default_rng(0)
    lam = 1
    F = rng.standard_normal((p, p)) + 1j * rng.standard_normal((p, p))
    S = rng.standard_normal((p, p)) + 1j * rng.standard_normal((p, p))
    T = twisted_conv(F, S, lam, p)
    best = None
    for c_a, c_m, c_b, c_k in itertools.product(range(p), repeat=4):
        FT = apply_phi(F, c_a, c_m, c_b, c_k, p)
        ST = apply_phi(S, c_a, c_m, c_b, c_k, p)
        TT = apply_phi(T, c_a, c_m, c_b, c_k, p)
        err = np.abs(TT - FT * ST).sum()
        if best is None or err < best[0]:
            best = (err, (c_a, c_m, c_b, c_k))
    print(f"p={p} lam={lam}: best chirps (a,m,b,k)={best[1]} err={best[0]:.3e}")


if __name__ == "__main__":
    main()
