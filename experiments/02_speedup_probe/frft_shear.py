"""Pin the exact fractional-Fourier (chirp-z) transform that un-shears the
twisted convolution.

The twisted conv T(a,b)=sum F(a',b') S(a-a',b-b') w^{-lam a'b} reduces (after a
b-FFT) to  T_hat(a,k) = sum_{a'} F_hat(a', k+lam a') S_hat(a-a', k+lam a'),
a *shear* in the (a', k) phase space.  The fractional Fourier transform
F^alpha (chirp-z) un-shears:  F^alpha f(u) = C sum_n f(n) w^{c(n^2+u^2)-2c un},
with chirp c = c(lam).  We search c so that, in the F^alpha domain, the shear
becomes a pointwise product.  This is the exact FrFT constant for the shear.

We test directly: for each c, apply F^c to F and S along a, then check whether
the twisted conv becomes a standard (a-)convolution => pointwise in F^c domain.
"""

import numpy as np


def frft_1d(f, c, p):
    """F^c f(u) = sum_n f(n) w^{c(n^2+u^2) - 2c n u} (chirp-z fractional FT)."""
    w = np.exp(2j * np.pi / p)
    n = np.arange(p)
    out = np.zeros_like(f)
    for u in range(p):
        ph = (c * (n ** 2 + u ** 2) - 2 * c * n * u) % p
        out[u] = np.sum(f * w ** ph)
    return out


def twisted_conv_naive(F, S, lam, p):
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


def main():
    for p in (5, 7):
        rng = np.random.default_rng(0)
        lam = 1
        F = rng.standard_normal((p, p)) + 1j * rng.standard_normal((p, p))
        S = rng.standard_normal((p, p)) + 1j * rng.standard_normal((p, p))
        T = twisted_conv_naive(F, S, lam, p)
        # try: apply F^c along a (rows) to T, F, S; check T' == F' * S' pointwise?
        best = None
        for c in range(p):
            FT = np.stack([frft_1d(F[:, b], c, p) for b in range(p)], axis=1)
            ST = np.stack([frft_1d(S[:, b], c, p) for b in range(p)], axis=1)
            TT = np.stack([frft_1d(T[:, b], c, p) for b in range(p)], axis=1)
            err = np.abs(TT - FT * ST).sum()
            if best is None or err < best[0]:
                best = (err, c)
        print(f"p={p} lam={lam}: best chirp c={best[1]} err={best[0]:.3e}")


if __name__ == "__main__":
    main()
