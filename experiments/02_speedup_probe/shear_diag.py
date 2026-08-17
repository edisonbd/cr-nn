"""Find the exact symplectic (chirp-z) transform that diagonalizes the twisted
convolution — the fast O(p^3 log p) matrix-free Heisenberg convolution.

The per-lambda twisted convolution (verified in chirpz_probe.py) is
    T_lam(a,b) = sum_{a',b'} F(a',b') S(a-a', b-b') w^{-lam a' b}
(the ONLY non-abelian part).  It is a *shear* (symplectic twist), which is
diagonalized by the symplectic / fractional Fourier transform

    Phi_c f(u,v) = sum_{a,b} f(a,b) w^{a u + b v + c a b}

for a chirp c = c(lam).  We numerically solve for c(lam) and check
    Phi_c(T) = Phi_c(F) * Phi_c(S)  (pointwise).

This is the exact FrFT/chirp-z for the shear — the missing constant of §4.4.
"""

import numpy as np
import torch


def twisted_conv(F, S, lam, p):
    """Naive twisted convolution (verified correct in chirpz_probe.py)."""
    w = np.exp(2j * np.pi / p)
    T = np.zeros_like(F)
    for a in range(p):
        for b in range(p):
            acc = 0j
            for a1 in range(p):
                for b1 in range(p):
                    acc += F[a1, b1] * S[(a - a1) % p, (b - b1) % p] \
                           * w ** ((-lam * a1 * b) % p)
            T[a, b] = acc
    return T


def symp_fft(f, c, p):
    """Phi_c f(u,v) = sum_{a,b} f(a,b) w^{a u + b v + c a b}."""
    w = np.exp(2j * np.pi / p)
    a = np.arange(p)[:, None]
    b = np.arange(p)[None, :]
    u = np.arange(p)[:, None]
    v = np.arange(p)[None, :]
    # f(a,b) -> sum over a,b with phase a*u + b*v + c*a*b
    phase = (a * u[:, None] % p)  # placeholder, do properly below
    # vectorized: for each (u,v), sum_{a,b} f[a,b] w^{a u + b v + c a b}
    out = np.zeros((p, p), dtype=np.complex128)
    for uu in range(p):
        for vv in range(p):
            ph = (np.arange(p)[:, None] * uu + np.arange(p)[None, :] * vv
                  + c * np.arange(p)[:, None] * np.arange(p)[None, :]) % p
            out[uu, vv] = np.sum(f * w ** ph)
    return out


def main():
    for p in (5, 7):
        rng = np.random.default_rng(0)
        for lam in (1, 2, p - 1):
            F = rng.standard_normal((p, p)) + 1j * rng.standard_normal((p, p))
            S = rng.standard_normal((p, p)) + 1j * rng.standard_normal((p, p))
            T = twisted_conv(F, S, lam, p)
            # search c minimizing off-diagonal after symplectic FFT
            best = None
            for c in range(p):
                FT = symp_fft(F, c, p)
                ST = symp_fft(S, c, p)
                TT = symp_fft(T, c, p)
                err = np.abs(TT - FT * ST).sum()
                if best is None or err < best[0]:
                    best = (err, c)
            print(f"p={p} lam={lam}: best chirp c={best[1]} "
                  f"diag_err={best[0]:.3e}")


if __name__ == "__main__":
    main()
