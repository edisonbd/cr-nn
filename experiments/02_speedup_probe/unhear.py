"""Pin the chirp constant that un-shears the Heisenberg twisted convolution.

Derivation: after an a-FFT the twisted conv becomes
    T_hat(m,b) = sum_{b'} F_hat(m+lam b, b') S_hat(m, b-b')       (verified)
This is a b-convolution with the FREQUENCY sheared by m -> m+lam b.  A shear is
un-done by a chirp multiply e^{i a m^2} in the m variable (the metaplectic
"shear" generator).  We numerically solve for the chirp constant a = a(lam)
such that, after chirp-in-m + b-FFT, the operation is pointwise.
"""

import numpy as np


def fft1d(x, sign=1, p=None):
    p = p or len(x)
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


def main():
    p = 5
    rng = np.random.default_rng(0)
    lam = 1
    F = rng.standard_normal((p, p)) + 1j * rng.standard_normal((p, p))
    S = rng.standard_normal((p, p)) + 1j * rng.standard_normal((p, p))
    T = twisted_conv(F, S, lam, p)
    w = np.exp(2j * np.pi / p)

    # a-FFT along a (axis 0) with sign -1
    Fh = np.stack([fft1d(F[:, b], -1, p) for b in range(p)], axis=1)  # (m, b)
    Sh = np.stack([fft1d(S[:, b], -1, p) for b in range(p)], axis=1)  # (m, b)
    Th = np.stack([fft1d(T[:, b], -1, p) for b in range(p)], axis=1)  # (m, b)

    best = None
    for a in range(p):
        # chirp in m: multiply row m by w^{a m^2}
        m = np.arange(p)
        chirp = w ** (a * m * m % p)
        Fhc = Fh * chirp[:, None]
        Shc = Sh * chirp[:, None]
        Thc = Th * chirp[:, None]
        # b-FFT along b (axis 1)
        Fhcb = np.stack([fft1d(Fhc[m, :], -1, p) for m in range(p)], axis=0)
        Shcb = np.stack([fft1d(Shc[m, :], -1, p) for m in range(p)], axis=0)
        Thcb = np.stack([fft1d(Thc[m, :], -1, p) for m in range(p)], axis=0)
        err = np.abs(Thcb - Fhcb * Shcb).sum()
        if best is None or err < best[0]:
            best = (err, a)
    print(f"p={p} lam={lam}: best chirp a={best[1]} err={best[0]:.3e}")


if __name__ == "__main__":
    main()
