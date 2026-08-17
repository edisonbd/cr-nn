"""Chirp-z (fractional-Fourier) matrix-free Heisenberg group convolution.

Staged numerical verification:
  A. naive group convolution      (O(p^6) ground truth)
  B. c-FFT decomposition + NAIVE twisted (a,b) convolution  (O(p^5))
  C. c-FFT decomposition + FFT-accelerated twisted convolution (O(p^3 log p))
  D. compare C against cr_group_convolve (the matrix-valued reference)

Convention: w = e^{+2 pi i / p}.
Group law (single-sided, project convention):
  (a,b,c)(a1,b1,c1) = (a+a1, b+b1, c+c1+a b1);  (a,b,c)^{-1} = (-a,-b,-c+ab).
Convolution with this law:
  (f*s)(a,b,c) = sum_{a1,b1,c1} f(a1,b1,c1) s(a-a1, b-b1, c-c1 - a1(b-b1)).

c-FFT (negative exp) + shift theorem give the per-lambda twisted convolution:
  T_lam(a,b) = sum_{a1,b1} fh(a1,b1,lam) sh(a-a1,b-b1,lam) w^{-lam a1 (b-b1)}
  (f*s)(a,b,c) = (1/p) sum_lam w^{+lam c} T_lam(a,b).

The twist w^{-lam a1(b-b1)} is the only non-commutative part.  Completing the
square (mu = lam * 2^{-1} mod p):
  w^{-lam a1 (b-b1)} = w^{-mu (a1+b-b1)^2 + mu a1^2 + mu (b-b1)^2}
turns the twisted convolution into a standard convolution with a chirp kernel,
which is diagonalized by a 2D FFT.  This file pins the exact constants down by
numerical test.
"""

from __future__ import annotations

import torch

from crnn.layers.cr_attention_backend import cr_group_convolve


def _naive_group_conv2(f, s, p):
    """Cleaner O(p^6) with explicit c index."""
    out = torch.zeros_like(f)
    for a1 in range(p):
        for b1 in range(p):
            for c1 in range(p):
                fv = f[a1, b1, c1]
                for b in range(p):
                    bb = (b - b1) % p
                    for a in range(p):
                        aa = (a - a1) % p
                        cidx = (a - a1) % p
                        # target c: c = c1 + a1*(b-b1) + c_target -> c_target = c - c1 - a1(b-b1)
                        for c in range(p):
                            ct = (c - c1 - a1 * ((b - b1) % p)) % p
                            out[a, b, c] += fv * s[aa, bb, ct]
    return out


def _cfft(f, p, sign=-1.0):
    """1D FFT along last (c) axis: sum_c f w^{sign * lam * c}."""
    # negative sign: conj(ifft(conj)) ? use direct fft with manual phase
    lam = torch.arange(p, device=f.device, dtype=torch.float32)
    c = torch.arange(p, device=f.device, dtype=torch.float32)
    W = torch.exp(torch.tensor(sign, dtype=torch.float32) * 2j * torch.pi
                  * (lam[:, None] * c[None, :]) / p)
    return torch.einsum("...c,kc->...k", f.to(torch.complex64), W)


def _cifft(f, p, sign=1.0):
    """1D inverse FFT along last axis: (1/p) sum_lam f w^{+lam c}."""
    lam = torch.arange(p, device=f.device, dtype=torch.float32)
    c = torch.arange(p, device=f.device, dtype=torch.float32)
    W = torch.exp(torch.tensor(sign, dtype=torch.float32) * 2j * torch.pi
                  * (lam[:, None] * c[None, :]) / p)
    return torch.einsum("...k,kc->...c", f.to(torch.complex64), W) / p


def _decomp_naive(f, s, p):
    """c-FFT + NAIVE twisted (a,b) convolution + c-IFFT. O(p^5)."""
    B = f.shape[0]
    w = torch.exp(torch.tensor(2j * torch.pi / p, dtype=f.dtype, device=f.device))
    fh = _cfft(f, p, sign=-1.0)                     # (B, p, p, p) over lam
    sh = _cfft(s.unsqueeze(0), p, sign=-1.0)        # (1, p, p, p)
    T = torch.zeros(B, p, p, p, dtype=torch.complex64, device=f.device)
    for lam in range(p):
        for a in range(p):
            for b in range(p):
                acc = 0j
                for a1 in range(p):
                    for b1 in range(p):
                        aa = (a - a1) % p
                        bb = (b - b1) % p
                        phase = w ** ((-lam * a1 * ((b - b1) % p)) % p)
                        acc += fh[:, a1, b1, lam] * sh[0, aa, bb, lam] * phase
                T[:, a, b, lam] = acc
    out = _cifft(T, p, sign=1.0)                    # (1/p) sum_lam w^{+lam c}
    return out


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for p in (3, 5, 7):
        f = torch.randn(1, p, p, p, dtype=torch.complex64, device=device)
        s = torch.randn(p, p, p, dtype=torch.complex64, device=device)
        # A: naive ground truth
        ref = _naive_group_conv2(f[0], s, p).unsqueeze(0)
        # B: decomposition naive
        got = _decomp_naive(f, s, p)
        err = (ref - got).abs().max().item()
        rel = err / ref.abs().max().item()
        print(f"p={p}: decomp-vs-naive  max abs={err:.3e} rel={rel:.3e} "
              f"{'OK' if rel < 1e-4 else 'MISMATCH'}")

    # D: compare matrix reference for a smaller sanity (needs B-dims align)
    for p in (3, 5):
        B = 1
        f = torch.randn(B, p, p, p, dtype=torch.complex64, device=device)
        s = torch.randn(p, p, p, dtype=torch.complex64, device=device)
        ref = cr_group_convolve(f, s, p)
        got = _decomp_naive(f, s, p)
        err = (ref - got).abs().max().item()
        rel = err / ref.abs().max().item()
        print(f"p={p}: decomp-vs-matrix  max abs={err:.3e} rel={rel:.3e} "
              f"{'OK' if rel < 1e-4 else 'MISMATCH'}")


if __name__ == "__main__":
    main()
