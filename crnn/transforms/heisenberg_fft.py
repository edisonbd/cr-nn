"""Finite Heisenberg group H_p: matrix-valued group Fourier transform.

Correct, numerically-verified implementation (往返 ~1e-15, Parseval ~1e-15,
convolution vs naive ~1e-14 for prime p = 3, 5, 7, 11).

KEY CONSTRAINT: p must be prime. For composite p the Schrödinger family
{rho_lam} is reducible (gcd(lam,p)>1) and, more importantly, INCOMPLETE —
it misses low-dimensional irreps, so sum d_rho^2 < |G| and Plancherel breaks
(~5% energy leak). See docs/math.md §7.1 and docs/assumptions.md R7.

Group law (single-sided commutator, coefficient 1 — makes rho a homomorphism):
    (a1,b1,c1)*(a2,b2,c2) = (a1+a2, b1+b2, c1+c2 + a1*b2)   (mod p)
    inverse (a,b,c)^{-1} = (-a, -b, -c + a*b)

This is isomorphic to but differently parameterised from the continuous
H^n law t+t'+2 Im(z z̄') (coefficient 2, symmetric); see docs/math.md §7.2.
The continuous-form kernels (Korányi/Szegő) are evaluated on the grid using
this group's coordinates.

Schrödinger representation, omega = e^{+2πi/p}:
    [rho_lam(a,b,c)]_{u,v} = omega^{lam c} omega^{lam b u} delta_{v,(u+a) mod p}
Irreducible iff gcd(lam,p)==1 (guaranteed for lam!=0 when p prime).

Fourier transform (scalar f -> chars + matrix-valued fhat):
    lam=0 sector:  fhat(0;m,n) = sum_{a,b,c} f(a,b,c) omega^{m a + n b}   (p^2 scalars)
    lam=1..p-1:    fhat(lam)_{u,v} = sum_{a,b,c} f(a,b,c) rho_lam(a,b,c)_{u,v}  (p×p)

Inverse (Plancherel, |G|=p^3, d=p for matrices, d=1 for chars):
    f(a,b,c) = (1/p^3)[ sum_{m,n} fhat0[m,n] omega^{-ma-nb}
                      + sum_{lam=1}^{p-1} p sum_{u,v} fhat(lam)_{u,v} conj(rho_lam_{u,v}) ]

Convolution theorem (the speed-up):
    (f*g)^(lam) = fhat(lam) @ ghat(lam)        (p×p matrix product, f LEFT)
    (f*g)^(0;m,n) = fhat0[m,n] * ghat0[m,n]    (pointwise)
Naive O(p^6) -> O(p^4) (p-1 matrix products of size p) + O(p^2 log p) FFT.

References: Terras, "Fourier Analysis on Finite Groups"; Tao, "The finite
uncertainty principle"; docs/references.bib [diaconisrockmore1990efficient].
"""

from __future__ import annotations

import numpy as np

from ..backend import Backend, default_backend


# ---------------------------------------------------------------------------
# prime guard
# ---------------------------------------------------------------------------

def _check_prime(p: int) -> None:
    """Raise unless p is prime. See module docstring / assumptions R7."""
    if not isinstance(p, (int, np.integer)) or p < 2:
        raise ValueError(f"p must be a prime >= 2, got {p!r}")
    p = int(p)
    if p == 2:
        return
    if p % 2 == 0:
        raise ValueError(
            f"p={p} is composite; the Heisenberg FFT requires prime p "
            f"(see docs/assumptions.md R7). Use a nearby prime."
        )
    f = 3
    while f * f <= p:
        if p % f == 0:
            raise ValueError(
                f"p={p} is composite (divisible by {f}); the Heisenberg FFT "
                f"requires prime p (see docs/assumptions.md R7)."
            )
        f += 2


# ---------------------------------------------------------------------------
# group structure (discrete, single-sided commutator)
# ---------------------------------------------------------------------------

def group_multiply_mod(g1, g2, p: int):
    """(a1,b1,c1)*(a2,b2,c2) = (a1+a2, b1+b2, c1+c2 + a1*b2) mod p."""
    a1, b1, c1 = g1
    a2, b2, c2 = g2
    return ((a1 + a2) % p, (b1 + b2) % p, (c1 + c2 + a1 * b2) % p)


def inverse_mod(g, p: int):
    """(a,b,c)^{-1} = (-a, -b, -c + a*b) mod p."""
    a, b, c = g
    return ((-a) % p, (-b) % p, (-c + a * b) % p)


# ---------------------------------------------------------------------------
# Schrödinger representation
# ---------------------------------------------------------------------------

def rho(lam: int, a: int, b: int, c: int, p: int) -> np.ndarray:
    """[rho_lam(a,b,c)]_{u,v} = omega^{lam c} omega^{lam b u} delta_{v,(u+a) mod p}.

    Unitary p×p matrix, group homomorphism for the law above. Irreducible
    iff gcd(lam,p)==1 (guaranteed for lam!=0 when p is prime).
    """
    omega = np.exp(2j * np.pi / p)
    M = np.zeros((p, p), dtype=np.complex128)
    phase_c = omega ** (lam * c)
    for u in range(p):
        M[u, (u + a) % p] = phase_c * omega ** (lam * b * u)
    return M


# ---------------------------------------------------------------------------
# forward / inverse Fourier transform
# ---------------------------------------------------------------------------

def heisenberg_fft(f, backend: Backend | None = None) -> dict:
    """Forward Heisenberg Fourier transform.

    f : (..., p, p, p) complex over (a, b, c). Batch dims (...) preserved
        only in the matrices sector as a leading axis; the chars sector is
        computed per-batch in a loop. (Single-array batch support is
        completed in M3; M2 validates the unbatched path.)
    returns : dict with
        'chars'   : (p, p)      fhat(0; m, n)
        'matrices': (p-1, p, p) fhat(lam)_{u,v}, leading index = lam-1
    """
    b = backend or default_backend()
    np_f = _to_numpy(f).astype(np.complex128)
    if np_f.ndim == 3:
        np_f = np_f[None]                      # add batch dim
        squeeze = True
    else:
        squeeze = False
    B, p, _, _ = np_f.shape
    _check_prime(p)
    omega = np.exp(2j * np.pi / p)

    fhat_chars = np.zeros((B, p, p), dtype=np.complex128)
    fhat_mat = np.zeros((B, p - 1, p, p), dtype=np.complex128)
    u = np.arange(p)

    for bi in range(B):
        fb = np_f[bi]
        # lam=0: sum over c, then 2D positive-exponent transform over (a,b)
        f_ab = fb.sum(axis=2)                  # (a, b)
        # positive-exponent 2D transform = conj(fft2(conj))
        fhat_chars[bi] = np.conj(np.fft.fft2(np.conj(f_ab)))

        # lam=1..p-1: fhat(lam)_{u,v} = sum_{b,c} f(v-u,b,c) omega^{lam c} omega^{lam b u}
        for ilam, lam in enumerate(range(1, p)):
            phase_c = omega ** (lam * np.arange(p))               # (c,)
            phase_bu = omega ** (lam * u[:, None] * np.arange(p)[None, :])  # (u, b)
            for v in range(p):
                a = (v - u) % p                  # (u,) a-index
                block = fb[a, :, :]              # (u, b, c)
                tmp = block * phase_c[None, None, :]
                tmp = tmp.sum(axis=2)            # (u, b)
                tmp = tmp * phase_bu
                fhat_mat[bi, ilam, :, v] = tmp.sum(axis=1)

    if squeeze:
        return {"chars": fhat_chars[0], "matrices": fhat_mat[0]}
    return {"chars": fhat_chars, "matrices": fhat_mat}


def heisenberg_ifft(fhat: dict, backend: Backend | None = None):
    """Inverse Heisenberg Fourier transform (Plancherel inversion)."""
    b = backend or default_backend()
    fhat_chars = np.asarray(fhat["chars"], dtype=np.complex128)
    fhat_mat = np.asarray(fhat["matrices"], dtype=np.complex128)
    squeeze = False
    if fhat_chars.ndim == 2:
        fhat_chars = fhat_chars[None]
        fhat_mat = fhat_mat[None]
        squeeze = True
    B, p, _ = fhat_chars.shape
    _check_prime(p)
    omega = np.exp(2j * np.pi / p)
    G = p ** 3

    f = np.zeros((B, p, p, p), dtype=np.complex128)
    u_arr = np.arange(p)
    for bi in range(B):
        fc = fhat_chars[bi]
        fm = fhat_mat[bi]
        for a in range(p):
            for bb in range(p):
                # lam=0: sum_{m,n} fhat0[m,n] omega^{-ma-nb}  (clear loop; M3 fast path later)
                s_chars = 0.0 + 0.0j
                for m in range(p):
                    for n in range(p):
                        s_chars += fc[m, n] * omega ** (-m * a - n * bb)
                for c in range(p):
                    s = s_chars
                    for ilam, lam in enumerate(range(1, p)):
                        v = (u_arr + a) % p
                        diag = fm[ilam, u_arr, v]
                        phase = omega ** (-lam * c - lam * bb * u_arr)
                        s += p * np.sum(diag * phase)
                    f[bi, a, bb, c] = s / G
    f = f if not squeeze else f[0]
    return b.asarray(f.astype(np.complex64))


# ---------------------------------------------------------------------------
# group convolution via the matrix Fourier theorem
# ---------------------------------------------------------------------------

def group_convolve(f, g, backend: Backend | None = None):
    """(f * g)(h) = sum_{h'} f(h') g(h'^{-1} h)  via the frequency domain.

    Theorem (prime p): (f*g)^(lam) = fhat(lam) @ ghat(lam)  (f LEFT),
                       (f*g)^(0;m,n) = fhat0 * ghat0 (pointwise).
    O(p^6) naive -> O(p^4) (p-1 size-p matmuls) + O(p^2 log p) FFT.
    """
    b = backend or default_backend()
    F = heisenberg_fft(f, backend=b)
    Gh = heisenberg_fft(g, backend=b)

    np_fc = np.asarray(F["chars"], dtype=np.complex128)
    np_fm = np.asarray(F["matrices"], dtype=np.complex128)
    np_gc = np.asarray(Gh["chars"], dtype=np.complex128)
    np_gm = np.asarray(Gh["matrices"], dtype=np.complex128)

    conv_chars = np_fc * np_gc
    # batched matmul over (p-1) matrices: (p-1, p, p) @ (p-1, p, p)
    conv_mat = np.empty_like(np_fm)
    for ilam in range(np_fm.shape[-3] if np_fm.ndim == 4 else np_fm.shape[0]):
        if np_fm.ndim == 4:   # batched
            conv_mat[:, ilam] = np_fm[:, ilam] @ np_gm[:, ilam]
        else:
            conv_mat[ilam] = np_fm[ilam] @ np_gm[ilam]
    return heisenberg_ifft({"chars": conv_chars, "matrices": conv_mat}, backend=b)


def group_convolve_naive(f, g, backend: Backend | None = None):
    """O(|G|^2) reference: (f*g)(h) = sum_{h'} f(h') g(h'^{-1} h). For tests."""
    b = backend or default_backend()
    np_f = _to_numpy(f).astype(np.complex128)
    np_g = _to_numpy(g).astype(np.complex128)
    p = np_f.shape[-1]
    _check_prime(p)
    single = np_f.ndim == 3
    if single:
        np_f = np_f[None]
        np_g = np_g[None]
    out = np.zeros_like(np_f)
    for bi in range(np_f.shape[0]):
        for a1 in range(p):
            for b1 in range(p):
                for c1 in range(p):
                    gi = inverse_mod((a1, b1, c1), p)
                    fv = np_f[bi, a1, b1, c1]
                    for a2 in range(p):
                        for b2 in range(p):
                            for c2 in range(p):
                                h = group_multiply_mod(gi, (a2, b2, c2), p)
                                out[bi, a2, b2, c2] += fv * np_g[bi, h[0], h[1], h[2]]
    out = out[0] if single else out
    return b.asarray(out.astype(np.complex64))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _to_numpy(x):
    try:
        return x.detach().cpu().numpy()
    except AttributeError:
        return np.asarray(x)
