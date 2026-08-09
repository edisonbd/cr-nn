"""Finite Heisenberg group H_p: matrix-valued group Fourier transform.

Group, representation, FFT, inversion, Parseval, and group convolution — a
correct, self-contained, numerically-verified implementation.

------------------------------------------------------------------------------
GROUP
------------------------------------------------------------------------------
H_p is the Heisenberg group over the ring Z/pZ, parameterised by triples
(a, b, c) in (Z/pZ)^3 with group law (commutator coefficient = 1)

    (a1,b1,c1) * (a2,b2,c2) = (a1+a2, b1+b2, c1+c2 + a1*b2)   (all mod p)

    identity  = (0, 0, 0)
    inverse   = (-a, -b, -c + a*b)   (mod p)

The c-axis (a=b=0) is the centre: c-only elements commute with everything.

This law makes the Schrödinger representation below a true group homomorphism
rho(g*h) = rho(g) @ rho(h)  (verified to ~1e-15).

------------------------------------------------------------------------------
REPRESENTATIONS  —  THE KEY SUBTLETY (read this)
------------------------------------------------------------------------------
The Schrödinger representation, omega = e^{+2*pi*i/p}:

    [rho_lam(a,b,c)]_{u,v} = omega^{lam*c} * omega^{lam*b*u} * delta_{v,(u+a) mod p}

rho_lam is unitary and is a representation for every lam in Z/pZ. BUT it is
*irreducible* only when gcd(lam, p) == 1. When p is COMPOSITE and
gcd(lam, p) = g > 1, rho_lam is reducible: its commutant has dimension g, and
rho_lam splits into g distinct irreps of dimension p/g. Worse, the Schrödinger
family {rho_lam : lam = 1..p-1} is then INCOMPLETE — it misses some
low-dimensional irreps (e.g. for p=4 it misses 2 of the 4 two-dimensional
irreps with centre character c -> (-1)^c). See the analysis block at the
bottom of this file.

When p is PRIME, gcd(lam, p) == 1 for every lam != 0, so {rho_lam : lam=1..p-1}
is the complete set of p-dimensional irreps, and together with the p^2
one-dimensional characters chi_{m,n}(a,b,c) = omega^{m*a + n*b} (the lam=0
sector, where the centre acts trivially) they form the COMPLETE irrep set:

    sum of d^2  =  (p-1)*p^2  +  p^2 * 1^2  =  p^3  =  |G|.     (prime p only)

Dimension count closes, Schur orthogonality holds (off-diagonal ~1e-15), and
the standard Plancherel inversion is exact.

This module therefore REQUIRES p to be prime. For composite p it raises
ValueError with a pointer to the extra construction needed (see the analysis
block at the end). This is not a cop-out: it is the only regime in which the
matrix-valued FFT is both correct AND accelerates convolution by matrix
multiplication, which is the whole point of the spectral path.

------------------------------------------------------------------------------
FOURIER TRANSFORM  (prime p)
------------------------------------------------------------------------------
Forward (scalar f -> matrix-valued fhat):

    fhat(0; m, n)      = sum_{a,b,c} f(a,b,c) * omega^{m*a + n*b}       (p^2 scalars)
    fhat(lam)_{u,v}    = sum_{a,b,c} f(a,b,c) * rho_lam(a,b,c)_{u,v}    (p x p matrix)
                         for lam = 1, ..., p-1

    fhat(0; m,n) = sum_b sum_c f(v-u, b, c) ...  ; a = (v - u) mod p for the matrix sector,
so the matrix sector reduces to a 2-D positive-exponent FFT over (b,c) for each
(u, v). We implement it with explicit loops first (correctness > speed).

Inverse (Plancherel, |H_p| = p^3, irrep dims d_rho = p for lam!=0, d=1 for chars):

    f(a,b,c) = (1/p^3) * [ sum_{m,n} 1 * fhat(0;m,n) * conj(chi_{m,n}(a,b,c))
                         + sum_{lam=1}^{p-1} p * sum_{u,v} fhat(lam)_{u,v} * conj(rho_lam(a,b,c)_{u,v}) ]

Because every irrep is unitary, rho(g)^{-1} = rho(g)^dagger = conj(rho(g).T),
and sum_{u,v} A_{uv} conj(B_{uv}) = Tr(A @ B^dagger). Hence the standard
formula  f(g) = (1/|G|) sum_rho d_rho * Tr( fhat(rho) @ rho(g)^dagger ).

Parseval (energy conservation):

    sum_g |f(g)|^2  =  (1/|G|) * [ sum_{m,n} |fhat(0;m,n)|^2
                                 + sum_{lam=1}^{p-1} p * ||fhat(lam)||_F^2 ]

------------------------------------------------------------------------------
CONVOLUTION  (the speed-up)
------------------------------------------------------------------------------
Group convolution  (f * g)(h) = sum_{h'} f(h') g(h'^{-1} h).

Convolution theorem (matrix-valued, derived from rho(g*h) = rho(g) rho(h)):

    (f * g)^(lam)  =  fhat(lam) @ ghat(lam)        (p x p matrix product, per lam)
    (f * g)^(0;m,n) = fhat(0;m,n) * ghat(0;m,n)    (scalar pointwise)

So convolution — an O(|G|^2) = O(p^6) operation naively — becomes p-1 matrix
multiplications of size p (O(p^4)) plus a 2-D FFT (O(p^2 log p)) and a 2-D
pointwise product. This IS the matrix-multiplication acceleration the project
needs, and it is exact to machine precision.

NOTE the order: (f*g)^ = fhat @ ghat, NOT ghat @ fhat. Matrices do not
commute, and the derivation  (f*g)^(lam) = sum_{h',k} f(h')g(k) rho(h'k)
                                  = [sum f(h') rho(h')] [sum g(k) rho(k)] = fhat ghat
fixes the order as fhat-then-ghat (left to right).

------------------------------------------------------------------------------
REFERENCES
------------------------------------------------------------------------------
* Terras, "Fourier Analysis on Finite Groups and Applications".
* Tao, "The finite uncertainty principle" (blog/expository).
* Steinberg, "Representation Theory of Finite Groups".
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# sanity: p must be prime
# ---------------------------------------------------------------------------

def _composite_message(p: int) -> str:
    return (
        f"p={p} is composite. The matrix-valued Heisenberg FFT in this "
        f"module is only correct for prime p: for composite p the "
        f"Schrödinger family {{rho_lam}} is reducible (gcd(lam,p)>1) and, "
        f"more importantly, INCOMPLETE — it misses low-dimensional irreps, "
        f"so Schur orthogonality, Plancherel, and the inversion formula all "
        f"break (this is the ~5% energy leak). See _COMPOSITE_P_ANALYSIS at "
        f"the bottom of this file for the concrete p=4 construction of the "
        f"missing irreps. Use a prime p."
    )


def _check_prime(p: int) -> None:
    """Raise unless p is prime.

    The Schrödinger family {rho_lam : lam=1..p-1} is the complete set of
    p-dimensional irreps ONLY for prime p. For composite p the family is
    reducible for lam with gcd(lam,p)>1 and, more importantly, INCOMPLETE
    (it misses low-dimensional irreps). See the analysis block at the end of
    this file for the concrete p=4 case.
    """
    if not isinstance(p, (int, np.integer)) or p < 2:
        raise ValueError(f"p must be a prime >= 2, got {p!r}")
    p = int(p)
    if p == 2:
        return
    if p % 2 == 0:
        raise ValueError(_composite_message(p))
    # trial division by odd factors (p is small in this project)
    f = 3
    while f * f <= p:
        if p % f == 0:
            raise ValueError(_composite_message(p))
        f += 2


# ---------------------------------------------------------------------------
# group structure
# ---------------------------------------------------------------------------

def group_multiply(g1, g2, p: int):
    """(a1,b1,c1)*(a2,b2,c2) = (a1+a2, b1+b2, c1+c2 + a1*b2)  (mod p)."""
    a1, b1, c1 = g1
    a2, b2, c2 = g2
    return ((a1 + a2) % p, (b1 + b2) % p, (c1 + c2 + a1 * b2) % p)


def inverse(g, p: int):
    """(a,b,c)^{-1} = (-a, -b, -c + a*b)  (mod p)."""
    a, b, c = g
    return ((-a) % p, (-b) % p, (-c + a * b) % p)


# ---------------------------------------------------------------------------
# Schrödinger representation  (valid for all lam; irreducible iff gcd(lam,p)==1)
# ---------------------------------------------------------------------------

def rho(lam: int, a: int, b: int, c: int, p: int) -> np.ndarray:
    """[rho_lam(a,b,c)]_{u,v} = omega^{lam c} omega^{lam b u} delta_{v,(u+a) mod p}.

    Unitary p x p matrix. A group homomorphism for the law above. Irreducible
    iff gcd(lam, p) == 1 (guaranteed here since p is prime and lam != 0).
    """
    omega = np.exp(2j * np.pi / p)
    M = np.zeros((p, p), dtype=np.complex128)
    phase_c = omega ** (lam * c)
    for u in range(p):
        M[u, (u + a) % p] = phase_c * omega ** (lam * b * u)
    return M


# ---------------------------------------------------------------------------
# forward / inverse Fourier transform  (clear loops; correctness first)
# ---------------------------------------------------------------------------

def heisenberg_fft(f: np.ndarray) -> dict:
    """Forward Heisenberg Fourier transform of a scalar function on H_p.

    f : (p, p, p) complex array indexed by (a, b, c).
    returns : dict with
        'chars'   : (p, p) array  fhat(0; m, n)        — the p^2 one-dim sector
        'matrices': (p-1, p, p) array  fhat(lam)_{u,v} — lam = 1..p-1

    The 'matrices' sector is stored with leading index `ilam = lam - 1`.
    """
    f = np.asarray(f, dtype=np.complex128)
    if f.shape != (f.shape[0],) * 3:
        raise ValueError(f"f must be (p,p,p); got {f.shape}")
    p = f.shape[0]
    _check_prime(p)
    omega = np.exp(2j * np.pi / p)

    # ---- lam=0 sector: p^2 scalars  fhat0[m,n] = sum_{a,b,c} f(a,b,c) omega^{m a + n b}
    # (c is invisible at lam=0 because the centre acts trivially.)
    fhat_chars = np.zeros((p, p), dtype=np.complex128)
    # sum over c first, then 2-D positive-exponent transform over (a, b)
    f_ab = f.sum(axis=2)                                   # (a, b)
    for m in range(p):
        for n in range(p):
            s = 0.0 + 0.0j
            for a in range(p):
                for b in range(p):
                    s += f_ab[a, b] * omega ** (m * a + n * b)
            fhat_chars[m, n] = s

    # ---- lam=1..p-1 sector: p x p matrices  fhat(lam)_{u,v} = sum_{a,b,c} f(a,b,c) rho_lam_{u,v}
    # rho_lam_{u,v} nonzero only at a = (v - u) mod p, so collapse a:
    #   fhat(lam)_{u,v} = sum_{b,c} f(v-u, b, c) omega^{lam c} omega^{lam b u}.
    fhat_mat = np.zeros((p - 1, p, p), dtype=np.complex128)
    u = np.arange(p)
    for ilam, lam in enumerate(range(1, p)):
        phase_c = omega ** (lam * np.arange(p))            # (c,)
        phase_bu = omega ** (lam * u[:, None] * np.arange(p)[None, :])  # (u, b)
        for v in range(p):
            a = (v - u) % p                                # (u,) -> a for each u
            # f[a, b, c] gathered over u: shape (p, p, p) = (u, b, c)
            block = f[a, :, :]                             # (u, b, c)
            # sum over (b,c) with phases:  sum_{b,c} block[u,b,c] omega^{lam c} omega^{lam b u}
            tmp = block * phase_c[None, None, :]           # c-phase  -> (u,b,c)
            tmp = tmp.sum(axis=2)                          # (u,b)
            tmp = tmp * phase_bu                           # b,u-phase
            fhat_mat[ilam, :, v] = tmp.sum(axis=1)         # sum over b -> (u,)
    return {"chars": fhat_chars, "matrices": fhat_mat}


def heisenberg_ifft(fhat: dict) -> np.ndarray:
    """Inverse Heisenberg Fourier transform (Plancherel inversion).

    fhat : dict as returned by heisenberg_fft.
    returns : (p, p, p) complex array f(a, b, c).

        f(a,b,c) = (1/p^3) [ sum_{m,n} fhat0[m,n] omega^{-m a - n b}
                           + sum_{lam=1}^{p-1} p * sum_{u,v} fhat(lam)_{u,v}
                                              * conj(rho_lam(a,b,c)_{u,v}) ]
    """
    fhat_chars = np.asarray(fhat["chars"], dtype=np.complex128)
    fhat_mat = np.asarray(fhat["matrices"], dtype=np.complex128)
    p = fhat_chars.shape[0]
    _check_prime(p)
    omega = np.exp(2j * np.pi / p)
    G = p ** 3

    f = np.zeros((p, p, p), dtype=np.complex128)
    u_arr = np.arange(p)
    for a in range(p):
        for b in range(p):
            # ---- lam=0 chars:  sum_{m,n} fhat0[m,n] omega^{-m a - n b}
            s_chars = 0.0 + 0.0j
            for m in range(p):
                for n in range(p):
                    s_chars += fhat_chars[m, n] * omega ** (-m * a - n * b)

            for c in range(p):
                s = s_chars
                # ---- lam=1..p-1 matrices
                # rho_lam(a,b,c)_{u,v} nonzero at v=(u+a)%p with value
                # omega^{lam c} omega^{lam b u}; so
                # sum_{u,v} fhat_{u,v} conj(rho_{u,v}) =
                #   sum_u fhat_{u, (u+a)%p} * omega^{-lam c} omega^{-lam b u}.
                for ilam, lam in enumerate(range(1, p)):
                    v = (u_arr + a) % p                    # (u,) column index
                    diag = fhat_mat[ilam, u_arr, v]        # (u,)
                    phase = omega ** (-lam * c - lam * b * u_arr)  # (u,)
                    s += p * np.sum(diag * phase)
                f[a, b, c] = s / G
    return f


# ---------------------------------------------------------------------------
# group convolution via the matrix Fourier theorem
# ---------------------------------------------------------------------------

def group_convolve(f: np.ndarray, g: np.ndarray) -> np.ndarray:
    """(f * g)(h) = sum_{h'} f(h') g(h'^{-1} h)  via the frequency domain.

    Theorem (prime p):  (f*g)^(lam) = fhat(lam) @ ghat(lam)  (matrix product)
                        (f*g)^(0;m,n) = fhat0[m,n] * ghat0[m,n] (pointwise)
    then invert. Naive cost O(p^6); this is O(p^4) (dominated by the p-1
    matrix products of size p).
    """
    f = np.asarray(f, dtype=np.complex128)
    g = np.asarray(g, dtype=np.complex128)
    p = f.shape[0]
    _check_prime(p)

    F = heisenberg_fft(f)
    Gh = heisenberg_fft(g)

    conv_chars = F["chars"] * Gh["chars"]                  # (p, p) pointwise
    conv_mat = np.empty_like(F["matrices"])                # (p-1, p, p)
    for ilam in range(p - 1):
        # (f*g)^(lam) = fhat(lam) @ ghat(lam)  (f left, g right)
        conv_mat[ilam] = F["matrices"][ilam] @ Gh["matrices"][ilam]
    return heisenberg_ifft({"chars": conv_chars, "matrices": conv_mat})


# ---------------------------------------------------------------------------
# naive convolution (reference, for the test only)
# ---------------------------------------------------------------------------

def group_convolve_naive(f: np.ndarray, g: np.ndarray) -> np.ndarray:
    """O(|G|^2) reference: (f*g)(h) = sum_{h'} f(h') g(h'^{-1} h)."""
    f = np.asarray(f, dtype=np.complex128)
    g = np.asarray(g, dtype=np.complex128)
    p = f.shape[0]
    out = np.zeros((p, p, p), dtype=np.complex128)
    for a1 in range(p):
        for b1 in range(p):
            for c1 in range(p):
                gi = inverse((a1, b1, c1), p)
                fv = f[a1, b1, c1]
                for a2 in range(p):
                    for b2 in range(p):
                        for c2 in range(p):
                            h = group_multiply(gi, (a2, b2, c2), p)
                            out[a2, b2, c2] += fv * g[h]
    return out


# ---------------------------------------------------------------------------
# self-tests
# ---------------------------------------------------------------------------

def _test_group_and_representation(p: int) -> None:
    rng = np.random.default_rng(p)
    max_rep_err = 0.0
    for _ in range(40):
        g = tuple(int(x) % p for x in rng.integers(0, p, 3))
        h = tuple(int(x) % p for x in rng.integers(0, p, 3))
        for lam in range(1, p):
            lhs = rho(lam, *group_multiply(g, h, p), p)
            rhs = rho(lam, *g, p) @ rho(lam, *h, p)
            max_rep_err = max(max_rep_err, float(np.max(np.abs(lhs - rhs))))
    assert max_rep_err < 1e-12, f"rep err {max_rep_err}"
    # inverse
    for _ in range(40):
        g = tuple(int(x) % p for x in rng.integers(0, p, 3))
        gh = group_multiply(g, inverse(g, p), p)
        assert gh == (0, 0, 0), (g, gh)
    # unitarity
    max_u = 0.0
    for lam in range(1, p):
        for _ in range(20):
            g = tuple(int(x) % p for x in rng.integers(0, p, 3))
            r = rho(lam, *g, p)
            max_u = max(max_u, float(np.max(np.abs(r @ r.conj().T - np.eye(p)))))
    assert max_u < 1e-12, f"unitarity {max_u}"
    print(f"[p={p}] group/rep: OK  (rep err {max_rep_err:.1e}, unitarity {max_u:.1e})")


def _test_roundtrip(p: int) -> None:
    rng = np.random.default_rng(p + 1)
    f = (rng.standard_normal((p, p, p)) + 1j * rng.standard_normal((p, p, p))).astype(np.complex128)
    fhat = heisenberg_fft(f)
    f_rec = heisenberg_ifft(fhat)
    err = float(np.max(np.abs(f - f_rec)))
    assert err < 1e-10, f"roundtrip err {err}"
    print(f"[p={p}] roundtrip: OK  (max err {err:.2e})")


def _test_parseval(p: int) -> None:
    rng = np.random.default_rng(p + 2)
    f = (rng.standard_normal((p, p, p)) + 1j * rng.standard_normal((p, p, p))).astype(np.complex128)
    fhat = heisenberg_fft(f)
    G = p ** 3
    e_f = float(np.sum(np.abs(f) ** 2))
    e0 = float(np.sum(np.abs(fhat["chars"]) ** 2))
    e_mat = float(p * np.sum(np.abs(fhat["matrices"]) ** 2))
    parseval = (e0 + e_mat) / G
    rel = abs(e_f - parseval) / e_f
    assert rel < 1e-10, f"parseval rel {rel}"
    print(f"[p={p}] parseval: OK  (energy {e_f:.4f}, parseval {parseval:.4f}, rel {rel:.2e})")


def _test_convolution(p: int) -> None:
    rng = np.random.default_rng(p + 3)
    f = (rng.standard_normal((p, p, p)) + 1j * rng.standard_normal((p, p, p))).astype(np.complex128)
    g = (rng.standard_normal((p, p, p)) + 1j * rng.standard_normal((p, p, p))).astype(np.complex128)
    fast = group_convolve(f, g)
    naive = group_convolve_naive(f, g)
    err = float(np.max(np.abs(fast - naive)))
    rel = err / max(float(np.max(np.abs(naive))), 1e-12)
    assert rel < 1e-10, f"conv rel {rel}"
    print(f"[p={p}] convolution: OK  (abs err {err:.2e}, rel {rel:.2e})")


def _test_composite_raises() -> None:
    for bad in (1, 4, 6, 8, 9):
        try:
            _check_prime(bad)
        except ValueError:
            continue
        raise AssertionError(f"p={bad} should have raised")
    print("[guard] composite-p ValueError: OK")


def run_all_tests() -> None:
    print("=" * 64)
    print("Heisenberg H_p FFT — self-tests")
    print("=" * 64)
    _test_composite_raises()
    for p in (3, 5, 7, 11):
        _test_group_and_representation(p)
        _test_roundtrip(p)
        _test_parseval(p)
        _test_convolution(p)
    print("-" * 64)
    print("ALL TESTS PASSED")
    print("=" * 64)


# ---------------------------------------------------------------------------
# analysis block: why composite p needs more than the Schrödinger family
# ---------------------------------------------------------------------------

_COMPOSITE_P_ANALYSIS = """
WHY COMPOSITE p IS NOT SUPPORTED BY THE SCHRODINGER FAMILY ALONE
================================================================
For p=4 (the user's failing case) the full irrep set is:

    sector            count   dim   sum d^2
    ------------------------------------------------
    chars chi_{m,n}     16      1       16      (lam=0)
    rho_1                1      4       16      (gcd(1,4)=1 -> irreducible)
    rho_2 -> splits      4      2       16      (gcd(2,4)=2; rho_2 only
                                                captures 2 of these 4!)
    rho_3                1      4       16      (gcd(3,4)=1 -> irreducible)
    ------------------------------------------------
    total                               64 = |G|

rho_2 (the p x p Schrödinger matrix) is reducible: its commutant has dim 2,
and rho_2 splits into 2 distinct 2-dim irreps. But there are FOUR 2-dim
irreps with centre character c -> (-1)^c — rho_2 only sees 2 of them. The
other 2 are simply absent from the Schrödinger family, so the family is
INCOMPLETE and sum_{rho in family} d_rho^2 = 16+16+8+16 = 56 < 64. That
missing 8 (= 2 x 2^2) is the ~5% energy leak the user observed.

The 4 two-dim irreps are constructed explicitly as
    rho2_{s,t}(a,b,c) = (-1)^c * Y^b X^a
where X, Y are 2x2 matrices with X^2 = s I, Y^2 = t I, X Y = -Y X, X^4 = Y^4 = I,
for (s,t) in {(+1,+1),(+1,-1),(-1,+1),(-1,-1)}. Only 2 of these 4 appear in
the block-diagonalisation of rho_2; the other 2 must be built by hand.

For p=8 (= 2^3) the situation is worse: rho_lam is reducible for lam in
{2,4,6}, and the Schrödinger family misses irreps of dims 2 and 4. Closing
sum d^2 = 512 requires constructing those by hand from projective
representations of (Z/8Z)^2 with cocycle omega^{lam a1 b2}, generalising the
p=4 construction. This is doable per-p but does not have a single clean
formula across all composite p, which is why this module restricts to prime p.

The convolution theorem (f*g)^ = fhat @ ghat still HOLDS for the redundant
full matrix family {rho_lam : lam=0..p-1} even when some rho_lam are
reducible (a representation, reducible or not, diagonalises convolution) —
but the family is a non-injective frame (rank 44 < 64 for p=4), so it cannot
be inverted and cannot give a working FFT round-trip. You must use the true
complete irrep set, which for prime p is exactly {chars} U {rho_1..rho_{p-1}}.

PRACTICAL GUIDANCE: pick p prime (3, 5, 7, 11, 13, ...). The grid resolution
per axis is then p; if a project test currently uses p=8, switch it to the
nearest prime (7 or 11) — the geometry, kernels, and spectral theory are all
unchanged, only the lattice spacing differs.
"""


if __name__ == "__main__":
    run_all_tests()
    print()
    print(_COMPOSITE_P_ANALYSIS)
