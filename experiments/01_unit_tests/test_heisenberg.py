"""Unit tests for CR-NN operators: numerical correctness against analytic facts.

Each test pins one statement from docs/math.md. These are the M2 acceptance
criterion: if they pass, the spectral implementation is faithful to the
geometry. The Heisenberg FFT tests use PRIME p (see docs/assumptions.md R7:
composite p breaks Plancherel because the Schrödinger family is incomplete).

Run:
    python -m pytest experiments/01_unit_tests -q
"""

import numpy as np
import pytest

from crnn.backend import default_backend
from crnn.geometry.heisenberg import (
    group_multiply,
    inverse,
    koranyi_norm,
)
from crnn.geometry.operators import koranyi_kernel, szego_kernel_flat
from crnn.geometry.spectrum import sub_laplacian_eigenvalues
from crnn.transforms.heisenberg_fft import (
    group_convolve,
    group_convolve_naive,
    heisenberg_fft,
    heisenberg_ifft,
    _check_prime,
)


b = default_backend()
P = 5           # PRIME (see R7); small enough for fast naive convolution
N = 1           # CR complex dim; M2 validates n=1


# ---------------------------------------------------------------------------
# math.md §1: continuous group structure (analytic, p-independent)
# ---------------------------------------------------------------------------

class TestGroupStructure:
    """math.md §1.1–1.3: continuous group law, inverse, Korányi norm."""

    def test_identity(self):
        z = b.asarray([[1.0 + 2.0j, 3.0 - 1.0j]], dtype="complex64")
        t = b.asarray([0.5])
        z0 = b.asarray([[0.0 + 0.0j, 0.0 + 0.0j]], dtype="complex64")
        t0 = b.asarray([0.0])
        z2, t2 = group_multiply((z, t), (z0, t0), backend=b)
        assert _close(z2, z)
        assert _close(t2, t)

    def test_inverse(self):
        z = b.asarray([[1.0 + 2.0j]], dtype="complex64")
        t = b.asarray([0.5])
        zi, ti = inverse((z, t))
        z2, t2 = group_multiply((z, t), (zi, ti), backend=b)
        assert _close(z2, b.asarray([[0.0 + 0.0j]], dtype="complex64"))
        assert _close(t2, b.asarray([0.0]))

    def test_nonabelian(self):
        # The z-component is commutative; the t-component carries the
        # non-abelian structure via 2 Im(z·z̄'). math.md §1.1.
        z1 = b.asarray([[1.0 + 0.0j]], dtype="complex64")
        t1 = b.asarray([0.0])
        z2 = b.asarray([[0.0 + 1.0j]], dtype="complex64")
        t2 = b.asarray([0.0])
        _, t_gh = group_multiply((z1, t1), (z2, t2), backend=b)
        _, t_hg = group_multiply((z2, t2), (z1, t1), backend=b)
        assert _close(t_gh, b.asarray([-2.0]))
        assert _close(t_hg, b.asarray([2.0]))
        assert not _close(t_gh, t_hg, atol=1e-5)

    def test_koranyi_norm_homogeneity(self):
        # ρ(δ_r g) = r ρ(g) under (z,t)→(rz, r²t). math.md §1.3.
        z = b.asarray([[3.0 + 4.0j]], dtype="complex64")
        t = b.asarray([6.0])
        rho0 = koranyi_norm(z, t, backend=b)
        r = 2.0
        rho_r = koranyi_norm(z * r, t * (r ** 2), backend=b)
        assert _close(rho_r, r * rho0, rtol=1e-5)

    def test_koranyi_norm_at_origin(self):
        z = b.asarray([[0.0 + 0.0j]], dtype="complex64")
        t = b.asarray([0.0])
        assert _close(koranyi_norm(z, t, backend=b), b.asarray([0.0]))


# ---------------------------------------------------------------------------
# math.md §4: closed-form kernels
# ---------------------------------------------------------------------------

class TestKernels:
    """math.md §4.1–4.2: Korányi & Szegő kernels, closed form."""

    def test_koranyi_kernel_singularity_order(self):
        # Γ(g) ~ ρ^{-Q}, Q=2n+2. Γ at radius ρ vs 2ρ differs by factor 2^Q.
        n = N
        Q = 2 * n + 2
        z1 = b.asarray([[1.0 + 0.0j]], dtype="complex64")
        t1 = b.asarray([0.0])
        z2 = z1 * 2.0
        g1 = koranyi_kernel(z1, t1, n=n, eta=0.0, backend=b)
        g2 = koranyi_kernel(z2, t1, n=n, eta=0.0, backend=b)
        ratio = float(_to_numpy(g1)[0]) / float(_to_numpy(g2)[0])
        assert abs(ratio - 2 ** Q) < 1e-3 * (2 ** Q)

    def test_szego_kernel_homogeneity(self):
        # S(g) ~ (|z|²-it)^{-(n+1)}, homogeneous degree -2(n+1) under δ_r.
        n = N
        z1 = b.asarray([[1.0 + 0.0j]], dtype="complex64")
        t1 = b.asarray([1.0])
        s1 = szego_kernel_flat(z1, t1, n=n, eta=0.0, backend=b)
        r = 2.0
        s2 = szego_kernel_flat(z1 * r, t1 * (r ** 2), n=n, eta=0.0, backend=b)
        ratio = _to_numpy(s1)[0] / _to_numpy(s2)[0]
        assert abs(abs(ratio) - 2 ** (2 * (n + 1))) < 1e-3 * 2 ** (2 * (n + 1))


# ---------------------------------------------------------------------------
# math.md §3: spectrum
# ---------------------------------------------------------------------------

class TestSpectrum:
    """math.md §3.2: σ_{k,λ} = (2k+n)|λ|."""

    def test_eigenvalue_formula(self):
        K, p, n = 5, P, N
        sigma = _to_numpy(sub_laplacian_eigenvalues(K, p, n, backend=b))
        lam = np.fft.fftfreq(p) * p
        for k in range(K):
            for li in range(p):
                expected = (2 * k + n) * abs(lam[li])
                assert abs(sigma[k, li] - expected) < 1e-4, (k, li)

    def test_eigenvalues_nonnegative(self):
        sigma = _to_numpy(sub_laplacian_eigenvalues(4, P, N, backend=b))
        assert np.all(sigma >= -1e-6)


# ---------------------------------------------------------------------------
# math.md §7 / assumptions A1, R7: Heisenberg FFT (prime p)
# ---------------------------------------------------------------------------

class TestHeisenbergFFT:
    """math.md §7.2: matrix-valued Heisenberg FFT, Plancherel, convolution."""

    def test_prime_guard(self):
        for bad in (1, 4, 6, 8, 9):
            with pytest.raises(ValueError):
                _check_prime(bad)
        for good in (2, 3, 5, 7, 11, 13):
            _check_prime(good)  # no raise

    def test_roundtrip(self):
        # f -> fhat -> f  recovers f to machine precision (prime p).
        f = _rand_complex((P, P, P))
        fhat = heisenberg_fft(f, backend=b)
        f_rec = heisenberg_ifft(fhat, backend=b)
        assert _close(f_rec, f, atol=1e-8)

    def test_parseval(self):
        # sum|f|^2 = (1/|G|)[ sum|fhat0|^2 + p sum_lam ||fhat_lam||_F^2 ]
        f = _rand_complex((P, P, P))
        fhat = heisenberg_fft(f, backend=b)
        G = P ** 3
        e_f = float(np.sum(np.abs(_to_numpy(f)) ** 2))
        chars = np.asarray(fhat["chars"])
        mats = np.asarray(fhat["matrices"])
        e0 = float(np.sum(np.abs(chars) ** 2))
        e_mat = float(P * np.sum(np.abs(mats) ** 2))
        rel = abs(e_f - (e0 + e_mat) / G) / e_f
        # complex64 input loses precision vs the complex128 transform; 1e-6
        # is the right bar (the _solution_fft.py pure-complex128 path hits 1e-15).
        assert rel < 1e-6, f"parseval rel {rel}"

    def test_group_convolve_matches_naive(self):
        # The load-bearing parity test (A1/R4): FFT group convolution must
        # equal the naive O(p^6) definition.
        f = _rand_complex((P, P, P))
        g = _rand_complex((P, P, P))
        fast = _to_numpy(group_convolve(f, g, backend=b))
        naive = _to_numpy(group_convolve_naive(f, g, backend=b))
        rel = np.max(np.abs(fast - naive)) / max(np.max(np.abs(naive)), 1e-12)
        assert rel < 1e-8, f"conv rel err {rel}"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _close(a, b, atol=1e-4, rtol=1e-4):
    a = _to_numpy(a)
    b = _to_numpy(b)
    return np.allclose(a, b, atol=atol, rtol=rtol)


def _to_numpy(x):
    try:
        return x.detach().cpu().numpy()
    except AttributeError:
        return np.asarray(x)


def _rand_complex(shape):
    rng = np.random.default_rng(0)
    re = rng.standard_normal(shape).astype(np.float32)
    im = rng.standard_normal(shape).astype(np.float32)
    return b.asarray((re + 1j * im).astype(np.complex64))
