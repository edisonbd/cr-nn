"""Heisenberg group structure and discrete grid.

Implements the math in ``docs/math.md`` §1 and §7:

* continuous group law  (z,t)·(z',t') = (z+z', t+t' + 2 Im(z·z̄'))
* Korányi–Cygan norm     ρ(z,t) = (|z|⁴ + t²)^{1/4}
* finite Heisenberg group H_p grid for the discrete FFT path (§7.1)

The grid is the (x, y, t) sampling of H_p = (Z/pZ)^{2n+1} with the *discrete*
group law, which is what the O(N log N) Diaconis–Rockmore FFT operates on
(assumption A1). We do NOT sample the continuous group and pretend it's a
group — that would break the convolution structure the FFT relies on (R4).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..backend import Backend, default_backend


# ---------------------------------------------------------------------------
# Continuous-group helpers (used for kernel evaluation on the grid)
# ---------------------------------------------------------------------------

def group_multiply(g1, g2, backend: Backend | None = None):
    """Continuous group law (z,t)·(z',t') = (z+z', t+t' + 2 Im(z·z̄')).

    ``g1, g2`` are tuples ``(z, t)`` with ``z`` complex of shape (..., n) and
    ``t`` real of shape (...). Returns the same structure.
    """
    b = backend or default_backend()
    z1, t1 = g1
    z2, t2 = g2
    z1c = b.to_complex(z1) if not _is_complex(z1, b) else z1
    z2c = b.to_complex(z2) if not _is_complex(z2, b) else z2
    z = z1c + z2c
    # 2 * Im(z1 · conj(z2))  — dot product over the n axis (last).
    inner = b.sum(z1c * b.conj(z2c), axis=-1)
    t = t1 + t2 + 2.0 * b.imag(inner)
    return z, t


def inverse(g, backend: Backend | None = None):
    """Inverse of (z,t) is (-z, -t) under the continuous law."""
    z, t = g
    return -z, -t


def koranyi_norm(z, t, backend: Backend | None = None):
    """ρ(z,t) = (|z|⁴ + t²)^{1/4}, the Korányi–Cygan norm (math.md §4.1).

    ``z`` complex (..., n), ``t`` real (...). Returns real (...).
    """
    b = backend or default_backend()
    z_abs2 = b.sum(b.abs(z) ** 2, axis=-1)   # |z|²
    return b.sqrt(b.sqrt(z_abs2 ** 2 + t ** 2))


def _is_complex(x, b):
    # torch: torch.is_complex; mlx: mlx.array.dtype == complex64
    is_c = getattr(x, "is_complex", None)
    if callable(is_c):
        return is_c()
    dtype = getattr(x, "dtype", None)
    return dtype is not None and "complex" in str(dtype).lower()


# ---------------------------------------------------------------------------
# Discrete grid (finite Heisenberg group H_p)
# ---------------------------------------------------------------------------

@dataclass
class HeisenbergGrid:
    """Sampling of the finite Heisenberg group H_p on a (p,p,p) lattice.

    For CR dimension n the natural grid is (x_a, y_b, t_c) with
    a,b,c ∈ Z/pZ; for n>1 the (x,y) blocks stack along extra axes. We
    expose the n=1 block (one complex coordinate) and let higher-n code
    compose several. The default p≈N^{1/3} mapping lives in the embedding
    layer (math.md §7.1), not here — this class only owns the geometry.

    The group law on H_p is the reduction of the continuous law mod p
    (with the commutator term 2(xy'-yx') taken mod p). Using it directly
    — rather than a naive uniform sample — is what keeps group convolution
    reducible to FFT (assumption R4).

    Attributes
    ----------
    p : int
        Grid resolution per axis; total points N = p³ (for n=1).
    n : int
        CR complex dimension.
    """

    p: int
    n: int = 1

    @property
    def N(self) -> int:
        """Total number of grid points = p^(2n+1)."""
        return self.p ** (2 * self.n + 1)

    def coordinates(self, backend: Backend | None = None):
        """Return (x, y, t) coordinate arrays over Z/pZ.

        Shapes: x,y have shape (p,)*2n × (1,)*1 broadcastable; t has shape
        (p,) on the last axis. All are float32 in [0, p). We return the raw
        integer-mod coordinates (not scaled to physical units) — physical
        scaling is applied where kernels are evaluated, so the FFT sees the
        algebraically correct H_p.
        """
        b = backend or default_backend()
        p = self.p
        # 1D coordinate over Z/pZ
        idx = b.asarray([float(i) for i in range(p)], dtype="float32")
        # For n=1: x shape (p,1,1), y shape (1,p,1), t shape (1,1,p)
        x = idx.reshape((p, 1, 1))
        y = idx.reshape((1, p, 1))
        t = idx.reshape((1, 1, p))
        if self.n == 1:
            return x, y, t
        # Higher n: extend by tiling additional (x_j, y_j) axes. Returned as
        # lists per coordinate index; callers compose. (Implemented when n>1
        # layers are exercised; M2 focuses on n=1 for the unit tests.)
        raise NotImplementedError(
            f"HeisenbergGrid.coordinates for n={self.n} not yet implemented; "
            "M2 validates n=1, higher n follows in M3."
        )

    def group_multiply_mod(self, g1, g2):
        """Discrete group law mod p. g1, g2 are (x, y, t) integer triples.

        (x,y,t)·(x',y',t') = (x+x', y+y', t+t' + 2(x·y' - y·x')) mod p.

        This is the algebra that the FFT diagonalizes; the continuous
        group_multiply above is only for kernel evaluation.
        """
        p = self.p
        x1, y1, t1 = g1
        x2, y2, t2 = g2
        return (
            (x1 + x2) % p,
            (y1 + y2) % p,
            (t1 + t2 + 2 * (x1 * y2 - y1 * x2)) % p,
        )
