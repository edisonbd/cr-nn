"""CR operators on the Heisenberg group.

Implements ``docs/math.md`` §2 and §4:

* horizontal vector fields  X_j = ∂_{x_j} + 2 y_j ∂_t,  Y_j = ∂_{y_j} - 2 x_j ∂_t
* tangential CR operator    ∂̄_b = (1/2) Σ (X_j + i Y_j) d z̄_j
* sub-Laplacian              Δ_b = -Σ (X_j² + Y_j²)
* Korányi kernel             Γ(g) = c_n ρ(g)^{-Q},  Q = 2n+2
* flat Szegő kernel          S(g) = c_n' (|z|² - i t)^{-(n+1)}

All operators are written in the *spectral* representation (math.md §2.2):
we Fourier-transform along the center t, which turns X_j, Y_j into multiplicative
/ differential operators on each λ-slice, and the sub-Laplacian into the
rescaled harmonic oscillator. This is the only representation that lets the
unit tests check "Δ_b on a known eigenfunction returns the eigenvalue"
exactly — finite differences only give that approximately.

The finite-difference path is intentionally NOT provided here; if a stencil
version is needed for ablation it will live in a separate module to avoid
muddying the spectral reference.
"""

from __future__ import annotations

from ..backend import Backend, default_backend
from .heisenberg import koranyi_norm


# ---------------------------------------------------------------------------
# Horizontal vector fields (spectral form)
# ---------------------------------------------------------------------------

def _lambda_axis(fft_of_t):
    """Given the FFT of f along t, return the |λ| frequency grid (real).

    The Heisenberg Fourier parameter λ corresponds to the dual frequency of t.
    For a length-p axis, numpy/torch FFT frequencies are k=0..p-1 mapped to
    [0, 1, ..., p/2-1, -p/2, ..., -1] (for even p). We return λ as those
    integers (unscaled) — scaling by Δt is the caller's responsibility and is
    folded into the eigenvalue (2k+n)|λ| at spectrum.py.
    """
    # Implementation is backend-aware; callers pass the array and we read p
    # from its shape on the transformed axis. Kept as a helper rather than a
    # method so it stays pure.
    raise NotImplementedError  # see spectrum.py for the canonical frequency grid


def X_vector_fields(f, axis_t: int = -1, backend: Backend | None = None):
    """Apply X_j = ∂_{x_j} + 2 y_j ∂_t for each j, spectrally.

    After FFT along t, ∂_t → i λ, so X_j f → (∂_{x_j} + 2 i λ y_j) f̂ on each
    λ-slice. Returns a tensor with one extra leading axis indexed by j.

    ``f`` is complex64 with the t axis at ``axis_t``. The x_j, y_j axes are
    assumed to be the two axes immediately before ``axis_t`` (x_j at -3, y_j
    at -2 for n=1). This convention is fixed by HeisenbergGrid.coordinates.
    """
    b = backend or default_backend()
    # FFT along t
    f_hat = b.fft(f, axis=axis_t)
    p = f.shape[axis_t]
    lam = _freq_grid(p, b)  # shape (p,) real, broadcastable to axis_t
    # λ lives on axis_t; reshape for broadcast
    lam_shape = [1] * f_hat.ndim
    lam_shape[axis_t] = p
    lam = lam.reshape(lam_shape)

    n = (f_hat.ndim - 1) // 2  # x,y axes are 2n, plus t
    results = []
    for j in range(n):
        # ∂_{x_j}: spectral derivative along the x axis. We use FFT-based
        # derivative for exactness on the eigenfunction test.
        x_axis = axis_t - 2 * n + 2 * j   # x_j sits 2(n-j) before the t axis
        dx = _spectral_deriv(f_hat, axis=x_axis, backend=b)
        # 2 i λ y_j: y_j is at axis 2j+1; we need the y coordinate values.
        y_axis = axis_t - 2 * n + 2 * j + 1
        ya = y_axis % f_hat.ndim
        y_coords = _coord_grid(f_hat.shape[ya], b).reshape(
            [f_hat.shape[a] if a == ya else 1 for a in range(f_hat.ndim)]
        )
        term = dx + 2j * lam * y_coords * f_hat
        results.append(term)
    # Stack along a new leading axis → shape (n, ...)
    return _stack(b, results)


def Y_vector_fields(f, axis_t: int = -1, backend: Backend | None = None):
    """Apply Y_j = ∂_{y_j} - 2 x_j ∂_t for each j, spectrally."""
    b = backend or default_backend()
    f_hat = b.fft(f, axis=axis_t)
    p = f.shape[axis_t]
    lam = _freq_grid(p, b)
    lam_shape = [1] * f_hat.ndim
    lam_shape[axis_t] = p
    lam = lam.reshape(lam_shape)

    n = (f_hat.ndim - 1) // 2
    results = []
    for j in range(n):
        y_axis = axis_t - 2 * n + 2 * j + 1
        dy = _spectral_deriv(f_hat, axis=y_axis, backend=b)
        x_axis = axis_t - 2 * n + 2 * j
        xa = x_axis % f_hat.ndim
        x_coords = _coord_grid(f_hat.shape[xa], b).reshape(
            [f_hat.shape[a] if a == xa else 1 for a in range(f_hat.ndim)]
        )
        term = dy - 2j * lam * x_coords * f_hat
        results.append(term)
    return _stack(b, results)


# ---------------------------------------------------------------------------
# ∂̄_b  and  Δ_b
# ---------------------------------------------------------------------------

def Dbard(f, axis_t: int = -1, backend: Backend | None = None):
    """∂̄_b f = (1/2) Σ_j (X_j + i Y_j) f  d z̄_j.

    Note (X_j + i Y_j) simplifies: the iλ cross terms cancel partially,
    leaving the (0,1)-type operator. Returns the per-j components stacked
    on a leading axis, matching math.md §2.1.
    """
    b = backend or default_backend()
    X = X_vector_fields(f, axis_t=axis_t, backend=b)
    Y = Y_vector_fields(f, axis_t=axis_t, backend=b)
    # (X_j + i Y_j); both already in Fourier-λ domain
    combined = X + 1j * Y
    return combined * 0.5


def Delta_b(f, axis_t: int = -1, backend: Backend | None = None):
    """Sub-Laplacian Δ_b = -Σ_j (X_j² + Y_j²).

    Implemented spectrally: Δ_b diagonalizes to (2k+n)|λ| on the Hermite
    basis (math.md §3.2). Here we apply it as -Σ(X_j²+Y_j²) directly in the
    Fourier-λ domain via repeated spectral differentiation. The eigenvalue
    test in experiments/01 checks this against (2k+n)|λ|.

    Returns array of same shape as ``f`` (inverse-FFT'd back to t-domain).
    """
    b = backend or default_backend()
    # Work in λ-domain throughout; X,Y already FFT along t.
    X = X_vector_fields(f, axis_t=axis_t, backend=b)  # (n, ...)
    Y = Y_vector_fields(f, axis_t=axis_t, backend=b)  # (n, ...)
    # X_j² = X_j(X_j f): apply X_j to each component. Since X_j is diagonal
    # in λ but mixes x, we reuse X_vector_fields treating the (n,...) array
    # as n independent fields. Simplest correct approach: loop.
    n = X.shape[0]
    acc = b.zeros(X.shape[1:], dtype="complex64")
    for j in range(n):
        # X_j (X_j f): apply X_j to the j-th component
        Xj_of_Xj = _apply_Xj(X[j], j, f.ndim, axis_t, b)
        Yj_of_Yj = _apply_Yj(Y[j], j, f.ndim, axis_t, b)
        acc = acc - (Xj_of_Xj + Yj_of_Yj)
    # inverse FFT along t to return to spatial domain
    return b.ifft(acc, axis=axis_t)


def _apply_Xj(field, j, ndim, axis_t, b):
    """Apply X_j to a single field already in λ-domain."""
    # Reuse the per-j computation factored out of X_vector_fields.
    n = (ndim - 1) // 2
    x_axis = axis_t - 2 * n + 2 * j
    dx = _spectral_deriv(field, axis=x_axis, backend=b)
    p = field.shape[axis_t]
    lam = _freq_grid(p, b)
    lam_shape = [1] * field.ndim
    lam_shape[axis_t] = p
    lam = lam.reshape(lam_shape)
    y_axis = axis_t - 2 * n + 2 * j + 1
    ya = y_axis % field.ndim
    y_coords = _coord_grid(field.shape[ya], b).reshape(
        [field.shape[a] if a == ya else 1 for a in range(field.ndim)]
    )
    return dx + 2j * lam * y_coords * field


def _apply_Yj(field, j, ndim, axis_t, b):
    n = (ndim - 1) // 2
    y_axis = axis_t - 2 * n + 2 * j + 1
    dy = _spectral_deriv(field, axis=y_axis, backend=b)
    p = field.shape[axis_t]
    lam = _freq_grid(p, b)
    lam_shape = [1] * field.ndim
    lam_shape[axis_t] = p
    lam = lam.reshape(lam_shape)
    x_axis = axis_t - 2 * n + 2 * j
    xa = x_axis % field.ndim
    x_coords = _coord_grid(field.shape[xa], b).reshape(
        [field.shape[a] if a == xa else 1 for a in range(field.ndim)]
    )
    return dy - 2j * lam * x_coords * field


# ---------------------------------------------------------------------------
# Kernels (closed form)
# ---------------------------------------------------------------------------

def koranyi_kernel(z, t, n: int, backend: Backend | None = None,
                   eta: float = 1e-6):
    """Γ(g) = c_n ρ(g)^{-Q},  Q = 2n+2  (math.md §4.1).

    ``z`` complex (..., n), ``t`` real (...). Returns real (...). The
    normalization c_n follows Folland (1975); here we use the unnormalized
    form and let the layer learn a scale — exact c_n only matters for the
    Δ_b Γ = δ identity, which the unit test checks up to a known constant.

    ``eta`` regularizes the singularity at ρ=0 (R5 / math.md §4.3).
    """
    b = backend or default_backend()
    Q = 2 * n + 2
    rho = koranyi_norm(z, t, backend=b)
    rho2 = rho ** 2 + eta
    return rho2 ** (-(Q / 2))   # ρ^{-Q} = (ρ²)^{-Q/2}


def szego_kernel_flat(z, t, n: int, backend: Backend | None = None,
                      eta: float = 1e-6):
    """Flat Szegő kernel S(g) = c_n' (|z|² - i t)^{-(n+1)}  (math.md §4.2).

    Complex-valued, principal branch (R5). ``eta`` regularizes the origin.

    Note |z|² - i t has modulus sqrt(|z|⁴ + t²) = ρ², so this kernel has the
    same singularity order as the Korányi kernel — consistent with S being
    the projection onto CR-holomorphic functions, which are ρ^{-(n+1)}
    homogeneous rather than ρ^{-Q}.
    """
    b = backend or default_backend()
    zc = b.to_complex(z) if not _is_complex(z, b) else z
    tc = b.to_complex(t) if not _is_complex(t, b) else t
    absz2 = b.sum(b.abs(zc) ** 2, axis=-1)        # |z|², real → promote
    absz2_c = b.to_complex(absz2)
    w = absz2_c - 1j * tc
    # regularize: |w|² = ρ⁴; add eta to |w| to avoid branch-point blowup
    w_reg = w + b.to_complex(b.asarray(eta, dtype="float32"))
    # (n+1)-th power, principal branch
    return b.pow(w_reg, -(n + 1))


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _freq_grid(p, b):
    """FFT frequency grid: [0,1,...,p/2-1, -p/2, ..., -1] as float32."""
    if p % 2 == 0:
        half = p // 2
        freqs = list(range(half)) + list(range(-half, 0))
    else:
        half = (p - 1) // 2
        freqs = list(range(half + 1)) + list(range(-half, 0))
    return b.asarray([float(k) for k in freqs], dtype="float32")


def _coord_grid(p, b):
    """Coordinate values [0,1,...,p-1] as float32 (unscaled physical units)."""
    return b.asarray([float(k) for k in range(p)], dtype="float32")


def _spectral_deriv(f, axis: int, backend: Backend):
    """Derivative along ``axis`` via FFT: d/dx ↔ i·k after FFT.

    Exact for band-limited functions; this is what makes the Δ_b eigenvalue
    test check out to machine precision rather than O(Δx²).
    """
    b = backend
    p = f.shape[axis]
    k = _freq_grid(p, b)
    shape = [1] * f.ndim
    shape[axis] = p
    k = k.reshape(shape)
    k = b.to_complex(k)
    f_hat = b.fft(f, axis=axis)
    df_hat = 1j * k * f_hat
    return b.ifft(df_hat, axis=axis)


def _stack(b, arrays):
    """Stack a list of arrays along a new leading axis.

    torch and mlx both expose ``stack(arrays, axis=0)``; torch calls the kwarg
    ``dim``, mlx ``axis``. Dispatch on the backend name to keep the Protocol
    surface minimal.
    """
    if getattr(b, "name", None) == "torch":
        import torch
        return torch.stack(arrays, dim=0)
    # MLX path (M6)
    import mlx.core as mx  # pragma: no cover
    return mx.stack(arrays, axis=0)  # pragma: no cover


def _is_complex(x, b):
    is_c = getattr(x, "is_complex", None)
    if callable(is_c):
        return is_c()
    dtype = getattr(x, "dtype", None)
    return dtype is not None and "complex" in str(dtype).lower()
