"""Backend abstraction layer.

A thin trait over an array library so that CR-NN operators can run on either
PyTorch (research / debugging, CPU+CUDA) or MLX (Apple Silicon, Metal).

Design principle (from docs/math.md §4.3 and assumptions R5/R6):
    *anything expressible with the backend's built-in FFT + complex arithmetic
    must be written that way* — no custom kernels except for the Korányi/Szegő
    complex-power singularity, and even there only when built-ins fall short.

Only the primitives that CR-NN operators actually need are exposed here. We do
NOT aim for full array-API compliance; CR operators are too specialized for
array-api-compat to cover them (see docs/references.bib, [arrayapi]).

Complex dtype convention
------------------------
Both backends use complex64 as the common complex dtype (MLX has no complex128,
so torch is pinned to complex64 for cross-backend numerical parity; see R6).
Arrays are stored with the last axis as the *fast* (contiguous) axis; FFT axes
follow numpy/torch convention.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Backend(Protocol):
    """Trait describing the array primitives CR-NN operators depend on.

    Implementations hold no mutable state beyond a device/dtype spec; they are
    intended to be created once and passed explicitly to every operator
    constructor. This makes the dependency on the backend *visible* in the type
    signature rather than hidden behind a global, which matters because the
    numerical-parity tests (tests/) instantiate both backends side by side.
    """

    # ----- identity / config -------------------------------------------------
    name: str          # "torch" | "mlx"
    float_dtype: str   # "float32"
    complex_dtype: str # "complex64"

    # ----- array construction ------------------------------------------------
    def astype(self, x, dtype: str):
        ...

    def zeros(self, shape, dtype: str | None = None):
        ...

    def ones(self, shape, dtype: str | None = None):
        ...

    def asarray(self, x, dtype: str | None = None):
        ...

    # ----- complex <-> real views -------------------------------------------
    # CR operators are natively complex-valued; the Szegő projection output
    # feeds a real-valued head, so real/complex interop is on the hot path.
    def to_complex(self, x):
        """Promote a real array to complex64 with zero imaginary part."""

    def to_real(self, x):
        """Return the real part as a float32 array."""

    def real(self, x):
        ...

    def imag(self, x):
        ...

    def conj(self, x):
        ...

    # ----- elementwise -------------------------------------------------------
    def exp(self, x):
        ...

    def log(self, x):
        """Principal-branch complex log. Both backends must agree on the
        branch cut (negative real axis); see assumption R5."""

    def pow(self, base, exp):
        """Complex power via exp(exp * log(base)), principal branch."""

    def abs(self, x):
        ...

    def sqrt(self, x):
        ...

    # ----- FFT (the load-bearing primitive) ---------------------------------
    # Heisenberg-FFT, Szegő convolution and the sub-Laplacian spectral basis
    # all reduce to these. Keeping the surface minimal makes MLX's mlx.core.fft
    # a drop-in (see docs/references.bib [mlx2024]).
    def fft(self, x, axis: int = -1):
        ...

    def ifft(self, x, axis: int = -1):
        ...

    def fftn(self, x, axes=None):
        ...

    def ifftn(self, x, axes=None):
        ...

    def fftshift(self, x, axes=None):
        ...

    # ----- linear algebra ----------------------------------------------------
    def matmul(self, a, b):
        ...

    def sum(self, x, axis=None):
        ...

    # ----- autograd plumbing -------------------------------------------------
    # These are the *only* autograd-aware hooks. Custom kernels (Phase 4 only)
    # register their vjp through these. For built-in ops autograd is automatic.
    def detach(self, x):
        ...

    def requires_grad_(self, x, flag: bool = True):
        ...


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

def get_backend(name: str = "torch") -> "Backend":
    """Return a backend instance by name.

    Importing torch/mlx is deferred to this call so that a torch-only machine
    never imports mlx (and vice versa). The default is ``"torch"`` for the
    research phase (M1–M5); MLX is added in M6.
    """
    if name == "torch":
        from .backends import torch_backend

        return torch_backend.TorchBackend()
    if name == "mlx":
        try:
            from .backends import mlx_backend
        except ImportError as e:  # pragma: no cover - env dependent
            raise ImportError(
                "MLX backend requires Apple Silicon and `pip install mlx`. "
                "It is scheduled for M6; for now use the default torch backend."
            ) from e
        return mlx_backend.MLXBackend()
    raise ValueError(f"unknown backend: {name!r} (expected 'torch' or 'mlx')")


# Default singleton for convenience; operators that take an explicit `backend=`
# kwarg should prefer that over this global so tests can swap backends.
_default: Backend | None = None


def default_backend() -> Backend:
    global _default
    if _default is None:
        _default = get_backend("torch")
    return _default
