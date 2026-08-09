"""PyTorch backend for CR-NN.

This is the reference implementation used during the research phase (M1–M5).
It targets CPU and CUDA, complex64 for cross-backend parity with MLX (see
docs/assumptions.md R6 — MLX has no complex128).

All primitives defer to torch built-ins; no custom kernels. The Korányi/Szegő
complex-power singularity is handled in operator code via the ``pow``/``log``
primitives here, keeping the backend itself kernel-free. Custom Metal kernels
for MLX are a Phase-4 concern and live in the MLX backend, not here.
"""

from __future__ import annotations

import torch

# torch complex dtypes map cleanly; we pin to complex64 for parity with MLX.
_FLOAT = torch.float32
_COMPLEX = torch.complex64


class TorchBackend:
    """Concrete :class:`crnn.backend.Backend` over PyTorch.

    Stateless except for the device, which defaults to CUDA if available so
    the speed probes (M3) pick up the GPU automatically. Tests that need
    CPU/determinism pass ``device="cpu"`` explicitly via the constructor —
    kept out of the Protocol because it's torch-specific.
    """

    name = "torch"
    float_dtype = "float32"
    complex_dtype = "complex64"

    def __init__(self, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    # ----- dtype helpers ----------------------------------------------------
    def _resolve(self, dtype: str | None, default):
        if dtype is None:
            return default
        return {"float32": _FLOAT, "complex64": _COMPLEX}[dtype]

    def astype(self, x, dtype: str):
        return x.to(self._resolve(dtype, _FLOAT))

    # ----- construction -----------------------------------------------------
    def zeros(self, shape, dtype: str | None = None):
        return torch.zeros(shape, dtype=self._resolve(dtype, _FLOAT), device=self.device)

    def ones(self, shape, dtype: str | None = None):
        return torch.ones(shape, dtype=self._resolve(dtype, _FLOAT), device=self.device)

    def asarray(self, x, dtype: str | None = None):
        t = torch.as_tensor(x, device=self.device)
        if dtype is not None:
            t = t.to(self._resolve(dtype, _FLOAT))
        elif t.dtype not in (_FLOAT, _COMPLEX):
            t = t.to(_FLOAT)
        return t

    # ----- complex <-> real -------------------------------------------------
    def to_complex(self, x):
        if torch.is_complex(x):
            return x
        return x.to(_COMPLEX)

    def to_real(self, x):
        return x.real if torch.is_complex(x) else x

    def real(self, x):
        return x.real

    def imag(self, x):
        return x.imag

    def conj(self, x):
        return torch.conj(x)

    # ----- elementwise ------------------------------------------------------
    def exp(self, x):
        return torch.exp(x)

    def log(self, x):
        # Principal branch; torch.log on complex uses the principal branch
        # with cut on the negative real axis — matches MLX/NumPy convention.
        return torch.log(x)

    def pow(self, base, exp):
        # Implement as exp(exp * log(base)) to guarantee principal-branch
        # agreement with the MLX backend (R5). For real base we still go
        # through the complex path only when base is complex.
        if torch.is_complex(base):
            return torch.exp(exp * torch.log(base))
        return torch.pow(base, exp)

    def abs(self, x):
        return torch.abs(x)

    def sqrt(self, x):
        return torch.sqrt(x)

    # ----- FFT --------------------------------------------------------------
    # The load-bearing primitive. torch.fft.* operate on the last axis by
    # default and support arbitrary dim tuples for the n-D variants, matching
    # the Protocol's axis/axes signature.
    def fft(self, x, axis: int = -1):
        return torch.fft.fft(x, dim=axis)

    def ifft(self, x, axis: int = -1):
        return torch.fft.ifft(x, dim=axis)

    def fftn(self, x, axes=None):
        return torch.fft.fftn(x, dim=axes)

    def ifftn(self, x, axes=None):
        return torch.fft.ifftn(x, dim=axes)

    def fftshift(self, x, axes=None):
        return torch.fft.fftshift(x, dim=axes)

    # ----- linear algebra ---------------------------------------------------
    def matmul(self, a, b):
        return torch.matmul(a, b)

    def sum(self, x, axis=None):
        return torch.sum(x, dim=axis) if axis is not None else torch.sum(x)

    # ----- autograd ---------------------------------------------------------
    def detach(self, x):
        return x.detach()

    def requires_grad_(self, x, flag: bool = True):
        return x.requires_grad_(flag)
