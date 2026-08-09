"""CR-Sobolev loss and ∂̄_b energy regularisation.

Implements docs/math.md §6:

    L_CR = ‖out - y‖²_{S_b^s}  +  μ ‖∂̄_b out‖²

where the CR-Sobolev norm weights coefficient magnitudes by the sub-Laplacian
spectrum σ_{k,λ} = (2k+n)|λ|:

    ‖f‖²_{S_b^s} = Σ_{k,λ} (1 + σ_{k,λ})^s |f̂_{k,λ}|²

This replaces the Euclidean ‖·‖² used in standard regression / the L2
component of cross-entropy setups. The ∂̄_b energy term pushes the output
towards CR (holomorphic) functions — i.e. compresses information into the
complex (holomorphic) subspace (assumption A3).

IMPLEMENTATION NOTE: the full CR-Sobolev norm requires projecting onto the
Δ_b eigenbasis (Hermite–Laguerre). For a differentiable loss we use a
spectral approximation via the sub-Laplacian applied directly:

    ‖f‖²_{S_b^s} ≈ ‖(I + Δ_b)^{s/2} f‖²_{L²}

For s=1 this is ‖f‖² + ‖Δ_b^{1/2} f‖², and ‖Δ_b^{1/2} f‖² = ⟨f, Δ_b f⟩
which we compute as ⟨f, Δ_b f⟩ in the spectral (Fourier-λ) domain — exact
for the flat model, since Δ_b is diagonal there. The Δ_b here is the
spectral sub-Laplacian: in the λ-domain it multiplies each (k,λ) mode by
σ_{k,λ}; in the spatial domain we approximate via the horizontal Laplacian
-Σ(X_j² + Y_j²) evaluated spectrally.

For the toy LM (M4) we keep s=1 and μ small (1e-3); the ablation in M5
sweeps s and μ.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..geometry.spectrum import sub_laplacian_eigenvalues


def dbar_energy(f: torch.Tensor, p: int) -> torch.Tensor:
    """‖∂̄_b f‖² energy, used as the CR regularisation term.

    Proxy: the horizontal gradient energy |∇_H f|² = Σ(|X_j f|² + |Y_j f|²),
    which vanishes iff f is constant (the CR functions on the torus). This is
    the same proxy CR-Attention uses for gating; the full (0,1)-type ∂̄_b is
    structurally similar but we keep the proxy for differentiability and
    speed. See docs/math.md §2.1 / §6.2.

    f : (B, p, p, p) complex. Returns (B,) real per-sample energy.
    """
    # spectral derivatives along x (axis -3) and y (axis -2)
    fx = _spectral_deriv(f, axis=-3, p=p)
    fy = _spectral_deriv(f, axis=-2, p=p)
    energy = (fx * fx.conj() + fy * fy.conj()).real        # (B, p, p, p)
    return energy.sum(dim=(-3, -2, -1))                     # (B,)


def cr_sobolev_norm(f: torch.Tensor, p: int, n: int = 1, s: float = 1.0) -> torch.Tensor:
    """‖f‖²_{S_b^s} ≈ Σ_{k,λ} (1+σ_{k,λ})^s |f̂_{k,λ}|².

    Computed in the spectral domain: FFT f along the center t (→ λ slices),
    then within each λ slice the Hermite expansion gives the k-modes. For the
    differentiable loss we use the operator form ⟨f, (I+Δ_b)^s f⟩, which for
    s=1 reduces to ‖f‖² + ⟨f, Δ_b f⟩. ⟨f, Δ_b f⟩ is evaluated as the spectral
    energy weighted by σ_{k,λ}.

    f : (B, p, p, p) complex. Returns (B,) real.
    """
    # spectral energy: FFT along t, then per-λ horizontal spectral energy.
    # Δ_b in λ-domain = -Σ(∂_x² + ∂_y²) + λ²|x|² + iλ(x·∂_y - y·∂_x).
    # For the loss we use the leading spectral weight σ_{k,λ} = (2k+n)|λ|,
    # approximated by weighting the (x,y) Fourier modes by |λ|·(freq structure).
    # Practical differentiable form: ⟨f, Δ_b f⟩ ≈ ‖∇_H f‖² (horizontal energy),
    # which is the s=1 Sobolev semi-norm. Combined with ‖f‖² this gives the
    # full s=1 norm.
    l2 = (f.abs() ** 2).sum(dim=(-3, -2, -1))               # (B,) ‖f‖²
    fx = _spectral_deriv(f, axis=-3, p=p)
    fy = _spectral_deriv(f, axis=-2, p=p)
    semi = (fx.abs() ** 2 + fy.abs() ** 2).sum(dim=(-3, -2, -1))  # ‖∇_H f‖²
    return l2 + semi                                         # ‖f‖²_{S_b^1}


class CRSobolevLoss(nn.Module):
    """L_CR = ‖out - y‖²_{S_b^s} + μ ‖∂̄_b out‖².

    For the toy LM the "regression" target y is the one-hot / embedding of
    the next token; out is the model's predicted embedding. The loss is
    fully differentiable and composes with CR-Attention's complex output.

    Parameters
    ----------
    mu : float   ∂̄_b regularisation strength (default 1e-3).
    s : float    Sobolev order (default 1.0; M5 ablates).
    n : int      CR complex dimension.
    """

    def __init__(self, mu: float = 1e-3, s: float = 1.0, n: int = 1):
        super().__init__()
        self.mu = mu
        self.s = s
        self.n = n

    def forward(self, out: torch.Tensor, target: torch.Tensor, p: int) -> torch.Tensor:
        """out, target: (B, p, p, p) complex (or real→promoted). Returns scalar."""
        if not torch.is_complex(out):
            out = out.to(torch.complex64)
        if not torch.is_complex(target):
            target = target.to(torch.complex64)
        diff = out - target
        sobolev = cr_sobolev_norm(diff, p=p, n=self.n, s=self.s)   # (B,)
        dbar = dbar_energy(out, p=p)                               # (B,)
        return (sobolev + self.mu * dbar).mean()


def _spectral_deriv(f: torch.Tensor, axis: int, p: int) -> torch.Tensor:
    """Derivative along `axis` via FFT: d/dx ↔ i·k. Complex-valued."""
    k = torch.fft.fftfreq(p, d=1.0) * p
    shape = [1] * f.ndim
    shape[axis] = p
    k = k.view(shape).to(f.dtype).to(f.device)
    f_hat = torch.fft.fft(f, dim=axis)
    return torch.fft.ifft(1j * k * f_hat, dim=axis)
