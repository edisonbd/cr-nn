"""Fluid attention: geometric-orthogonal spectral flow on H_p.

The user-direction (2026-08-10): treat a text window as a *fluid* on the
Heisenberg group, replace the matrix-valued group convolution (O(p^4)
batched matmuls) with a *geometric orthogonal* spectral operator, and mix
information by letting the fluid flow.

Implementation:
    f --FFT_t--> --FFT2_(x,y)--> f_hat(lam, xi_x, xi_y)
    f_hat <- W(lam, xi) * f_hat          (pointwise complex modulation)
    --IFFT2--> --IFFT_t--> out

with the learnable flow spectrum

    W(lam, xi) = exp(-t * (|lam| + |xi|^2)) * (1 + 0.01 * (Wr + i Wi))

where t is a learnable flow time (init 0.1), Wr/Wi a learnable complex
residual spectrum (init 0).  Then a residual complex channel mixer, a
complex per-channel gain and the dbar_b gate complete the block, exactly as
in VecCRAttention, so FluidCRAttention is a drop-in for it.

HONEST MATH NOTE: pointwise modulation in the abelian Fourier basis is an
orthogonal-geometric operator but is NOT the exact H_p group convolution
(which requires the matrix-valued transform; the Szego projection lives
there).  This trades the non-commutative structure for O(N log N) diagonal
computation.  The diffusion init keeps a "fluid" (heat-flow-like) semantics;
the exact CR projection remains available via VecCRAttention.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .complex_nn import ComplexLinear


def horizontal_energy(f: torch.Tensor, p: int) -> torch.Tensor:
    """Per-point |grad_H f|^2 via spectral derivatives (dbar_b proxy)."""
    fx = _spectral_deriv(f, axis=-3, p=p)
    fy = _spectral_deriv(f, axis=-2, p=p)
    return (fx * fx.conj() + fy * fy.conj()).real


def _spectral_deriv(f: torch.Tensor, axis: int, p: int) -> torch.Tensor:
    k = torch.fft.fftfreq(p, d=1.0) * p
    shape = [1] * f.ndim
    shape[axis] = p
    k = k.view(shape).to(f.dtype).to(f.device)
    f_hat = torch.fft.fft(f, dim=axis)
    return torch.fft.ifft(1j * k * f_hat, dim=axis)


class FluidCRAttention(nn.Module):
    """Diagonal spectral-flow attention on the H_p grid (complex field).

    Expressiveness of the spectral filter is controlled by ``spectrum``:
      "full"      : per-mode learnable complex weights Wr/Wi (p, p, p);
                    any diagonal filter, 2 p^3 params per layer.
      "mlp"       : a small MLP over frequency coordinates (lam, xi_x, xi_y)
                    generates the complex weights; smooth and generalises
                    across p, few params.
      "diffusion" : base heat-flow spectrum only (no learnable residual).
    All variants keep the orthogonal diagonal structure (O(N log N), no
    matrix products); the learnable residual multiplies the diffusion base.
    """

    def __init__(self, d_model: int, p: int | None = None, n_cr: int = 1,
                 eta: float = 1e-6, gate: bool = True, mix: bool = True,
                 t_init: float = 0.1, spectrum: str = "full",
                 spec_scale: float = 0.05, spectral_mix: bool = False):
        super().__init__()
        self.d_model = d_model
        self.p = p
        self.n_cr = n_cr
        self.eta = eta
        self.gate = gate
        self.mix = mix
        self.spectrum = spectrum
        self.spec_scale = spec_scale
        self.spectral_mix = spectral_mix
        if spectrum not in ("full", "mlp", "diffusion"):
            raise ValueError(f"spectrum={spectrum!r} "
                             "(expected 'full'|'mlp'|'diffusion')")
        if spectrum == "full" and p is None:
            raise ValueError("spectrum='full' requires p at construction")
        self.t = nn.Parameter(torch.tensor(t_init, dtype=torch.float32))
        if spectrum == "full":
            self.Wr = nn.Parameter(torch.zeros(p, p, p))
            self.Wi = nn.Parameter(torch.zeros(p, p, p))
        elif spectrum == "mlp":
            self.spec_mlp = nn.Sequential(
                nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, 2))
            nn.init.zeros_(self.spec_mlp[-1].weight)
            nn.init.zeros_(self.spec_mlp[-1].bias)
        if mix:
            self.channel_mix = ComplexLinear(d_model, d_model)
        if spectral_mix:
            # cross-channel coupling in the frequency domain: a shared
            # complex d x d affine applied at every mode.  O(p^3 d^2), no
            # p x p matrices -- the missing non-diagonal expressivity.
            self.spec_mix = ComplexLinear(d_model, d_model)
        self.gain_r = nn.Parameter(torch.ones(d_model))
        self.gain_i = nn.Parameter(torch.zeros(d_model))
    def _flow_weights(self, p, device, dtype):
        lam = torch.fft.fftfreq(p, d=1.0).to(device)          # (p,)
        xi = torch.fft.fftfreq(p, d=1.0).to(device)           # (p,)
        lam = lam.reshape(1, 1, p)
        xix = xi.reshape(p, 1, 1)
        xiy = xi.reshape(1, p, 1)
        diff = torch.exp(-self.t * (lam.abs() + xix ** 2 + xiy ** 2))
        if self.spectrum == "diffusion":
            res = torch.ones_like(diff)
        elif self.spectrum == "full":
            res = 1.0 + self.spec_scale * torch.complex(self.Wr, self.Wi)
        else:  # mlp: learned filter function of the frequency coordinates
            coords = torch.stack(
                [xix.expand(p, p, p), xiy.expand(p, p, p),
                 lam.expand(p, p, p)], dim=-1) / p
            out = self.spec_mlp(coords)                        # (p,p,p,2)
            res = 1.0 + self.spec_scale * torch.complex(out[..., 0],
                                                        out[..., 1])
        return (diff * res).to(dtype)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, N, d = z.shape
        p = self.p or round(N ** (1 / 3))
        fg = z.permute(0, 2, 1).reshape(B * d, p, p, p).contiguous()

        # orthogonal spectral flow
        f_hat = torch.fft.fft(fg, dim=-1)
        f_hat = torch.fft.fft2(f_hat, dim=(-3, -2))
        if self.spectral_mix:
            f_hat = f_hat.reshape(B, d, p, p, p).permute(0, 2, 3, 4, 1)
            f_hat = self.spec_mix(f_hat)
            f_hat = f_hat.permute(0, 4, 1, 2, 3).reshape(B * d, p, p, p)
        W = self._flow_weights(p, fg.device, fg.dtype)
        out = f_hat * W
        out = torch.fft.ifft2(out, dim=(-3, -2))
        out = torch.fft.ifft(out, dim=-1)

        if self.mix:
            out = out.reshape(B, p, p, p, d)
            out = out + self.channel_mix(out)
            out = out.reshape(B * d, p, p, p)

        gain = torch.complex(self.gain_r, self.gain_i)
        out = out * gain.repeat(B).reshape(-1, 1, 1, 1)

        if self.gate:
            e = horizontal_energy(fg, p)
            out = out * torch.sigmoid(-e).to(out.dtype)

        return out.reshape(B, N, d)
