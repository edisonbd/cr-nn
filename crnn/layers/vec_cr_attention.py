"""Vector-valued CR-Attention (pre-M5 CR completion).

Upgrades CRAttention so the whole block works on a *complex vector-valued
field* on H_p instead of B*d independent real scalar fields:

  * the d channels travel through the block as a complex field
    (B, p, p, p, d) -- one field on H_p with d complex components;
  * the Szego projection (group convolution, scalar kernel) acts per
    channel, exactly as in CRAttention;
  * a learnable complex channel mixer (d x d complex affine, applied at
    every grid point) couples the channels *inside* the CR aggregation,
    replacing the Euclidean out_proj that used to be the only cross-channel
    mechanism;
  * dbar_b energy gating is applied per grid point, per channel.

The block keeps the complex structure end-to-end: residual and FFN operate
on complex tensors (see VecCRBlock), and only the final readout maps
complex -> real.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..curvature.perturbation import apply_log_correction, apply_perturbation
from .complex_nn import ComplexFFN, ComplexRMSNorm, ComplexLinear
from .fluid_attention import FluidCRAttention
from .piecewise_cr_attention import PiecewiseCRAttention
from .cr_attention_backend import cr_group_convolve


def _nearest_prime(n: int) -> int:
    n = int(n)
    while True:
        if _is_prime(n):
            return n
        n += 1


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    f = 3
    while f * f <= n:
        if n % f == 0:
            return False
        f += 2
    return True


_KERNEL_CACHE: dict = {}


def build_szego_kernel(p: int, n_cr: int, eta: float, device, dtype):
    """Flat Szego kernel S(z,t) = conj(w)^{n+1} / (|w|^2 + eta)^{n+1} on the
    p x p x p grid (fftshifted coords), modulus-regularised (math.md 4.3)."""
    key = (p, n_cr, eta, str(device), str(dtype))
    if key not in _KERNEL_CACHE:
        coords = np.fft.fftfreq(p, d=1.0) * p
        xx = coords.reshape(p, 1, 1)
        yy = coords.reshape(1, p, 1)
        tt = coords.reshape(1, 1, p)
        w = (xx ** 2 + yy ** 2) - 1j * tt
        n1 = n_cr + 1
        wsq = (w * np.conj(w)).real + eta
        S = np.conj(w) ** n1 / (wsq ** n1)
        _KERNEL_CACHE[key] = torch.from_numpy(S.astype(np.complex64)).to(
            device=device, dtype=dtype)
    return _KERNEL_CACHE[key]


def horizontal_energy(f: torch.Tensor, p: int) -> torch.Tensor:
    """Per-point |grad_H f|^2 via spectral derivatives (proxy for |dbar_b f|)."""
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


class VecCRAttention(nn.Module):
    """Vector-valued Szego-projection attention.

    Input  : (B, N, d) complex field (N = p^3)
    Output : (B, N, d) complex field (residual is the block's job)
    """

    def __init__(self, d_model: int, p: int | None = None, n_cr: int = 1,
                 eta: float = 1e-6, gate: bool = True, mix: bool = True,
                 M: int = 0, eps_max: float = 0.1,
                 log_correction: bool = False, eps_init: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.p = p
        self.n_cr = n_cr
        self.eta = eta
        self.gate = gate
        self.mix = mix
        self.M = M
        self.eps_max = eps_max
        self.log_correction = log_correction
        if p is not None and not _is_prime(p):
            raise ValueError(f"p={p} must be prime (R7)")
        if mix:
            self.channel_mix = ComplexLinear(d_model, d_model)
        if M > 0:
            self.eps = nn.Parameter(torch.full((M,), eps_init))
        if log_correction:
            self.eps_log = nn.Parameter(torch.zeros(1))
        # complex per-channel gain (FiLM-style) applied to the aggregated
        # field; init identity so the layer starts as the bare projection.
        self.gain_r = nn.Parameter(torch.ones(d_model))
        self.gain_i = nn.Parameter(torch.zeros(d_model))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, N, d = z.shape
        p = self.p or _nearest_prime(round(N ** (1 / 3)))
        if p ** 3 != N:
            z = self._pad_to_grid(z, p)
            N = p ** 3
        fg = z.permute(0, 2, 1).reshape(B * d, p, p, p).contiguous()

        S = build_szego_kernel(p, self.n_cr, self.eta, fg.device, fg.dtype)
        out = cr_group_convolve(fg, S, p)                 # (B*d, p,p,p)

        if self.M > 0:
            out = self._apply_perturbation(out, p)

        if self.mix:
            out = out.reshape(B, p, p, p, d)
            # residual mixer: out <- out + W(out); identity path keeps the
            # bare projection as the init behaviour and stabilises gradients.
            out = out + self.channel_mix(out)
            out = out.reshape(B * d, p, p, p)

        # complex per-channel gain; the flat layout is (B*d, p, p, p) with
        # channel index = b*d + c, so repeat the d gains B times.
        gain = torch.complex(self.gain_r, self.gain_i)
        out = out * gain.repeat(B).reshape(-1, 1, 1, 1)

        if self.gate:
            e = horizontal_energy(fg, p)                   # (B*d, p,p,p) real
            gate_w = torch.sigmoid(-e)
            out = out * gate_w.to(out.dtype)

        return out.reshape(B, N, d)                        # (B, N, d) complex

    def _apply_perturbation(self, out, p):
        # S_curved f ~= S_flat f + sum_j eps_j * Delta_b^j[S_flat f]
        eps_eff = self.eps_max * torch.tanh(self.eps)
        out = apply_perturbation(out, eps_eff, self.M, normalize=True)
        if self.log_correction:
            eps_log_eff = self.eps_max * torch.tanh(self.eps_log)
            out = apply_log_correction(out, eps_log_eff, p, self.eta)
        return out

    def _pad_to_grid(self, z, p):
        N_target = p ** 3
        B, N, d = z.shape
        if N == N_target:
            return z
        if N < N_target:
            pad = torch.zeros(B, N_target - N, d, device=z.device,
                              dtype=z.dtype)
            return torch.cat([z, pad], dim=1)
        return z[:, :N_target, :]


class VecCRBlock(nn.Module):
    """CR-Block operating fully in complex space.

    norm -> VecCRAttention -> residual -> norm -> ComplexFFN -> residual.
    Input/output: (B, N, d) complex.
    """

    def __init__(self, d_model: int, p: int | None = None, n_cr: int = 1,
                 eta: float = 1e-6, gate: bool = True, mix: bool = True,
                 ff_expansion: int = 4, M: int = 0, eps_max: float = 0.1,
                 log_correction: bool = False, eps_init: float = 0.0,
                 attn_type: str = "szego", spectrum: str = "full",
                 spectral_mix: bool = False, n_flow: int = 3,
                 nl: str = "gelu", twist: bool = True,
                 prune_rate: float = 0.0, checkpoint: bool = False):
        super().__init__()
        self.checkpoint = checkpoint
        self.norm1 = ComplexRMSNorm(d_model)
        if attn_type == "fluid":
            self.attn = FluidCRAttention(d_model, p=p, n_cr=n_cr, eta=eta,
                                         gate=gate, mix=mix,
                                         spectrum=spectrum,
                                         spectral_mix=spectral_mix)
        elif attn_type == "piecewise":
            self.attn = PiecewiseCRAttention(d_model, p=p, n_cr=n_cr, eta=eta,
                                             gate=gate, mix=mix,
                                             n_flow=n_flow,
                                             spectrum=spectrum,
                                             spectral_mix=spectral_mix,
                                             nl=nl, twist=twist,
                                             prune_rate=prune_rate,
                                             checkpoint=checkpoint)
        else:
            self.attn = VecCRAttention(d_model, p=p, n_cr=n_cr, eta=eta,
                                       gate=gate, mix=mix, M=M,
                                       eps_max=eps_max,
                                       log_correction=log_correction,
                                       eps_init=eps_init)
        self.norm2 = ComplexRMSNorm(d_model)
        self.ffn = ComplexFFN(d_model, ff_expansion)

    def forward(self, z):
        z = z + self.attn(self.norm1(z))
        if self.checkpoint:
            z = z + torch.utils.checkpoint.checkpoint(
                self.ffn, self.norm2(z), use_reentrant=False)
        else:
            z = z + self.ffn(self.norm2(z))
        return z
