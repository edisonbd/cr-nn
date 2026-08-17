"""Pure-geometric CR block: no Euclidean linear layers, no matrix multiplies.

The softmax attention and the Euclidean FFN / channel mixer are all replaced by
*group transforms* (the Heisenberg Szegő projection and the cyclic-group DFT on
the channel fibre) plus pointwise nonlinearities and per-channel gains.  The
only non-pointwise operations are FFTs (matrix-free in the implementation
sense, and group-theoretic in the mathematical sense), so the block carries
**no nn.Linear and no batched matmul** — no Euclidean geometry, no matrix
scheme.

Components
----------
* GeoNorm        : per-position modulus normalization (the CR/fibre modulus,
                   not a Euclidean feature norm with mean subtraction).
* GeoChannelMix  : cross-channel coupling via the DFT on the channel cyclic
                   group Z_d (FFT -> pointwise complex gain -> IFFT).  This
                   replaces ComplexLinear with a matrix-free group transform.
* GeoCRBlock     : norm -> piecewise Szegő attention -> residual -> norm ->
                   GeoChannelMix -> pointwise split-GELU -> per-channel gain ->
                   residual.  Fully complex, fully matrix-free.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .piecewise_cr_attention import PiecewiseCRAttention, _apply_nl


class GeoNorm(nn.Module):
    """Per-position modulus normalization (CR fibre modulus, no mean shift)."""

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.gamma_r = nn.Parameter(torch.ones(d))
        self.gamma_i = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, z):
        rms = z.abs().square().mean(dim=-1, keepdim=True).sqrt()
        z_hat = z / (rms + self.eps)
        return z_hat * torch.complex(self.gamma_r, self.gamma_i)


class GeoChannelMix(nn.Module):
    """Cross-channel mixing via the cyclic-group DFT (matrix-free)."""

    def __init__(self, d: int):
        super().__init__()
        self.gain_r = nn.Parameter(torch.ones(d))
        self.gain_i = nn.Parameter(torch.zeros(d))

    def forward(self, z):                      # (..., d) complex
        zh = torch.fft.fft(z, dim=-1)
        zh = zh * torch.complex(self.gain_r, self.gain_i)
        return torch.fft.ifft(zh, dim=-1)


class GeoCRBlock(nn.Module):
    """Fully geometric, matrix-free CR block.

    norm -> PiecewiseCRAttention -> residual -> norm -> GeoChannelMix ->
    split-GELU -> per-channel gain -> residual.
    """

    def __init__(self, d_model: int, p: int | None = None, n_cr: int = 1,
                 eta: float = 1e-6, gate: bool = True, n_flow: int = 3,
                 spectrum: str = "full", spec_scale: float = 0.05,
                 nl: str = "gelu", twist: bool = True, ff_rounds: int = 3):
        super().__init__()
        self.norm1 = GeoNorm(d_model)
        self.attn = PiecewiseCRAttention(
            d_model, p=p, n_cr=n_cr, eta=eta, gate=gate, mix=False,
            n_flow=n_flow, spectrum=spectrum, spec_scale=spec_scale,
            nl=nl, twist=twist, spectral_mix=False)
        self.norm2 = GeoNorm(d_model)
        # channel-FNO: K rounds of cyclic-DFT channel mixing + nonlinearity,
        # a matrix-free (group-theoretic) stand-in for the Euclidean FFN.
        self.ff_rounds = int(ff_rounds)
        self.mixes = nn.ModuleList([GeoChannelMix(d_model)
                                    for _ in range(self.ff_rounds)])
        self.gain_r = nn.Parameter(torch.ones(d_model))
        self.gain_i = nn.Parameter(torch.zeros(d_model))
        self.nl = nl

    def forward(self, z):
        z = z + self.attn(self.norm1(z))
        h = self.norm2(z)
        for mix in self.mixes:
            h = _apply_nl(mix(h), torch.zeros(1, device=h.device), self.nl)
        h = h * torch.complex(self.gain_r, self.gain_i)
        z = z + h
        return z
