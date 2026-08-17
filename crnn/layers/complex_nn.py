"""Complex-valued building blocks for the fully-CR stack (pre-M5 CR completion).

These replace the Euclidean fallbacks that the M4 CRBlock still used:

  * ComplexLinear   -- a true complex affine map z -> W z + b (W, b complex),
                       implemented as a 2x2 block over (real, imag) so the
                       complex structure is preserved (not a concat of
                       independent real/imag linear maps).
  * ComplexLayerNorm -- normalises the *modulus* per feature (a magnitude
                       preserving norm), then applies a real scale and a
                       complex shift.  This is the CR-friendly replacement
                       for LayerNorm (which is Euclidean by construction).
  * ComplexFFN      -- complex linear -> split-activation (GELU on real/imag
                       separately: non-holomorphic by design, matching the
                       holomorphic/anti-holomorphic split of CR geometry)
                       -> complex linear.

Complex parameters are stored as real/imag pairs (two real Parameters) so
AdamW and the autograd graph stay fully supported.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def complex_affine(z, Wr, Wi, br, bi):
    """Complex affine: (Wr + i Wi) @ z + (br + i bi).

    z  : (..., d_in) complex
    Wr, Wi : (d_out, d_in) real
    br, bi : (d_out,) real
    Returns (..., d_out) complex.
    """
    zr, zi = z.real, z.imag
    out_r = zr @ Wr.t() - zi @ Wi.t() + br
    out_i = zr @ Wi.t() + zi @ Wr.t() + bi
    return torch.complex(out_r, out_i)


class ComplexLinear(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.Wr = nn.Parameter(torch.empty(d_out, d_in))
        self.Wi = nn.Parameter(torch.empty(d_out, d_in))
        self.br = nn.Parameter(torch.zeros(d_out))
        self.bi = nn.Parameter(torch.zeros(d_out))
        # Complex Xavier: each of Wr/Wi is one component of the complex
        # weight; doing plain xavier on both doubles the modulus variance
        # vs a real layer.  Scale the gain by 1/sqrt(2) so |W| keeps the
        # real-layer variance (standard complex-network init practice).
        gain = 1.0 / math.sqrt(2.0)
        nn.init.xavier_uniform_(self.Wr, gain=gain)
        nn.init.xavier_uniform_(self.Wi, gain=gain)

    def forward(self, z):
        return complex_affine(z, self.Wr, self.Wi, self.br, self.bi)


class ComplexLayerNorm(nn.Module):
    """Modulus-normalising layer norm with real scale and complex shift."""

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d))
        self.beta_r = nn.Parameter(torch.zeros(d))
        self.beta_i = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, z):
        mu = z.mean(dim=-1, keepdim=True)
        var = (z - mu).abs().square().mean(dim=-1, keepdim=True)
        z_hat = (z - mu) / torch.sqrt(var + self.eps)
        z_hat = z_hat * self.gamma
        return torch.complex(z_hat.real + self.beta_r, z_hat.imag + self.beta_i)


class ComplexRMSNorm(nn.Module):
    """Modulus RMS norm with a complex per-feature weight (no mean shift).

    This is the complex-network standard (Deep Complex Networks style):
    normalise each position by the RMS modulus over features, then apply a
    learnable complex gain.  It replaces ComplexLayerNorm in VecCRBlock:
    no mean subtraction keeps the analytic-signal structure of the field.
    """

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.gamma_r = nn.Parameter(torch.ones(d))
        self.gamma_i = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, z):
        rms = z.abs().square().mean(dim=-1, keepdim=True).sqrt()
        z_hat = z / (rms + self.eps)
        gamma = torch.complex(self.gamma_r, self.gamma_i)
        return z_hat * gamma


class ComplexFFN(nn.Module):
    """Complex FFN: ComplexLinear -> split GELU -> ComplexLinear."""

    def __init__(self, d_model: int, expansion: int = 4):
        super().__init__()
        self.fc1 = ComplexLinear(d_model, d_model * expansion)
        self.fc2 = ComplexLinear(d_model * expansion, d_model)
        self.act = nn.GELU()

    def forward(self, z):
        h = self.fc1(z)
        h = torch.complex(self.act(h.real), self.act(h.imag))
        return self.fc2(h)


class GeoFFN(nn.Module):
    """CR-hypersurface FFN: complex hyperplanes + phase-preserving collapse.

    The Euclidean FFN (ComplexFFN) expands d -> 4d -> d, materialising a 4d
    intermediate and mixing with a non-holomorphic split-GELU.  This replaces it
    with ``rounds`` of a full complex-hyperplane mix (ComplexLinear d->d, still
    O(d^2) degrees of freedom — each output is a complex hyperplane
    <w_j, z> = 0) each followed by a phase-preserving radial collapse
    (modReLU/softmodrelu): channels whose modulus falls under the per-round
    threshold are annihilated, so the field collapses onto a smaller CR
    sub-surface at each round.  No 4x expansion -> the intermediate is d complex
    (2d real) instead of 4d real, ~2-4x less memory, while keeping full O(d^2)
    channel mixing (unlike the O(d) circulant mixer of cr-geo).
    """

    def __init__(self, d_model: int, rounds: int = 2, nl: str = "softmodrelu"):
        super().__init__()
        from .piecewise_cr_attention import _apply_nl
        self.rounds = int(rounds)
        self.nl = nl
        self.mix = nn.ModuleList([ComplexLinear(d_model, d_model)
                                  for _ in range(self.rounds)])
        self.bias = nn.Parameter(torch.zeros(self.rounds, d_model))
        self._apply_nl = _apply_nl

    def forward(self, z):
        h = z
        for k in range(self.rounds):
            h = self.mix[k](h)
            h = self._apply_nl(h, self.bias[k], self.nl)
        return h

    def collapse_ratio(self, z):
        """Fraction of channels annihilated at each round (|z| under threshold)."""
        with torch.no_grad():
            h = z
            ratios = []
            for k in range(self.rounds):
                h = self.mix[k](h)
                mag = h.abs()
                ratios.append((mag <= -self.bias[k]).float().mean().item())
                h = self._apply_nl(h, self.bias[k], self.nl)
            return ratios
