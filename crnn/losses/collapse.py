"""Collapse loss: task loss on the real dimension + fluid-collapse term.

User direction (2026-08-10): the loss should measure the *collapse* between
the real (target) dimension and the combined complex fluid:

    L = ce_weight * CE(logits, tokens)
      + col_weight * ( ||Re(h) - Re(t)||^2_{S_b} + ||Im(h)||^2 )
      + mu * ||dbar_b h||^2

where h is the model's complex fluid field (B*d, p, p, p) and t the complex
embedding field of the target tokens.  The first collapse term drives the
fluid's real part onto the target structure (CR-Sobolev weighted, so low
frequencies matter more); the ||Im(h)||^2 term penalises "uncollapsed"
imaginary energy; the dbar_b term keeps the fluid holomorphic (the Szego
collapse).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cr_sobolev import cr_sobolev_norm, dbar_energy


class CollapseLoss(nn.Module):
    def __init__(self, ce_weight: float = 1.0, col_weight: float = 1.0,
                 mu: float = 1e-3, s: float = 1.0, n: int = 1):
        super().__init__()
        self.ce_weight = ce_weight
        self.col_weight = col_weight
        self.mu = mu
        self.s = s
        self.n = n

    def forward(self, logits, tokens, hidden_grid, target_grid, p):
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                             tokens.reshape(-1))
        real_diff = (hidden_grid.real - target_grid.real).to(torch.complex64)
        sob = cr_sobolev_norm(real_diff, p=p, n=self.n, s=self.s).mean()
        im_energy = (hidden_grid.imag ** 2).sum(dim=(-3, -2, -1)).mean()
        dbar = dbar_energy(hidden_grid, p=p).mean()
        total = (self.ce_weight * ce
                 + self.col_weight * (sob + im_energy)
                 + self.mu * dbar)
        stats = {"ce": ce.detach().item(),
                 "sob": sob.detach().item(),
                 "im": im_energy.detach().item(),
                 "dbar": dbar.detach().item()}
        return total, stats
