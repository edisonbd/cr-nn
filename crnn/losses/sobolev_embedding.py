"""CR-Sobolev as the *main* loss for the toy LM (CR-completion, pre-M5).

The M4 CombinedLoss only added the CR term as a small regulariser
(cr_weight=0.01, and with target=detach it reduced to the dbar_b energy).
This loss makes the CR-Sobolev norm the primary objective, as intended by
docs/math.md section 6:

    L = ce_weight * CE(logits, tokens)
      + so_weight * ( ||h - t||^2_{S_b^s} + mu * ||dbar_b h||^2 )

where h is the model's complex hidden field (B*d, p,p,p) and t is the
complex embedding field of the *target* tokens (detached).  The Sobolev
norm for s=1 is exact in the flat model: ||f||^2 + <f, Delta_b f> =
||f||^2 + ||grad_H f||^2 (cr_sobolev_norm), which is what makes the loss
spectrally weighted rather than Euclidean.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cr_sobolev import cr_sobolev_norm, dbar_energy


class SobolevEmbeddingLoss(nn.Module):
    def __init__(self, ce_weight: float = 0.1, so_weight: float = 1.0,
                 mu: float = 1e-3, s: float = 1.0, n: int = 1):
        super().__init__()
        self.ce_weight = ce_weight
        self.so_weight = so_weight
        self.mu = mu
        self.s = s
        self.n = n

    def forward(self, logits, tokens, hidden_grid, target_grid, p):
        """hidden_grid, target_grid: (B*d, p, p, p) complex."""
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                             tokens.reshape(-1))
        diff = hidden_grid - target_grid
        sob = cr_sobolev_norm(diff, p=p, n=self.n, s=self.s).mean()
        dbar = dbar_energy(hidden_grid, p=p).mean()
        total = (self.ce_weight * ce
                 + self.so_weight * (sob + self.mu * dbar))
        stats = {"ce": ce.detach().item(),
                 "sob": sob.detach().item(),
                 "dbar": dbar.detach().item()}
        return total, stats
