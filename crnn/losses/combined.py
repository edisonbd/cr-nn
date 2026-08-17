"""Combined CE + CR-Sobolev loss for the toy LM (M4).

The toy LM is a classification task, so the primary loss is cross-entropy on
logits.  The CR-Sobolev loss from docs/math.md section 6 is a *representation*
regulariser (assumption A3): it pushes the model's hidden field towards the
holomorphic (CR) subspace, compressing information into the complex
dimension.

For the regulariser we apply CRSobolevLoss to the packed hidden grid with
``target = hidden.detach()`` by default.  With that target the Sobolev
mismatch term vanishes identically and the regulariser reduces to
mu * ||dbar_b h||^2 -- the pure holomorphic-compression penalty.  This is the
cleanest M4 version of "CR-Sobolev as representation regulariser"; the
full ||out - y||^2_S regression form is exercised by the unit tests and can
be enabled with ``sobolev_target="embedding"`` (targets the token embeddings).

Parameters
----------
ce_weight : float   weight of the cross-entropy term
cr_weight : float   weight of the CR regulariser (0 disables it)
mu : float          dbar_b energy strength inside CRSobolevLoss
s : float           Sobolev order (default 1)
n : int             CR complex dimension (default 1)
sobolev_target : str  "detach" (default) | "embedding"
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cr_sobolev import CRSobolevLoss


class CombinedLoss(nn.Module):
    def __init__(self, ce_weight: float = 1.0, cr_weight: float = 0.0,
                 mu: float = 1e-3, s: float = 1.0, n: int = 1,
                 sobolev_target: str = "detach"):
        super().__init__()
        self.ce_weight = ce_weight
        self.cr_weight = cr_weight
        self.sobolev_target = sobolev_target
        self.cr_loss = CRSobolevLoss(mu=mu, s=s, n=n)

    def forward(self, logits: torch.Tensor, tokens: torch.Tensor,
                hidden_grid: torch.Tensor, p: int,
                embed_grid: torch.Tensor | None = None):
        """logits: (B, W, vocab); tokens: (B, W); hidden_grid: (B*d, p, p, p) complex."""
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), tokens.reshape(-1))
        total = self.ce_weight * ce
        stats = {"ce": ce.detach().item()}
        if self.cr_weight > 0 and hidden_grid is not None:
            if self.sobolev_target == "detach":
                target = hidden_grid.detach()
            elif self.sobolev_target == "embedding":
                if embed_grid is None:
                    raise ValueError("sobolev_target='embedding' requires embed_grid")
                target = embed_grid.detach()
            else:
                raise ValueError(f"unknown sobolev_target={self.sobolev_target!r}")
            cr = self.cr_loss(hidden_grid, target, p=p)
            total = total + self.cr_weight * cr
            stats["cr"] = cr.detach().item()
        return total, stats
