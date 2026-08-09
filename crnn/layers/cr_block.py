"""CR-Block: CR-Attention + CR-FFN with residual + layernorm.

Mirrors a standard Transformer block but with CR-Attention replacing the
softmax attention. The block operates on real tensors of shape (B, N, d_model);
CR-Attention handles the complex embedding internally.

Note on the residual: because CR-Attention's output projection maps
complex→real (2d→d), the residual stays in real d_model space throughout —
no complex residual bookkeeping needed. This keeps the block a drop-in
replacement for nn.TransformerEncoderLayer in the toy LM (M4).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .cr_attention import CRAttention
from .cr_feedforward import CRFeedForward


class CRBlock(nn.Module):
    """One CR-NN block: LayerNorm → CR-Attention → residual →
    LayerNorm → CR-FFN → residual."""

    def __init__(self, d_model: int, p: int | None = None, n_cr: int = 1,
                 M: int = 0, eta: float = 1e-6, gate: bool = True,
                 ff_expansion: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CRAttention(d_model, p=p, n_cr=n_cr, M=M, eta=eta, gate=gate)
        self.norm2 = nn.LayerNorm(d_model)
        # CR-FFN works on 2*d (real||imag); wrap with d<->2d projection
        self.ffn = CRFeedForward(d_model, expansion=ff_expansion)
        self.ffn_proj_in = nn.Linear(d_model, 2 * d_model)
        self.ffn_proj_out = nn.Linear(2 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # attention sublayer
        x = x + self.attn(self.norm1(x))
        # FFN sublayer: d -> 2d (real||imag) -> FFN -> 2d -> d
        h = self.ffn_proj_in(self.norm2(x))
        h = self.ffn(h)
        x = x + self.ffn_proj_out(h)
        return x
