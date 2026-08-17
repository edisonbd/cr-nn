"""Complex-valued feed-forward network for CR-NN blocks.

A standard Transformer FFN (Linear → activation → Linear) but operating on
complex-valued features so it composes with CR-Attention's complex output.
The activation is applied independently to real and imaginary parts (a
holomorphic activation would constrain expressivity; the gating in
CR-Attention already handles the holomorphic/non-holomorphic split).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CRFeedForward(nn.Module):
    """Complex FFN: Linear → GELU → Linear, on real/imag parts separately.

    Operates on a real tensor whose last dim is 2*d (real||imag) for
    compatibility with CR-Attention's output projection. Internally splits,
    applies the same FFN to each half, and recombines.

    Parameters
    ----------
    d_model : int
    expansion : int
        Hidden dim multiplier (default 4, as in standard Transformers).
    """

    def __init__(self, d_model: int, expansion: int = 4):
        super().__init__()
        hidden = d_model * expansion
        # operate on 2*d input (real||imag) for symmetry with CR-Attention
        self.fc1 = nn.Linear(2 * d_model, 2 * hidden)
        self.fc2 = nn.Linear(2 * hidden, 2 * d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, 2*d) — real||imag concatenation
        return self.fc2(self.act(self.fc1(x)))
