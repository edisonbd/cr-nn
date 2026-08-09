"""CR-NN neural network layers.

The CR-Attention layer is the attention replacement (docs/math.md §3.1);
CR-FFN and CR-Block compose it into a usable transformer-like stack.
"""

from .cr_attention import CRAttention
from .cr_feedforward import CRFeedForward
from .cr_block import CRBlock

__all__ = ["CRAttention", "CRFeedForward", "CRBlock"]
