"""CR-NN neural network layers.

The CR-Attention layer is the attention replacement (docs/math.md §3.1);
CR-FFN and CR-Block compose it into a usable transformer-like stack.
"""

from .cr_attention import CRAttention
from .cr_feedforward import CRFeedForward
from .cr_block import CRBlock
from .complex_nn import ComplexLinear, ComplexLayerNorm, ComplexRMSNorm, ComplexFFN
from .vec_cr_attention import VecCRAttention, VecCRBlock
from .fluid_attention import FluidCRAttention
from .piecewise_cr_attention import (
    PiecewiseCRAttention, PiecewiseCRBlock, complex_modrelu)
from .geo_cr import GeoCRBlock, GeoChannelMix, GeoNorm

__all__ = ["CRAttention", "CRFeedForward", "CRBlock",
           "ComplexLinear", "ComplexLayerNorm", "ComplexRMSNorm", "ComplexFFN",
           "VecCRAttention", "VecCRBlock", "FluidCRAttention",
           "PiecewiseCRAttention", "PiecewiseCRBlock", "complex_modrelu",
           "GeoCRBlock", "GeoChannelMix", "GeoNorm"]
