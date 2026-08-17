"""CR-NN losses: CR-Sobolev + ∂̄_b energy regularisation (docs/math.md §6)."""

from .cr_sobolev import CRSobolevLoss, dbar_energy, cr_sobolev_norm
from .combined import CombinedLoss
from .sobolev_embedding import SobolevEmbeddingLoss
from .collapse import CollapseLoss

__all__ = ["CRSobolevLoss", "dbar_energy", "cr_sobolev_norm",
           "CombinedLoss", "SobolevEmbeddingLoss", "CollapseLoss"]
