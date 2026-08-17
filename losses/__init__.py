"""CR-NN losses: CR-Sobolev + ∂̄_b energy regularisation (docs/math.md §6)."""

from .cr_sobolev import CRSobolevLoss, dbar_energy, cr_sobolev_norm

__all__ = ["CRSobolevLoss", "dbar_energy", "cr_sobolev_norm"]
