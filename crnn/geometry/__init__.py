"""CR-NN geometry package: Heisenberg group, operators, spectrum.

See ``docs/math.md`` for the canonical definitions; these modules are their
implementation. Everything here is backend-agnostic and takes an explicit
``backend`` argument so the same code runs under torch and (later) MLX.
"""

from .heisenberg import HeisenbergGrid, group_multiply, inverse, koranyi_norm
from .operators import (
    Delta_b,
    Dbard,
    X_vector_fields,
    Y_vector_fields,
    koranyi_kernel,
    szego_kernel_flat,
)
from .spectrum import HermiteLaguerreBasis, sub_laplacian_eigenvalues

__all__ = [
    "HeisenbergGrid",
    "group_multiply",
    "inverse",
    "koranyi_norm",
    "X_vector_fields",
    "Y_vector_fields",
    "Dbard",
    "Delta_b",
    "koranyi_kernel",
    "szego_kernel_flat",
    "HermiteLaguerreBasis",
    "sub_laplacian_eigenvalues",
]
