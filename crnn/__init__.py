"""CR-NN: CR-geometry neural networks.

Replaces Transformer attention with the Szegő projection on the Heisenberg
group, trained with a CR-Sobolev loss. See ``docs/math.md`` for the canonical
mathematical definitions; this package is the implementation of those.

The public entry points are intentionally thin during the research phase:
everything is importable, but stable APIs are not promised until M3.
"""

__version__ = "0.1.0"

# Backend selection is deferred until first use to avoid forcing a torch import
# at package import time (keeps MLX-only environments usable). See backend.py.
from . import backend  # noqa: F401

__all__ = ["backend", "__version__"]
