"""CR-NN transforms: Heisenberg FFT and Szegő projection.

These are the modules that deliver the O(N log N) complexity claim
(math.md §7.1, assumption A1). They reduce group convolution on H_p to
standard FFTs via the Diaconis–Rockmore decomposition.
"""

from .heisenberg_fft import heisenberg_fft, heisenberg_ifft, group_convolve
from .szego import szego_projection_flat, szego_projection_curved

__all__ = [
    "heisenberg_fft",
    "heisenberg_ifft",
    "group_convolve",
    "szego_projection_flat",
    "szego_projection_curved",
]
