"""Correct variable test: Szegő projection annihilates anti-HOLOMORPHIC z̄^m?

The CR structure is about the complex coordinate z = x + iy (NOT the isotropic
w = |z|^2 - i t that appears in the kernel).  The Szegő projection projects onto
CR (holomorphic-in-z) functions and should annihilate anti-holomorphic z̄^m
(m>0).  Test f = z^m (holo) vs f = z̄^m (anti) survival under the group conv.

NOTE: f is a field on H_p; z = x + i y uses the (a,b) grid coords.  The group
convolution is f * S with S = conj(w)^{n+1}/|w|^{2(n+1)}, w = (a^2+b^2) - i c.
"""

import numpy as np
import torch

from crnn.layers.cr_attention_backend import cr_group_convolve
from crnn.layers.vec_cr_attention import build_szego_kernel


def main():
    p = 11
    S = build_szego_kernel(p, 1, 1e-6, "cpu", torch.complex64)
    coords = np.fft.fftfreq(p, d=1.0) * p
    a = coords.reshape(p, 1, 1)
    b = coords.reshape(1, p, 1)
    z = (a + 1j * b).repeat(p, axis=2)   # (p,p,p) complex coord z = x+iy
    zc = (a - 1j * b).repeat(p, axis=2)
    for m in (0, 1, 2, 3):
        f_holo = torch.from_numpy((z ** m).astype(np.complex64)).unsqueeze(0)
        f_anti = torch.from_numpy((zc ** m).astype(np.complex64)).unsqueeze(0)
        oh = cr_group_convolve(f_holo, S, p)
        oa = cr_group_convolve(f_anti, S, p)
        eh = (oh.abs() ** 2).sum().item()
        ea = (oa.abs() ** 2).sum().item()
        print(f"m={m}: holo(z^m) energy={eh:.3e}  anti(z̄^m) energy={ea:.3e}  "
              f"ratio anti/holo={ea/(eh+1e-30):.3e}")


if __name__ == "__main__":
    main()
