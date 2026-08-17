"""Same phase test but with a near-exact (tiny eta) Szegő kernel.

eta regularises the singularity; eta -> 0 recovers the exact holomorphic
projector (which annihilates anti-holomorphic w^m, m>0).  Check whether the
clean annihilation of the "dimension-collapse region" emerges as eta -> 0.
"""

import numpy as np
import torch

from crnn.layers.cr_attention_backend import cr_group_convolve
from crnn.layers.vec_cr_attention import build_szego_kernel


def main():
    p = 11
    coords = np.fft.fftfreq(p, d=1.0) * p
    xx = coords.reshape(p, 1, 1)
    yy = coords.reshape(1, p, 1)
    tt = coords.reshape(1, 1, p)
    w = (xx ** 2 + yy ** 2) - 1j * tt
    wc = w.conj()
    f_holo = torch.from_numpy(wc.astype(np.complex64)).unsqueeze(0)   # conj(w)
    f_anti = torch.from_numpy(w.astype(np.complex64)).unsqueeze(0)    # w
    for eta in (1e-6, 1e-9, 1e-12):
        S = build_szego_kernel(p, 1, eta, "cpu", torch.complex64)
        oh = cr_group_convolve(f_holo, S, p)
        oa = cr_group_convolve(f_anti, S, p)
        eh = (oh.abs() ** 2).sum().item()
        ea = (oa.abs() ** 2).sum().item()
        print(f"eta={eta:.0e}: holo(conj w) energy={eh:.3e}  "
              f"anti(w) energy={ea:.3e}  ratio anti/holo={ea/eh:.3e}")


if __name__ == "__main__":
    main()
