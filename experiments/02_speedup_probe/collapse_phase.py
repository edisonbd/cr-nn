"""Characterise which field phases the Szegő projection annihilates.

S(w)=conj(w)^{n+1}/(|w|^2+eta)^{n+1} is (up to eta) the holomorphic projector
onto ker dbar_b = the k=0 Landau level.  Its phase is e^{-i(n+1)theta}, so in
the theta (phase) domain the convolution f*S is resonant only at theta=+(n+1)
(f ~ w^{n+1}).  Everything else should be (approximately) annihilated — this is
the "dimension-collapse region" for the causal mask: map the FUTURE to an
anti-holomorphic phase and S kills it.

We test on the FULL grid (symmetric) which phases survive.
"""

import numpy as np
import torch

from crnn.layers.cr_attention_backend import cr_group_convolve
from crnn.layers.vec_cr_attention import build_szego_kernel


def main():
    p = 11
    S = build_szego_kernel(p, 1, 1e-6, "cpu", torch.complex64)
    coords = np.fft.fftfreq(p, d=1.0) * p
    xx = coords.reshape(p, 1, 1)
    yy = coords.reshape(1, p, 1)
    tt = coords.reshape(1, 1, p)
    w = (xx ** 2 + yy ** 2) - 1j * tt          # w = |z|^2 - i t
    wc = w.conj()
    fields = {
        "const(1)": torch.ones(1, p, p, p, dtype=torch.complex64),
        "conj(w) (holo)": torch.from_numpy((wc).astype(np.complex64)).unsqueeze(0),
        "w (anti-holo)": torch.from_numpy((w).astype(np.complex64)).unsqueeze(0),
        "conj(w)^2": torch.from_numpy((wc ** 2).astype(np.complex64)).unsqueeze(0),
        "w^2": torch.from_numpy((w ** 2).astype(np.complex64)).unsqueeze(0),
    }
    for name, f in fields.items():
        out = cr_group_convolve(f, S, p)
        # survival = energy of output relative to input
        surv = (out.abs() ** 2).sum().item()
        print(f"{name:>18}: out energy = {surv:.3e}")


if __name__ == "__main__":
    main()
