"""Verify the correct "future collapse": map the future to the image of dbar_b^†
(the higher Landau levels), which the Szegő projection annihilates.

dbar_b = (1/2)(X + iY),  X = d_a + 2 b d_c,  Y = d_b - 2 a d_c.
Its adjoint dbar_b^† = (1/2)(-X + iY).  The Szegő projection S projects onto
ker dbar_b (the k=0 Landau level), so S annihilates dbar_b^†-exact fields
(f = dbar_b^† g, the k>=1 levels).  This is the geometric "dimension collapse":
map the FUTURE to dbar_b^† g and S kills it.

We compute dbar_b^† g spectrally (FFT derivatives) and check S(dbar_b^† g) ~ 0
vs S(g) != 0.
"""

import torch

from crnn.layers.cr_attention_backend import cr_group_convolve
from crnn.layers.vec_cr_attention import build_szego_kernel


def spectral_deriv(f, axis, p):
    k = torch.fft.fftfreq(p, d=1.0) * p
    shape = [1] * f.ndim
    shape[axis] = p
    k = k.view(shape).to(f.dtype).to(f.device)
    fh = torch.fft.fft(f, dim=axis)
    return torch.fft.ifft(1j * k * fh, dim=axis)


def dbar_b_dag(f, p):
    """dbar_b^† f = (1/2)(-d_a f - 2 b d_c f + i d_b f - 2i a d_c f)."""
    coords = torch.arange(p, dtype=torch.float32, device=f.device)
    a = coords.view(p, 1, 1)
    b = coords.view(1, p, 1)
    da = spectral_deriv(f, axis=-3, p=p)
    db = spectral_deriv(f, axis=-2, p=p)
    dc = spectral_deriv(f, axis=-1, p=p)
    X = da + 2 * b * dc
    Y = db - 2 * a * dc
    return 0.5 * (-X + 1j * Y)


def main():
    for p in (5, 7, 11):
        S = build_szego_kernel(p, 1, 1e-6, "cpu", torch.complex64)
        g = torch.randn(1, p, p, p, dtype=torch.complex64)
        f_collapse = dbar_b_dag(g, p)              # future = dbar_b^† g
        out_g = cr_group_convolve(g, S, p)
        out_c = cr_group_convolve(f_collapse, S, p)
        surv_g = (out_g.abs() ** 2).sum().item()
        surv_c = (out_c.abs() ** 2).sum().item()
        print(f"p={p}: S(g) energy={surv_g:.3e}  S(dbar_b^† g) energy={surv_c:.3e}"
              f"  ratio={surv_c/(surv_g+1e-30):.3e}")


if __name__ == "__main__":
    main()
