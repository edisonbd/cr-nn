"""Validate szego=True: spectrum lambda=0 suppression + agreement with matrix Szego.

1. Print the diffusion base vs szego spectrum at lambda=0 (centre modes).
2. Scalar-szego spectral flow vs the true matrix Szego group convolution.
"""
import torch

from crnn.layers import PiecewiseCRAttention
from crnn.layers.cr_attention_backend import cr_group_convolve
from crnn.geometry.operators import szego_kernel_flat


def main():
    torch.manual_seed(0)
    dev = torch.device("cuda")
    p = 11
    d = 8

    # 1. spectrum structure: lambda=0 slice
    attn = PiecewiseCRAttention(d, p=p, n_flow=1, gate=False, mix=False,
                                nl="none", twist=False, szego=True).to(dev)
    S_hat = attn._szego_spectrum(p, dev, torch.complex64)  # (p,p,p) [xi,eta,lam]
    lam0 = S_hat[..., 0].abs().mean().item()          # lambda=0 slice
    lam_nz = S_hat[..., 1:].abs().mean().item()       # lambda!=0 slices
    print(f"szego spectrum: |S(lam=0)|={lam0:.4f}  |S(lam!=0)|={lam_nz:.4f}  "
          f"ratio={lam0/lam_nz:.5f}")

    # diffusion base for contrast (stage 0)
    attn_d = PiecewiseCRAttention(d, p=p, n_flow=1, gate=False, mix=False,
                                  nl="none", twist=False, szego=False).to(dev)
    with torch.no_grad():
        diff = torch.exp(-attn_d.t * 0)  # placeholder; recompute properly
    lam = torch.fft.fftfreq(p).to(dev)
    xi = torch.fft.fftfreq(p).to(dev)
    diffbase = torch.exp(-attn_d.t * (
        lam.reshape(1, 1, p).abs() + xi.reshape(p, 1, 1) ** 2
        + xi.reshape(1, p, 1) ** 2))
    print(f"diffusion base: |diff(lam=0)|={diffbase[..., 0].abs().mean():.4f}  "
          f"|diff(lam!=0)|={diffbase[..., 1:].abs().mean():.4f}")

    # 2. scalar szego vs matrix szego on a random field
    x = torch.randn(1, p ** 3, d, device=dev)
    # scalar szego: n_flow=1, no twist/nl/gate/mix => pure spectral project
    out_scalar = attn(x)                        # (1, p^3, d) complex
    # matrix szego: group convolution with the flat Szego kernel
    f = x.permute(0, 2, 1).reshape(d, p, p, p).to(torch.complex64)
    S = torch.from_numpy(szego_kernel_flat(p, 1, 1e-6)).to(dev, torch.complex64)
    out_matrix = cr_group_convolve(f, S, p)     # (d, p, p, p) complex
    out_matrix = out_matrix.reshape(d, p ** 3).permute(1, 0).unsqueeze(0)

    # cosine similarity per channel (scalar vs matrix)
    a = out_scalar.reshape(p ** 3, d)
    b = out_matrix.reshape(p ** 3, d)
    cos = (a.real * b.real + a.imag * b.imag).sum(0) / (
        a.abs().square().sum(0).sqrt() * b.abs().square().sum(0).sqrt() + 1e-8)
    print(f"scalar-szego vs matrix-szego: cos-sim mean={cos.mean():.4f} "
          f"min={cos.min():.4f}")

    # energy ratio: how much of the scalar output lives at lambda=0
    fhat = torch.fft.fft(out_scalar.reshape(d, p, p, p), dim=-1)
    fhat = torch.fft.fft2(fhat, dim=(-3, -2))
    e0 = fhat[..., 0].abs().square().sum().item()
    eall = fhat.abs().square().sum().item()
    print(f"scalar-szego output: energy at lam=0 = {100*e0/eall:.2f}% of total")


if __name__ == "__main__":
    main()
