"""Vectorised CR group convolution for the CR-Attention layer.

This is the production version of the speed-probe's group_convolve_vec
(experiments/02_speedup_probe/probe.py), moved into the package so the
CR-Attention layer can call it. It implements the same matrix-valued
Heisenberg FFT convolution theorem (docs/math.md §7.2), fully vectorised
with torch.fft2 / batched matmul / einsum — no python loops over p.

Parity against the reference loop path is checked in
experiments/02_speedup_probe (rel err ~1e-6, complex64).

Complexity: O(p^4) per channel (p-1 batched p×p matmuls + FFT), i.e.
O(N^{4/3}) for N = p^3 — the asymptotic win over softmax attention's O(N^2 d)
that the speed probe confirmed at N ≳ 10K.
"""

from __future__ import annotations

import torch

from ..transforms.heisenberg_fft import _check_prime


def cr_group_convolve(f: torch.Tensor, g: torch.Tensor, p: int) -> torch.Tensor:
    """Group convolution (f * g) on H_p via the matrix Fourier theorem.

    f : (B, p, p, p) complex64 — the signal (one scalar field per channel).
    g : (p, p, p) or (B, p, p, p) complex64 — the kernel. A 3-D kernel is
        broadcast across the batch (same kernel for every channel), which is
        the common case for the Szegő projection (one kernel per layer).
    Returns (B, p, p, p) complex64.

    (f*g)^(lam) = fhat(lam) @ ghat(lam)  (f LEFT), then inverse transform.
    """
    _check_prime(p)
    B = f.shape[0]
    # broadcast kernel to batch if needed
    if g.ndim == 3:
        g = g.unsqueeze(0).expand(B, *g.shape)            # (B, p, p, p)
    dtype = f.dtype
    device = f.device
    omega = torch.exp(torch.tensor(2j * torch.pi / p, dtype=dtype, device=device))
    G = p ** 3

    chars_f, mat_f = _fft_forward(f, p, B, dtype, device, omega)
    chars_g, mat_g = _fft_forward(g, p, B, dtype, device, omega)

    # convolution theorem
    conv_chars = chars_f * chars_g
    conv_mat = torch.matmul(mat_f, mat_g)                # (B, p-1, p, p)

    return _ifft_inverse(conv_chars, conv_mat, p, B, dtype, device, omega, G)


# ---------------------------------------------------------------------------
# forward / inverse (vectorised; mirrors probe.py, parity-verified)
# ---------------------------------------------------------------------------

def _fft_forward(x, p, B, dtype, device, omega):
    u = torch.arange(p, device=device)
    v = torch.arange(p, device=device)
    # lam=0 chars: sum over c, 2D positive-exp transform over (a,b)
    x_ab = x.sum(dim=-1)                                 # (B, a, b)
    chars = torch.conj(torch.fft.fft2(torch.conj(x_ab)))  # (B, p, p)
    # lam=1..p-1 matrices
    a_idx = (v[None, :] - u[:, None]) % p                # (u, v) -> a
    gathered = x[:, a_idx, :, :]                         # (B, u, v, b, c)
    T = torch.conj(torch.fft.fft2(torch.conj(gathered), dim=(-2, -1)))  # (B,u,v,bhat,chat)
    lam = torch.arange(1, p, device=device)
    bfreq = (lam[:, None] * u[None, :]) % p              # (lam, u) -> bhat
    cfreq = lam                                           # (lam,)
    bz, lm, uu, vv = torch.meshgrid(
        torch.arange(B, device=device), torch.arange(p - 1, device=device),
        u, v, indexing="ij")
    bf_sel = bfreq[lm, uu]
    cf_sel = cfreq.view(1, p - 1, 1, 1).expand(B, p - 1, p, p)
    out = T[bz, uu, vv, bf_sel, cf_sel]                  # (B, lam, u, v)
    return chars, out


def _ifft_inverse(conv_chars, conv_mat, p, B, dtype, device, omega, G):
    u_arr = torch.arange(p, device=device)
    lam_arr = torch.arange(1, p, device=device)
    a_idx = torch.arange(p, device=device)
    # chars inverse: positive-exp 2D inverse
    chars_inv = torch.conj(torch.fft.ifft2(torch.conj(conv_chars), dim=(-2, -1))) * (p * p)
    out = chars_inv.unsqueeze(-1).expand(B, p, p, p).clone()
    # diagonal gather D[b, lam, a, u] = conv_mat[b, lam, u, (u+a)%p]
    v_for_a = (u_arr[None, :] + a_idx[:, None]) % p      # (a, u) -> v
    bb_idx, lam_idx, uu_idx, aa_idx = torch.meshgrid(
        torch.arange(B, device=device), torch.arange(p - 1, device=device),
        torch.arange(p, device=device), torch.arange(p, device=device),
        indexing="ij")
    D = conv_mat[bb_idx, lam_idx, uu_idx, v_for_a[aa_idx, uu_idx]]
    D = D.permute(0, 1, 3, 2)                            # (B, lam, a, u)
    b_grid, c_grid, u_grid = torch.meshgrid(
        a_idx, a_idx, u_arr, indexing="ij")
    expo = -lam_arr[:, None, None, None] * (c_grid[None] + b_grid[None] * u_grid[None])
    P = omega ** expo.to(dtype)
    mat_contrib = p * torch.einsum("zlau,lbcu->zabc", D, P)
    out = (out + mat_contrib) / G
    return out
