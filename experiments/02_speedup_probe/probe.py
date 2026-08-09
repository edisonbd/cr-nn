"""Speed probe: CR group convolution vs softmax attention.

This is the M3 early-stop gate (docs/assumptions.md, milestone M3). The
project's "faster / less memory" claim rests on CR group convolution being
O(p^4) = O(N^{4/3}) vs softmax attention's O(N^2 d). If the constant factor
is too large, the asymptotic win never materialises at practical N and we
stop to rethink.

What we measure, for a sweep of N = p^3 (p prime):
  1. softmax attention:   Q K^T softmax V        — O(N^2 d) time, O(N^2) memory
  2. CR group conv (ref): the M2 loop-based FFT path — correct, slow constant
  3. CR group conv (vec): vectorised torch matmul path — the realistic constant

We report wall time and peak memory (CUDA only; CPU memory is estimated).
The headline number is the crossover N where CR-vec beats softmax, and the
speedup ratio at the largest N tested.

CAVEAT: the M2 FFT is not yet the Diaconis–Rockmore fast path (it's the
correct O(p^4) matmul path, not O(p^3 log p)). So this probe gives a LOWER
bound on CR speedup — the true fast path only improves on it. If even this
lower bound beats softmax at practical N, the project's speed claim holds.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import numpy as np
import torch

from crnn.transforms.heisenberg_fft import (
    group_convolve as group_convolve_ref,
    heisenberg_fft,
    heisenberg_ifft,
    _check_prime,
)


# ---------------------------------------------------------------------------
# softmax attention baseline
# ---------------------------------------------------------------------------

def softmax_attention(q, k, v):
    """Standard scaled dot-product attention. q,k,v: (B, N, d)."""
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-1, -2)) / (d ** 0.5)   # (B, N, N)
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)                                  # (B, N, d)


# ---------------------------------------------------------------------------
# CR group convolution, vectorised torch path
# ---------------------------------------------------------------------------

def group_convolve_vec(f, g, p):
    """Vectorised CR group convolution via batched matmul.

    Same math as group_convolve_ref (the M2 correct path) but:
      - forward FFT: the (b,c) 2D transform and the per-lam phase sums are
        done with torch.fft2 / batched einsum instead of python loops;
      - the (p-1) matrix products are a single batched matmul;
      - inverse FFT: vectorised the same way.

    f, g: (B, p, p, p) complex64. Returns (B, p, p, p) complex64.

    This is the path whose constant factor actually determines whether CR
    beats softmax in practice. The reference loop path is only for parity.
    """
    B = f.shape[0]
    _check_prime(p)
    omega = torch.exp(torch.tensor(2j * torch.pi / p, dtype=f.dtype, device=f.device))
    G = p ** 3

    def fft_forward(x):
        # lam=0 chars: sum over c, then 2D positive-exp transform over (a,b).
        x_ab = x.sum(dim=-1)                                  # (B, a, b)
        chars = torch.conj(torch.fft.fft2(torch.conj(x_ab)))  # (B, p, p)
        # lam=1..p-1 matrices: fhat(lam)_{u,v} = sum_{b,c} f(v-u,b,c) w^{lam c} w^{lam b u}
        u = torch.arange(p, device=x.device)
        v = torch.arange(p, device=x.device)
        a_idx = (v[None, :] - u[:, None]) % p                 # (u, v) -> a
        gathered = x[:, a_idx, :, :]                          # (B, u, v, b, c)
        T = torch.conj(torch.fft.fft2(torch.conj(gathered), dim=(-2, -1)))  # (B,u,v,bhat,chat)
        lam = torch.arange(1, p, device=x.device)             # (p-1,)
        bfreq = (lam[:, None] * u[None, :]) % p               # (lam, u) -> bhat
        cfreq = lam                                            # (lam,) -> chat
        # gather out[b, lam, u, v] = T[b, u, v, bfreq[lam,u], cfreq[lam]]
        # build meshgrid indices (all shape B, p-1, p, p)
        bz, lm, uu, vv = torch.meshgrid(
            torch.arange(B, device=x.device), torch.arange(p - 1, device=x.device),
            u, v, indexing="ij")
        bf_sel = bfreq[lm, uu]                                 # (B, lam, u, v)
        # cfreq[lam] is just lam itself (constant per lam); broadcast.
        cf_sel = cfreq.view(1, p - 1, 1, 1).expand(B, p - 1, p, p)
        out = T[bz, uu, vv, bf_sel, cf_sel]                    # (B, lam, u, v)
        return chars, out

    chars_f, mat_f = fft_forward(f)
    chars_g, mat_g = fft_forward(g)

    # convolution theorem: chars pointwise, matrices batched matmul (f LEFT)
    conv_chars = chars_f * chars_g
    conv_mat = torch.matmul(mat_f, mat_g)                     # (B, p-1, p, p)

    # ---- inverse, fully vectorised ----
    # f(a,b,c) = (1/p^3)[ chars_inv(a,b) + p * sum_{lam,u} D[lam,a,u] * P[lam,b,c,u] ]
    # where D[lam,a,u] = conv_mat[lam, u, (u+a)%p]  (gather the v=u+a diagonal)
    #       P[lam,b,c,u] = omega^{-lam*(c + b*u)}
    u_arr = torch.arange(p, device=f.device)
    lam_arr = torch.arange(1, p, device=f.device)             # (p-1,)

    # chars inverse: 2D positive-exponent inverse = conj(ifft2(conj))*p^2
    chars_inv = torch.conj(torch.fft.ifft2(torch.conj(conv_chars), dim=(-2, -1))) * (p * p)
    # chars_inv: (B, p, p) over (a, b). Broadcast to (B, p, p, p) over (a,b,c).
    out = chars_inv.unsqueeze(-1).expand(B, p, p, p).clone()

    # gather diagonal D[lam, a, u] = conv_mat[:, lam, u, (u+a)%p]
    # conv_mat: (B, p-1, p, p) = (B, lam, u, v). We want, for each a, v=(u+a)%p.
    a_idx = torch.arange(p, device=f.device)
    v_for_a = (u_arr[None, :] + a_idx[:, None]) % p           # (a, u) -> v
    # D shape (B, lam, a, u): gather conv_mat along v-axis with v_for_a
    # conv_mat[:, lam, u, v] -> take v = v_for_a[a, u]
    # index shape for gather on dim=-1: (B, p-1, p, 1) per a... build all a at once.
    # D[b, lam, a, u] = conv_mat[b, lam, u, v_for_a[a,u]]
    # Use advanced indexing:
    bb_idx, lam_idx, uu_idx, aa_idx = torch.meshgrid(
        torch.arange(B, device=f.device), torch.arange(p - 1, device=f.device),
        torch.arange(p, device=f.device), torch.arange(p, device=f.device),
        indexing="ij")
    D = conv_mat[bb_idx, lam_idx, uu_idx, v_for_a[aa_idx, uu_idx]]  # (B, lam, u, a) -> reorder
    D = D.permute(0, 1, 3, 2)                                  # (B, lam, a, u)

    # phase P[lam, b, c, u] = omega^{-lam*(c + b*u)}
    b_grid, c_grid, u_grid = torch.meshgrid(
        a_idx, a_idx, u_arr, indexing="ij")                    # each (p, p, p) = (b,c,u)
    expo = -lam_arr[:, None, None, None] * (c_grid[None] + b_grid[None] * u_grid[None])  # (lam,b,c,u)
    P = omega ** expo.to(torch.complex64)                      # (lam,b,c,u)

    # matrix contribution: p * sum_{lam,u} D[z,lam,a,u] * P[lam,b,c,u]  -> (z,a,b,c)
    # use z for batch to avoid clashing with the b-axis.
    mat_contrib = p * torch.einsum("zlau,lbcu->zabc", D, P)    # (B, a, b, c)
    out = out + mat_contrib
    out = out / G
    return out


# ---------------------------------------------------------------------------
# benchmark harness
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    name: str
    p: int
    N: int
    time_ms: float
    mem_bytes: int


def _peak_mem_bytes():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated()
    return 0  # CPU: not tracked here


def _reset_mem():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def bench_softmax(N, d, B, device, repeat=3):
    torch.cuda.empty_cache() if device.type == "cuda" else None
    _reset_mem()
    q = torch.randn(B, N, d, device=device) / (d ** 0.5)
    k = torch.randn(B, N, d, device=device) / (d ** 0.5)
    v = torch.randn(B, N, d, device=device)
    # warmup
    for _ in range(2):
        _ = softmax_attention(q, k, v)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        _ = softmax_attention(q, k, v)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / repeat * 1000
    return BenchResult("softmax_attn", 0, N, dt, _peak_mem_bytes())


def bench_cr(p, d, B, device, repeat=3, ref=False):
    """CR group conv. d is folded into batch B (CR acts per-channel)."""
    _check_prime(p)
    N = p ** 3
    torch.cuda.empty_cache() if device.type == "cuda" else None
    _reset_mem()
    # f, g: (B*d, p, p, p) — treat each channel as an independent field.
    f = torch.randn(B * d, p, p, p, dtype=torch.complex64, device=device)
    g = torch.randn(B * d, p, p, p, dtype=torch.complex64, device=device)
    fn = (lambda f, g, p: group_convolve_ref(f, g)) if ref else group_convolve_vec
    # warmup
    for _ in range(1 if ref else 2):
        _ = fn(f, g, p)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        _ = fn(f, g, p)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / repeat * 1000
    return BenchResult("cr_conv_ref" if ref else "cr_conv_vec", p, N, dt, _peak_mem_bytes())


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"torch: {torch.__version__}")
    print()

    # Sweep primes. N = p^3. softmax is O(N^2 d), CR is O(p^4) per channel.
    # We pick d=64, B=1 to keep softmax's N^2 matrix tractable on CPU.
    d = 64
    B = 1
    primes = [3, 5, 7, 11, 13, 17, 19, 23, 29]
    print(f"{'p':>4} {'N=p^3':>8} {'softmax(ms)':>12} {'cr_vec(ms)':>12} "
          f"{'cr_ref(ms)':>12} {'speedup':>8}")
    print("-" * 64)

    results = []
    for p in primes:
        N = p ** 3
        # softmax N^2 matrix: N=24389 (p=29) => N^2 ~ 595M floats ~ 2.4GB.
        # Reduce softmax repeats at large N to keep runtime sane.
        sm_repeat = 3 if N < 5000 else (2 if N < 15000 else 1)
        if device.type == "cpu" and N > 27000:
            print(f"{p:>4} {N:>8}   (skipped: N^2 memory too large for CPU)")
            continue
        try:
            sm = bench_softmax(N, d, B, device, repeat=sm_repeat)
        except RuntimeError as e:
            print(f"{p:>4} {N:>8}   softmax OOM: {str(e)[:40]}")
            continue
        crv = bench_cr(p, d, B, device, repeat=3, ref=False)
        # ref path is slow; only run for small p
        crr = bench_cr(p, d, B, device, repeat=1, ref=True) if p <= 7 else None
        speedup = sm.time_ms / crv.time_ms if crv.time_ms > 0 else float("inf")
        crr_str = f"{crr.time_ms:>12.2f}" if crr else f"{'--':>12}"
        print(f"{p:>4} {N:>8} {sm.time_ms:>12.2f} {crv.time_ms:>12.2f} "
              f"{crr_str} {speedup:>8.2f}x")
        results.append((p, N, sm, crv, crr))

    print()
    print("Interpretation:")
    print("  speedup > 1 means CR group conv beats softmax attention at this N.")
    print("  cr_ref is the unoptimised M2 loop path (correctness reference);")
    print("  cr_vec is the vectorised torch path (realistic constant factor).")
    print("  Asymptotic: softmax O(N^2 d), CR O(N^{4/3}) — crossover expected")
    print("  at some N; if no crossover appears in this range, see report.")


if __name__ == "__main__":
    main()
