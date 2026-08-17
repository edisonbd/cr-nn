"""Verify the "dimension-collapse" causal mask mechanism.

Idea: instead of a hard triangular mask (which breaks the group convolution),
collapse the FUTURE positions to a single degenerate value (dimension collapse).
The Szegő kernel S(w)=conj(w)^{n+1}/(|w|^2+eta)^{n+1} has a phase
conj(w)^{n+1}=e^{-i(n+1)theta} whose mean over the group is ~0 (phase
averaging).  So a *constant* (collapsed) future contributes
    c * sum_{h in future} S(h^{-1} g)  ~  c * (phase-averaged) ~ 0,
i.e. the collapsed future is annihilated by the Szegő projection — a geometric
soft-mask that does not hard-zero the group structure.

This script checks that the output at PAST positions is (approximately)
independent of the FUTURE constant, i.e. the collapse effectively masks the
future.
"""

import torch

from crnn.layers.cr_attention_backend import cr_group_convolve
from crnn.layers.vec_cr_attention import build_szego_kernel


def main():
    for p in (5, 7, 11):
        S = build_szego_kernel(p, 1, 1e-6, "cpu", torch.complex64)
        # field: "past" = first half along c-axis is random signal, "future" =
        # second half along c-axis collapsed to a constant c0.
        f = torch.zeros(1, p, p, p, dtype=torch.complex64)
        past = torch.randn(1, p, p, p // 2, dtype=torch.complex64)
        f[..., : p // 2] = past
        for c0 in (0.0 + 0j, 1.0 + 2j, -3.0 + 0.5j):
            f[..., p // 2:] = c0   # collapse the future to a constant
            out = cr_group_convolve(f, S, p)
            out_past = out[..., : p // 2]
            # the past output should be ~independent of c0
            if c0 == 0.0 + 0j:
                ref = out_past.clone()
            else:
                leak = (out_past - ref).abs().max().item()
                scale = ref.abs().max().item()
                print(f"p={p} c0={c0}: future leak into past = "
                      f"{leak:.3e} (rel {leak/scale:.3e})")


if __name__ == "__main__":
    main()
