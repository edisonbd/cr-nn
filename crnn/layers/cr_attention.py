"""CR-Attention: the Szegő-projection-based attention replacement.

Implements docs/math.md §3.1. This layer replaces softmax attention with a
geometrically-motivated aggregator:

    1. embed tokens onto the H_p grid (sequence of length N = p^3, prime p)
    2. flat Szegő projection  Π_flat f = S_flat * f   (group conv, O(N^{4/3}))
    3. (optional) truncated curvature perturbation  Σ ε_j L_j[Π_flat f]
    4. ∂̄_b gating: suppress the non-holomorphic part (info -> complex dim)
    5. complex-valued linear mix -> real output

The Szegő projection is the CR analogue of "attending to the holomorphic
component". We do NOT claim this is a special case of softmax attention
(docs/math.md §8 / assumptions.md): it is a different aggregator whose kernel
comes from the manifold's spectral structure rather than an inner product.

Key engineering note: the speed probe (experiments/02_speedup_probe/REPORT.md)
showed crossover with softmax at N ≈ 10K. So this layer is intended for
long-context regimes; at short N it will be slower than softmax attention
without benefit. Callers should pick p so that N = p^3 matches their context
length (p prime: 23 -> N=12167, 29 -> N=24389).

Batching convention
-------------------
The input sequence (B, N, d) is reshaped to (B*d, p, p, p) so each channel
is an independent scalar field on H_p — group convolution acts per channel.
This is the same packing the speed probe used.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..geometry.operators import szego_kernel_flat
from .cr_attention_backend import cr_group_convolve


def _nearest_prime(n: int) -> int:
    """Smallest prime >= n (the grid resolution). p must be prime (R7)."""
    if n < 2:
        return 2
    n = int(n)
    while True:
        if _is_prime(n):
            return n
        n += 1


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    f = 3
    while f * f <= n:
        if n % f == 0:
            return False
        f += 2
    return True


class CRAttention(nn.Module):
    """Szegő-projection attention replacement.

    Parameters
    ----------
    d_model : int
        Feature dimension (must equal the channel count packed into the grid).
    p : int or None
        Grid resolution per axis; N = p^3 must be prime p. If None, picked
        from seq_len via _nearest_prime(N^{1/3}).
    n_cr : int
        CR complex dimension (default 1; M2 validates n=1).
    M : int
        Curvature perturbation truncation order (0 = flat only). Default 0
        for M3; M5 ablation sweeps M.
    eta : float
        Korányi/Szegő kernel singularity regularisation (math.md §4.3).
    gate : bool
        If True, apply ∂̄_b gating (step 4). Default True.

    Shapes
    ------
    input  : (B, N, d_model) real, N = p^3
    output : (B, N, d_model) real
    """

    def __init__(self, d_model: int, p: int | None = None, n_cr: int = 1,
                 M: int = 0, eta: float = 1e-6, gate: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_cr = n_cr
        self.M = M
        self.eta = eta
        self.gate = gate
        self.p = p
        if p is not None:
            if not _is_prime(p):
                raise ValueError(f"p={p} must be prime (see docs/assumptions.md R7)")
        # learnable curvature amplitudes (math.md §5.2); init 0 -> start flat
        if M > 0:
            self.eps = nn.Parameter(torch.zeros(M))
        # output complex->real mix
        self.out_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, d = x.shape
        p = self.p or _nearest_prime(round(N ** (1 / 3)))
        if p ** 3 != N:
            # pad/truncate sequence to p^3
            x = self._pad_to_grid(x, p)
            N = p ** 3
        # pack (B, N, d) -> (B*d, p, p, p) complex
        f = x.permute(0, 2, 1).reshape(B * d, p, p, p).contiguous()
        f_c = f.to(torch.complex64)

        # ---- step 1: flat Szegő projection S_flat * f (group conv) ----
        # build kernel on the grid (cached on first call for this p)
        S = self._get_szego_kernel(p, f_c.device, f_c.dtype)
        out = cr_group_convolve(f_c, S, p)              # (B*d, p,p,p)

        # ---- step 2: curvature perturbation (M5; M3 keeps M=0) ----
        if self.M > 0:
            out = self._apply_perturbation(out, p)

        # ---- step 3: ∂̄_b gating ----
        if self.gate:
            dbar = self._dbar_energy(f_c, p)            # (B*d, p,p,p) real
            gate_w = torch.sigmoid(-dbar.real)          # suppress non-holomorphic
            out = out * gate_w.to(out.dtype)

        # ---- step 4: complex -> real mix ----
        out = out.reshape(B, d, N).permute(0, 2, 1)     # (B, N, d) complex
        out_real = torch.cat([out.real, out.imag], dim=-1)  # (B, N, 2d)
        return self.out_proj(out_real)

    # ----- helpers -----

    def _pad_to_grid(self, x, p):
        N_target = p ** 3
        B, N, d = x.shape
        if N == N_target:
            return x
        if N < N_target:
            pad = torch.zeros(B, N_target - N, d, device=x.device, dtype=x.dtype)
            return torch.cat([x, pad], dim=1)
        return x[:, :N_target, :]

    _kernel_cache: dict = {}

    def _get_szego_kernel(self, p, device, dtype):
        key = (p, self.n_cr, self.eta, str(device), str(dtype))
        if key not in self._kernel_cache:
            # evaluate flat Szegő kernel S(z,t) = (|z|² - i t)^{-(n+1)} on the
            # p×p×p grid (fftshifted coords so the identity is at the centre).
            coords = np.fft.fftfreq(p, d=1.0) * p       # signed, length p
            xx = coords.reshape(p, 1, 1)
            yy = coords.reshape(1, p, 1)
            tt = coords.reshape(1, 1, p)
            absz2 = xx ** 2 + yy ** 2
            w = absz2 - 1j * tt
            # Regularise the singularity at the origin (math.md §4.3).
            # Naive (w+eta) doesn't help when |w|<<eta; instead regularise the
            # modulus: S = conj(w)^{n+1} / (|w|² + eta)^{n+1}. This keeps the
            # correct phase everywhere and only softens the magnitude peak.
            n1 = self.n_cr + 1
            wsq = (w * np.conj(w)).real + self.eta      # |w|² + eta, real
            S = np.conj(w) ** n1 / (wsq ** n1)
            S = S.astype(np.complex64)
            self._kernel_cache[key] = torch.from_numpy(S).to(device=device, dtype=dtype)
        return self._kernel_cache[key]

    def _apply_perturbation(self, out, p):
        # L_j[f] = Δ_b^j (f): placeholder using Δ_b spectral weights.
        # Full implementation in M5 (curvature ablation); M3 runs flat (M=0).
        raise NotImplementedError(
            "curvature perturbation is M5 work; set M=0 for M3"
        )

    def _dbar_energy(self, f, p):
        """|∂̄_b f| energy, used for gating. Spectral, O(p^3 log p).

        A full ∂̄_b requires the (0,1)-type operator; for gating we only need
        a scalar energy, so we use |∇_H f|² (horizontal gradient squared) as
        a proxy — it vanishes iff f is CR (constant), same qualitative signal.
        """
        # horizontal gradient via spectral derivative: |∂_x f|² + |∂_y f|²
        # using FFT-based derivative along x (axis -3) and y (axis -2).
        fx = _spectral_deriv(f, axis=-3, p=p)
        fy = _spectral_deriv(f, axis=-2, p=p)
        return (fx * fx.conj() + fy * fy.conj()).real


def _spectral_deriv(f: torch.Tensor, axis: int, p: int) -> torch.Tensor:
    """Derivative along `axis` via FFT: d/dx ↔ i·k. Complex-valued."""
    k = torch.fft.fftfreq(p, d=1.0) * p
    shape = [1] * f.ndim
    shape[axis] = p
    k = k.view(shape).to(f.dtype).to(f.device)
    f_hat = torch.fft.fft(f, dim=axis)
    return torch.fft.ifft(1j * k * f_hat, dim=axis)
