"""Piecewise-CR attention ("断点流形"): activation-segmented spectral flow.

User direction (2026-08-15): keep the *CR manifold structure* as the whole
replacement for softmax (no Euclidean fallback, no hybrid flat+curvature
mixing), stay away from the matrix scheme (the O(p^4) matrix-valued group
convolution), and recover the expressiveness lost by the diagonal spectral
flow by *cutting the manifold into many small manifolds with activation
functions*.

What this layer does
--------------------
A sequence of ``n_flow`` spectral stages.  Each stage is

    x --FFT3--> fh = W_k . fh        (diagonal complex filter, O(p^3 log p))
              --> fh = sigma(fh)     (radial breakpoint in the spectral domain)
              --> x  = IFFT3(fh)
              --> x  = sigma(x)      (radial breakpoint in the spatial domain)

then the usual residual complex channel mixer, complex per-channel gain and
the dbar_b gate complete the block (identical to VecCRAttention), so the
layer is a drop-in for it.

Why activation functions "cut the manifold"
-------------------------------------------
The diagonal (abelian) 3D FFT is a *commutative* operator; it misses the
H_p twist (the a*b shear in the group law).  The twist couples different
frequencies, so no single diagonal filter can express it (this is the
structural ceiling the FluidCRAttention hit).  A pointwise nonlinearity in
the *spatial* domain is, however, a nonlinear function of *all* Fourier
coefficients: it re-mixes the frequencies, which is exactly where the twist
re-enters.  Each activation sigma has breakpoints (for modReLU, the circle
|z| = -b in every complex fibre), and composing K stages x L layers
stratifies the field configuration space into exponentially many polyhedral
cells -- "infinitely many small manifolds", each carrying a local linear
chart of the CR structure.  This is the same universal-approximation
argument as the Fourier Neural Operator: alternating spectral-linear and
spatial-nonlinear steps is a dense operator class, here constrained to
O(K p^3 log p) and no p x p matrices.

The modReLU nonlinearity is phase-preserving (multiplies by a real radial
factor), so it respects the complex / CR structure: it thresholds the
modulus and never mixes the holomorphic phase with the real axis.

Cross-platform note: only torch.fft + pointwise ops are used (no einsum, no
batched matmul), so the same layer ports directly to MLX's fft/pointwise
primitives (M6).

Reference for modReLU: Arjovsky, Shah & Bengio, "Unitary Evolution Recurrent
Neural Networks", ICML 2016; Trabelsi et al., "Deep Complex Networks", ICLR
2018 (modReLU naming).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .complex_nn import ComplexLinear


def complex_modrelu(z: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Radial (phase-preserving) modReLU with a breakpoint circle |z| = -bias.

    out = relu(|z| + bias) * z / |z|.
    For bias < 0 the disk |z| < -bias is zeroed, stratifying each complex
    fibre into inside/outside cells.  For bias >= 0 it is the identity on all
    non-zero z (and 0 at z=0).
    """
    mag = z.abs()
    scale = torch.relu(mag + bias) / (mag + 1e-8)
    return z * scale


def _split_gelu(z: torch.Tensor) -> torch.Tensor:
    """Non-holomorphic split GELU on (real, imag), matching ComplexFFN."""
    return torch.complex(torch.nn.functional.gelu(z.real),
                         torch.nn.functional.gelu(z.imag))


def complex_radial(z: torch.Tensor) -> torch.Tensor:
    """Smooth, phase-preserving radial contraction (CR-friendly, no breakpoint).

    out = z * tanh(|z|) / |z|.
    This is a *smooth* modulus map (|z| -> tanh|z|, bounded in (0,1)) with no
    breakpoint and no learned bias: it never introduces a sharp "activation
    point" that distorts the complex surface, and it preserves the phase
    (the holomorphic/CR structure).  It acts as a built-in regulariser that
    suppresses the memorisation channel responsible for overfitting on noisy
    real text, while keeping the field bounded and CR-structured.
    """
    mag = z.abs()
    return z * torch.tanh(mag) / (mag + 1e-8)


def complex_softmodrelu(z: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Smooth (softplus) modReLU: no hard breakpoint, phase-preserving.

    out = z * softplus(|z| + bias) / (|z| + eps).
    softplus is smooth (no relu kink), so there is no sharp activation point;
    the threshold bias is retained but applied smoothly.
    """
    mag = z.abs()
    scale = torch.nn.functional.softplus(mag + bias) / (mag + 1e-8)
    return z * scale


def cr_prune(z: torch.Tensor, rate: float) -> torch.Tensor:
    """CR pruning: the structured, phase-preserving replacement for dropout.

    Dropout zeroes arbitrary scalar entries, which on a complex field distorts
    the CR structure (a zeroed component has no well-defined phase).  CR
    pruning instead zeroes a random fraction ``rate`` of the *channels* (whole
    CR sub-fields on H_p) and rescales the survivors by 1/(1-rate): zeroing a
    whole sub-field is phase-consistent (0 = 0·e^{iθ}), so the CR surface is
    pruned, not twisted.  Applied in training only, on the field (B, N, d).
    """
    if rate <= 0 or not torch.is_grad_enabled():
        return z
    B, N, d = z.shape
    keep = (torch.rand(B, 1, d, device=z.device) >= rate).to(z.dtype)
    return z * keep / (1.0 - rate)


def _apply_nl(z: torch.Tensor, bias: torch.Tensor, nl: str) -> torch.Tensor:
    if nl == "modrelu":
        return complex_modrelu(z, bias)
    if nl == "softmodrelu":
        return complex_softmodrelu(z, bias)
    if nl == "radial":
        return complex_radial(z)
    if nl == "gelu":
        return _split_gelu(z)
    if nl == "none":
        return z
    raise ValueError(f"nl={nl!r} "
                     "(expected 'modrelu'|'softmodrelu'|'radial'|'gelu'|'none')")


class PiecewiseCRAttention(nn.Module):
    """Activation-segmented (piecewise-manifold) spectral-flow attention.

    Expressiveness of the per-stage diagonal filter is controlled by
    ``spectrum`` exactly as in FluidCRAttention ("full" | "mlp" |
    "diffusion"); the nonlinear segmentation is controlled by ``nl``
    ("modrelu" | "gelu" | "none") and the number of stages ``n_flow``.
    """

    def __init__(self, d_model: int, p: int | None = None, n_cr: int = 1,
                 eta: float = 1e-6, gate: bool = True, mix: bool = True,
                 n_flow: int = 3, t_init: float = 0.1,
                 spectrum: str = "full", spec_scale: float = 0.05,
                 nl: str = "gelu", spectral_mix: bool = False,
                 twist: bool = True, prune_rate: float = 0.0,
                 checkpoint: bool = False, half: bool = False,
                 szego: bool = False):
        super().__init__()
        self.prune_rate = prune_rate
        self.checkpoint = checkpoint
        self.half = half
        self.d_model = d_model
        self.p = p
        self.n_cr = n_cr
        self.eta = eta
        self.gate = gate
        self.mix = mix
        self.n_flow = int(n_flow)
        self.spectrum = spectrum
        self.spec_scale = spec_scale
        self.nl = nl
        self.spectral_mix = spectral_mix
        self.twist = twist
        self.szego = bool(szego)
        self._szego_cache: dict = {}
        if self.n_flow < 1:
            raise ValueError("n_flow must be >= 1")
        if spectrum not in ("full", "mlp", "diffusion"):
            raise ValueError(f"spectrum={spectrum!r} "
                             "(expected 'full'|'mlp'|'diffusion')")
        if spectrum == "full" and p is None:
            raise ValueError("spectrum='full' requires p at construction")
        if nl not in ("modrelu", "softmodrelu", "radial", "gelu", "none"):
            raise ValueError(f"nl={nl!r} "
                             "(expected 'modrelu'|'softmodrelu'|'radial'|'gelu'|'none')")

        # global flow time (heat-flow base on stage 0, see _flow_weights)
        self.t = nn.Parameter(torch.tensor(t_init, dtype=torch.float32))

        if spectrum == "full":
            # per-stage per-mode complex residual (n_flow, p, p, p)
            self.Wr = nn.Parameter(torch.zeros(self.n_flow, p, p, p))
            self.Wi = nn.Parameter(torch.zeros(self.n_flow, p, p, p))
        elif spectrum == "mlp":
            self.spec_mlp = nn.Sequential(
                nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, 2))
            nn.init.zeros_(self.spec_mlp[-1].weight)
            nn.init.zeros_(self.spec_mlp[-1].bias)

        # per-stage radial breakpoints (one scalar threshold per stage per
        # domain; init 0 => identity, so the untrained stack is a pure
        # diagonal spectral flow and training learns the segmentation).
        self.spec_bias = nn.Parameter(torch.zeros(self.n_flow))
        self.spat_bias = nn.Parameter(torch.zeros(self.n_flow))

        # learnable symplectic twist (chirp) per stage: a pointwise spatial
        # phase exp(2pi i a_k * a*b / p) that carries the H_p non-commutative
        # coupling (the a*b shear of the group law).  This is the chirp-z /
        # fractional-Fourier factorisation of the twist: chirp -> FFT -> chirp,
        # all pointwise + FFT, no p x p matrices.  a_k init 0 => identity.
        if twist:
            self.chirp_pre = nn.Parameter(torch.zeros(self.n_flow))
            self.chirp_post = nn.Parameter(torch.zeros(self.n_flow))

        if mix:
            self.channel_mix = ComplexLinear(d_model, d_model)
        if spectral_mix:
            self.spec_mix = ComplexLinear(d_model, d_model)
        self.gain_r = nn.Parameter(torch.ones(d_model))
        self.gain_i = nn.Parameter(torch.zeros(d_model))

    def _chirp_grid(self, p: int, device, dtype):
        """(p, p, 1) unit-free grid of a*b/p for the twist phase."""
        a = torch.arange(p, device=device, dtype=torch.float32).reshape(p, 1, 1)
        b = torch.arange(p, device=device, dtype=torch.float32).reshape(1, p, 1)
        return (a * b / p).to(dtype)

    def _szego_spectrum(self, p: int, device, dtype) -> torch.Tensor:
        """Scalar-FFT3 of the flat Szegő kernel = the diagonal spectral weight.

        The true Szegő projection on H_p annihilates the λ=0 centre modes
        (∂̄_b-exact fields) and projects λ≠0 onto the CR-holomorphic vacuum.  Its
        scalar (abelian) spectrum FFT3(S) is the diagonal approximation used by
        the matrix-free spectral flow, and it *automatically* carries the
        λ=0 → 0 suppression that the diffusion base gets wrong.  Normalised to
        unit max modulus.
        """
        key = (p, self.n_cr, self.eta)
        if key not in self._szego_cache:
            # signed coords in FFT order (identity at index 0), matching
            # _get_szego_kernel in cr_attention.py: w = (x²+y²) - i·t
            c = torch.fft.fftfreq(p, d=1.0).to(device) * p
            xx = c.reshape(p, 1, 1)
            yy = c.reshape(1, p, 1)
            tt = c.reshape(1, 1, p)
            w = (xx ** 2 + yy ** 2) - 1j * tt
            n1 = self.n_cr + 1
            S = torch.conj(w) ** n1 / ((w * torch.conj(w)).real + self.eta) ** n1
            # scalar 3-D FFT in the same dim order as _stage (x,y,t)
            S_hat = torch.fft.fft(S, dim=-1)
            S_hat = torch.fft.fft2(S_hat, dim=(-3, -2))
            m = S_hat.abs().max()
            S_hat = S_hat / (m + 1e-12)
            # Explicitly annihilate the centre modes: the true (non-abelian)
            # Szegő projection has S_0 = 0, but the scalar (abelian) FFT3 cannot
            # express that null space; zeroing the lambda=0 slice is the
            # scalar-basis approximation of the centre-mode annihilation.
            S_hat[..., 0] = 0
            self._szego_cache[key] = S_hat.detach()
        return self._szego_cache[key].to(device=device, dtype=dtype)

    def _flow_weights(self, k: int, p: int, device, dtype) -> torch.Tensor:
        lam = torch.fft.fftfreq(p, d=1.0).to(device)      # (p,)
        xi = torch.fft.fftfreq(p, d=1.0).to(device)       # (p,)
        lam = lam.reshape(1, 1, p)
        xix = xi.reshape(p, 1, 1)
        xiy = xi.reshape(1, p, 1)
        diff = torch.exp(-self.t * (lam.abs() + xix ** 2 + xiy ** 2))
        # stage 0 carries the base spectral operator; later stages start at
        # identity (residual W=0) so the untrained stack is well-posed.
        if self.szego and k == 0:
            base = self._szego_spectrum(p, device, dtype)
        else:
            base = diff if k == 0 else torch.ones_like(diff)
        if self.spectrum == "diffusion":
            res = torch.ones_like(diff)
        elif self.spectrum == "full":
            W = torch.complex(self.Wr[k], self.Wi[k])     # (p, p, p)
            res = 1.0 + self.spec_scale * W
        else:  # mlp
            coords = torch.stack(
                [xix.expand(p, p, p), xiy.expand(p, p, p),
                 lam.expand(p, p, p)], dim=-1) / p
            out = self.spec_mlp(coords)                   # (p, p, p, 2)
            res = 1.0 + self.spec_scale * torch.complex(out[..., 0],
                                                        out[..., 1])
        return (base * res).to(dtype)

    def _stage(self, x, k, ab, B, d, p):
        """One spectral-flow stage (FFT -> weight -> act -> IFFT -> act).

        With ``half=True`` the *storage* dtype is complex32 (fp16) but the FFT
        is computed in complex64 (fp32) — mixed precision that halves field
        memory while keeping full precision in the singular Szegő kernel, and
        works at prime p (fp32 cuFFT has no power-of-2 restriction).
        """
        if self.half:
            x = x.to(torch.complex64)                     # cast up for the FFT
        if self.twist:
            x = x * torch.exp(2j * torch.pi * self.chirp_pre[k] * ab)
        fh = torch.fft.fft(x, dim=-1)
        fh = torch.fft.fft2(fh, dim=(-3, -2))
        fh = fh * self._flow_weights(k, p, x.device, x.dtype)
        if self.spectral_mix and k == self.n_flow - 1:
            fh = fh.reshape(B, d, p, p, p).permute(0, 2, 3, 4, 1)
            fh = self.spec_mix(fh)
            fh = fh.permute(0, 4, 1, 2, 3).reshape(B * d, p, p, p)
        fh = _apply_nl(fh, self.spec_bias[k], self.nl)
        x = torch.fft.ifft2(fh, dim=(-3, -2))
        x = torch.fft.ifft(x, dim=-1)
        if self.twist:
            x = x * torch.exp(2j * torch.pi * self.chirp_post[k] * ab)
        x = _apply_nl(x, self.spat_bias[k], self.nl)
        if self.half:
            x = x.to(torch.complex32)                     # cast down for storage
        return x

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, N, d = z.shape
        p = self.p or round(N ** (1 / 3))
        x = z.permute(0, 2, 1).reshape(B * d, p, p, p).contiguous()
        if self.half:
            x = x.to(torch.complex32)
        ab = self._chirp_grid(p, x.device, x.dtype) if self.twist else None

        for k in range(self.n_flow):
            if self.checkpoint:
                x = torch.utils.checkpoint.checkpoint(
                    self._stage, x, k, ab, B, d, p, use_reentrant=False)
            else:
                x = self._stage(x, k, ab, B, d, p)

        if self.mix:
            x = x.reshape(B, p, p, p, d)
            x = x + self.channel_mix(x)                   # residual mixer
            x = x.reshape(B * d, p, p, p)

        gain = torch.complex(self.gain_r, self.gain_i)
        x = x * gain.repeat(B).reshape(-1, 1, 1, 1)

        if self.gate:
            e = _horizontal_energy(x, p)                  # (B*d, p, p, p)
            x = x * torch.sigmoid(-e).to(x.dtype)

        x = x.reshape(B, N, d)
        if self.prune_rate > 0.0:
            x = cr_prune(x, self.prune_rate)              # CR pruning (dropout)
        return x


def _horizontal_energy(f: torch.Tensor, p: int) -> torch.Tensor:
    """Per-point |grad_H f|^2 via spectral derivatives (dbar_b proxy)."""
    fx = _spectral_deriv(f, axis=-3, p=p)
    fy = _spectral_deriv(f, axis=-2, p=p)
    return (fx * fx.conj() + fy * fy.conj()).real


def _spectral_deriv(f: torch.Tensor, axis: int, p: int) -> torch.Tensor:
    k = torch.fft.fftfreq(p, d=1.0) * p
    shape = [1] * f.ndim
    shape[axis] = p
    k = k.view(shape).to(f.dtype).to(f.device)
    f_hat = torch.fft.fft(f, dim=axis)
    return torch.fft.ifft(1j * k * f_hat, dim=axis)


class PiecewiseCRBlock(nn.Module):
    """CR-Block with a piecewise-manifold attention, fully in complex space.

    norm -> PiecewiseCRAttention -> residual -> norm -> ComplexFFN -> residual.
    Input/output: (B, N, d) complex.
    """

    def __init__(self, d_model: int, p: int | None = None, n_cr: int = 1,
                 eta: float = 1e-6, gate: bool = True, mix: bool = True,
                 ff_expansion: int = 4, n_flow: int = 3,
                 spectrum: str = "full", spec_scale: float = 0.05,
                 nl: str = "gelu", spectral_mix: bool = False,
                 twist: bool = True):
        super().__init__()
        from .complex_nn import ComplexRMSNorm, ComplexFFN
        self.norm1 = ComplexRMSNorm(d_model)
        self.attn = PiecewiseCRAttention(
            d_model, p=p, n_cr=n_cr, eta=eta, gate=gate, mix=mix,
            n_flow=n_flow, spectrum=spectrum, spec_scale=spec_scale,
            nl=nl, spectral_mix=spectral_mix, twist=twist)
        self.norm2 = ComplexRMSNorm(d_model)
        self.ffn = ComplexFFN(d_model, ff_expansion)

    def forward(self, z):
        z = z + self.attn(self.norm1(z))
        z = z + self.ffn(self.norm2(z))
        return z
