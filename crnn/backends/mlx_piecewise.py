"""MLX (Apple Silicon) port of the matrix-free piecewise CR attention.

This is the M6 cross-platform deliverable: the piecewise attention is pure
FFT + pointwise complex arithmetic, so it ports directly to MLX's fft and
pointwise primitives with **no einsum, no batched matmul, no autograd trickery**.
Running on Apple Silicon (Metal) this gives the same O(K p^3 log p) complexity
and O(p^3) memory as the torch path.

The module mirrors ``crnn.layers.piecewise_cr_attention.PiecewiseCRAttention``
so the torch model and the MLX model are numerically interchangeable (see
``parity_test``).  MLX exposes the same functional API (``mlx.core.fft``,
``mlx.core.complex``, pointwise ops), so the port is a 1:1 transliteration.

Notes / caveats
---------------
* MLX has only ``complex64`` (no complex128) -- fine, the torch path is also
  complex64.
* ``mlx.core.fft.fft``/``ifft`` take ``axis``; 3D FFT is three 1D FFTs.
* Parameters are stored as real arrays (real/imag pairs), exactly as in the
  torch module, so a checkpoint transfers directly.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


def complex_modrelu(z, bias):
    mag = mx.abs(z)
    scale = mx.maximum(mag + bias, 0.0) / (mag + 1e-8)
    return z * scale


def complex_radial(z):
    mag = mx.abs(z)
    return z * mx.tanh(mag) / (mag + 1e-8)


def _apply_nl(z, bias, nl):
    if nl == "modrelu":
        return complex_modrelu(z, bias)
    if nl == "radial":
        return complex_radial(z)
    if nl == "none":
        return z
    # split-GELU (non-holomorphic) in MLX: apply gelu to real/imag separately
    if nl == "gelu":
        return mx.complex(nn.gelu(z.real), nn.gelu(z.imag))
    raise ValueError(f"nl={nl!r}")


class MLXPiecewiseCRAttention(nn.Module):
    """MLX equivalent of PiecewiseCRAttention (matrix-free spectral flow)."""

    def __init__(self, d_model, p, n_flow=3, spectrum="full", spec_scale=0.05,
                 nl="gelu", twist=True, t_init=0.1):
        super().__init__()
        self.d_model = d_model
        self.p = p
        self.n_flow = n_flow
        self.spectrum = spectrum
        self.spec_scale = spec_scale
        self.nl = nl
        self.twist = twist
        self.t = mx.array(t_init)
        if spectrum == "full":
            self.Wr = mx.zeros((n_flow, p, p, p))
            self.Wi = mx.zeros((n_flow, p, p, p))
        self.spec_bias = mx.zeros((n_flow,))
        self.spat_bias = mx.zeros((n_flow,))
        if twist:
            self.chirp_pre = mx.zeros((n_flow,))
            self.chirp_post = mx.zeros((n_flow,))
        self.gain_r = mx.ones((d_model,))
        self.gain_i = mx.zeros((d_model,))

    def _flow_weights(self, k, p, dtype):
        lam = mx.fft.fftfreq(p)                    # (p,)
        xi = mx.fft.fftfreq(p)
        lam = mx.reshape(lam, (1, 1, p))
        xix = mx.reshape(xi, (p, 1, 1))
        xiy = mx.reshape(xi, (1, p, 1))
        diff = mx.exp(-self.t * (mx.abs(lam) + xix ** 2 + xiy ** 2))
        base = diff if k == 0 else mx.ones_like(diff)
        if self.spectrum == "diffusion":
            res = mx.ones_like(diff)
        elif self.spectrum == "full":
            res = 1.0 + self.spec_scale * mx.complex(self.Wr[k], self.Wi[k])
        else:
            raise ValueError(self.spectrum)
        return (base * res).astype(dtype)

    def _chirp_grid(self, p, dtype):
        a = mx.reshape(mx.arange(p).astype(mx.float32), (p, 1, 1))
        b = mx.reshape(mx.arange(p).astype(mx.float32), (1, p, 1))
        return (a * b / p).astype(dtype)

    def __call__(self, z):
        # z: (B, N, d) complex64.  Pack to (B*d, p, p, p).
        B, N, d = z.shape
        p = self.p
        z = mx.transpose(z, (0, 2, 1))             # (B, d, N)
        x = mx.reshape(z, (B * d, p, p, p))
        ab = self._chirp_grid(p, x.dtype) if self.twist else None

        for k in range(self.n_flow):
            if self.twist:
                ph = 2.0 * math.pi * self.chirp_pre[k] * ab
                x = x * mx.exp(mx.complex(mx.zeros_like(ph), ph))
            fh = mx.fft.fft(x, axis=-1)
            fh = mx.fft.fft(fh, axis=-3)
            fh = mx.fft.fft(fh, axis=-2)
            fh = fh * self._flow_weights(k, p, fh.dtype)
            fh = _apply_nl(fh, self.spec_bias[k], self.nl)
            x = mx.fft.ifft(fh, axis=-2)
            x = mx.fft.ifft(x, axis=-3)
            x = mx.fft.ifft(x, axis=-1)
            if self.twist:
                ph = 2.0 * math.pi * self.chirp_post[k] * ab
                x = x * mx.exp(mx.complex(mx.zeros_like(ph), ph))
            x = _apply_nl(x, self.spat_bias[k], self.nl)

        gain = mx.complex(self.gain_r, self.gain_i)
        x = x * mx.reshape(mx.tile(gain, (B,)), (B * d, 1, 1, 1))
        out = mx.reshape(mx.transpose(mx.reshape(x, (B, d, N)), (0, 2, 1)),
                         (B, N, d))
        return out


def parity_test(p=5, d=8, nl="radial", tol=1e-4):
    """torch vs MLX parity for the piecewise forward (run where mlx imports)."""
    import numpy as np
    import torch

    from crnn.layers.piecewise_cr_attention import PiecewiseCRAttention

    rng = np.random.default_rng(0)
    zr = rng.standard_normal((2, p ** 3, d)).astype(np.float32)
    zi = rng.standard_normal((2, p ** 3, d)).astype(np.float32)

    torch.manual_seed(0)
    ta = PiecewiseCRAttention(d_model=d, p=p, n_flow=2, nl=nl)
    zt = torch.complex(torch.from_numpy(zr), torch.from_numpy(zi))
    out_t = ta(zt).detach().numpy()

    ma = MLXPiecewiseCRAttention(d_model=d, p=p, n_flow=2, nl=nl)
    zm = mx.complex(mx.array(zr), mx.array(zi))
    out_m = np.array(ma(zm))

    err = np.abs(out_t - out_m).max()
    print(f"parity (p={p}, d={d}, nl={nl}): max abs diff = {err:.3e} "
          f"{'OK' if err < tol else 'MISMATCH'}")
    return err
