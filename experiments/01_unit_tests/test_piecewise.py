"""Tests for the piecewise-manifold attention.

Covers the modReLU radial breakpoint, the PiecewiseCRAttention layer across
its spectrum/nl variants, the PiecewiseCRBlock, the ToyLM ``cr-vec`` +
``piecewise`` path, and the ``nl="none"`` identity sanity check (n_flow
stages of pure diagonal flow must reduce to a single diagonal flow in the
no-nonlinearity, single-stage, diffusion case).
"""

import torch

from crnn.layers import (FluidCRAttention, PiecewiseCRAttention,
                         PiecewiseCRBlock, VecCRBlock, complex_modrelu,
                         GeoCRBlock)
from crnn.layers.piecewise_cr_attention import (complex_radial,
                                                complex_softmodrelu)
from crnn.models import ToyLM


def test_modrelu_breakpoint():
    z = torch.tensor([0.5 + 0.0j, -2.0 + 0.0j, 2.0j], dtype=torch.complex64)
    b = torch.tensor(-1.0)
    out = complex_modrelu(z, b)
    # |0.5| + (-1) = -0.5 -> relu = 0 -> zeroed (inside the breakpoint disk)
    assert out[0] == 0.0
    # |-2| + (-1) = 1 -> relu = 1 -> phase preserved: out = 1 * z/|z| = -1
    assert abs(out[1] - (-1.0)) < 1e-6
    # |2j| + (-1) = 1 -> out = 1 * z/|z| = 1j (phase preserved)
    assert abs(out[2] - 1.0j) < 1e-6


def test_modrelu_identity_at_zero_bias():
    z = torch.randn(4, 5, 3, dtype=torch.complex64) * 2
    out = complex_modrelu(z, torch.tensor(0.0))
    # b=0 => relu(|z|) = |z| => out = z (exactly, up to fp)
    assert torch.allclose(out, z, atol=1e-5)


def test_piecewise_attention_forward_backward():
    p = 5
    N = p ** 3
    attn = PiecewiseCRAttention(d_model=8, p=p, gate=True, mix=True,
                                n_flow=3, nl="modrelu")
    z = torch.randn(2, N, 8, dtype=torch.complex64, requires_grad=True)
    out = attn(z)
    assert out.shape == z.shape and torch.is_complex(out)
    assert torch.isfinite(out).all()
    out.sum().abs().backward()
    assert z.grad is not None
    assert attn.Wr.grad is not None
    assert attn.spec_bias.grad is not None
    assert attn.spat_bias.grad is not None
    assert attn.chirp_pre.grad is not None
    assert attn.channel_mix.Wr.grad is not None


def test_piecewise_spectrum_variants():
    for spec in ("full", "mlp", "diffusion"):
        p = 5
        N = p ** 3
        attn = PiecewiseCRAttention(d_model=8, p=p, spectrum=spec, n_flow=2)
        z = torch.randn(2, N, 8, dtype=torch.complex64, requires_grad=True)
        out = attn(z)
        assert out.shape == z.shape and torch.isfinite(out).all()
        out.sum().abs().backward()
        assert attn.t.grad is not None
        if spec == "full":
            assert attn.Wr.grad is not None
        elif spec == "mlp":
            assert attn.spec_mlp[-1].weight.grad is not None


def test_piecewise_nl_variants():
    for nl in ("modrelu", "gelu", "none"):
        p = 5
        N = p ** 3
        attn = PiecewiseCRAttention(d_model=8, p=p, nl=nl, n_flow=2)
        z = torch.randn(2, N, 8, dtype=torch.complex64, requires_grad=True)
        out = attn(z)
        assert out.shape == z.shape and torch.isfinite(out).all()
        out.sum().abs().backward()
        assert z.grad is not None


def test_piecewise_identity_limit_matches_fluid():
    # nl="none", n_flow=1, diffusion spectrum => a single diagonal spectral
    # stage with the heat-flow base: numerically the same as FluidCRAttention
    # with spectrum="diffusion" (same t init, same gain/mixer init).
    p = 5
    N = p ** 3
    d = 8
    torch.manual_seed(0)
    pw = PiecewiseCRAttention(d_model=d, p=p, n_flow=1, spectrum="diffusion",
                              nl="none", gate=False, mix=False)
    torch.manual_seed(0)
    fl = FluidCRAttention(d_model=d, p=p, spectrum="diffusion",
                          gate=False, mix=False)
    # reset the shared t init to a fixed value for a deterministic comparison
    with torch.no_grad():
        pw.t.copy_(torch.tensor(0.1))
        fl.t.copy_(torch.tensor(0.1))
        pw.gain_r.copy_(torch.ones(d))
        fl.gain_r.copy_(torch.ones(d))
    z = torch.randn(2, N, d, dtype=torch.complex64)
    assert torch.allclose(pw(z), fl(z), atol=1e-4)


def test_piecewise_block_forward_backward():
    p = 5
    N = p ** 3
    block = PiecewiseCRBlock(d_model=8, p=p, gate=True, mix=True, n_flow=2)
    z = torch.randn(2, N, 8, dtype=torch.complex64, requires_grad=True)
    out = block(z)
    assert out.shape == z.shape and torch.is_complex(out)
    out.sum().abs().backward()
    assert z.grad is not None
    assert block.ffn.fc1.Wr.grad is not None


def test_toylm_piecewise_forward_backward():
    p = 5
    N = p ** 3
    model = ToyLM(vocab=32, d_model=16, n_layers=2, p=p,
                  block_type="cr-vec", attn_type="piecewise", n_flow=2)
    x = torch.randint(0, 32, (2, N))
    logits, hidden = model(x, return_hidden=True)
    assert logits.shape == (2, N, 32)
    assert torch.is_complex(hidden)
    loss = logits.sum().abs()
    loss.backward()
    assert model.blocks[0].attn.Wr.grad is not None


def test_vec_cr_block_accepts_piecewise():
    p = 5
    N = p ** 3
    block = VecCRBlock(d_model=8, p=p, gate=True, mix=True,
                       attn_type="piecewise", n_flow=2, nl="modrelu")
    z = torch.randn(2, N, 8, dtype=torch.complex64, requires_grad=True)
    out = block(z)
    assert out.shape == z.shape and torch.is_complex(out)
    out.sum().abs().backward()
    assert z.grad is not None


def test_geo_block_no_linear():
    p = 5
    N = p ** 3
    block = GeoCRBlock(d_model=8, p=p, n_flow=2)
    # assert no nn.Linear / matmul layers anywhere in the block
    def has_linear(m):
        if isinstance(m, torch.nn.Linear):
            return True
        return any(has_linear(c) for c in m.children())
    assert not has_linear(block)
    z = torch.randn(2, N, 8, dtype=torch.complex64, requires_grad=True)
    out = block(z)
    assert out.shape == z.shape and torch.is_complex(out)
    assert torch.isfinite(out).all()
    out.sum().abs().backward()
    assert z.grad is not None


def test_toylm_cr_geo_forward_backward():
    p = 5
    N = p ** 3
    model = ToyLM(vocab=32, d_model=16, n_layers=2, p=p, block_type="cr-geo")
    x = torch.randint(0, 32, (2, N))
    logits, hidden = model(x, return_hidden=True)
    assert logits.shape == (2, N, 32)
    assert torch.is_complex(hidden)
    loss = logits.sum().abs()
    loss.backward()
    assert model.blocks[0].attn.Wr.grad is not None


def test_radial_smooth_phase_preserving():
    z = torch.tensor([0.5 + 0.0j, -2.0 + 1.0j, 1.0j], dtype=torch.complex64)
    out = complex_radial(z)
    # phase preserved: out / |out| == z / |z| up to fp
    ang_in = z / (z.abs() + 1e-8)
    ang_out = out / (out.abs() + 1e-8)
    assert torch.allclose(ang_in, ang_out, atol=1e-4)
    # bounded and smooth: |out| = tanh(|z|) < 1, no zeroing of |z|=2
    assert (out.abs() < 1.0).all()
    assert out.abs()[1] > 0.9  # tanh(2) ~ 0.964, NOT zeroed (no breakpoint)


def test_softmodrelu_smooth():
    z = torch.randn(16, 8, 4, dtype=torch.complex64) * 2
    out = complex_softmodrelu(z, torch.tensor(0.0))
    assert out.shape == z.shape and torch.isfinite(out).all()
    # no hard zeroing: all magnitudes strictly positive for nonzero z
    assert (out.abs()[z.abs() > 1e-3] > 0).all()


def test_piecewise_radial_nl():
    p = 5
    N = p ** 3
    for nl in ("radial", "softmodrelu"):
        attn = PiecewiseCRAttention(d_model=8, p=p, nl=nl, n_flow=2)
        z = torch.randn(2, N, 8, dtype=torch.complex64, requires_grad=True)
        out = attn(z)
        assert out.shape == z.shape and torch.isfinite(out).all()
        out.sum().abs().backward()
        assert z.grad is not None


def test_piecewise_twist_chirp():
    p = 5
    N = p ** 3
    # twist on vs off with a non-zero chirp strength must differ
    a = PiecewiseCRAttention(d_model=8, p=p, n_flow=1, nl="none",
                             gate=False, mix=False, twist=True)
    with torch.no_grad():
        a.chirp_pre[0].copy_(torch.tensor(1.0))
        a.t.copy_(torch.tensor(0.0))
    b = PiecewiseCRAttention(d_model=8, p=p, n_flow=1, nl="none",
                             gate=False, mix=False, twist=False)
    with torch.no_grad():
        b.t.copy_(torch.tensor(0.0))
    z = torch.randn(2, N, 8, dtype=torch.complex64)
    assert not torch.allclose(a(z), b(z))
    # chirp strength 0 (identity) on the twisted layer == untwisted layer
    with torch.no_grad():
        a.chirp_pre[0].copy_(torch.tensor(0.0))
    assert torch.allclose(a(z), b(z), atol=1e-5)
