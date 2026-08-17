"""Tests for the CR-completion components (pre-M5):
complex building blocks, vector-valued CR-Attention/CR-Block, the cr-vec
ToyLM and the Sobolev main loss.
"""

import torch

from crnn.layers import (ComplexFFN, ComplexLayerNorm, ComplexRMSNorm, ComplexLinear,
                         FluidCRAttention, VecCRAttention, VecCRBlock)
from crnn.losses import CollapseLoss, SobolevEmbeddingLoss
from crnn.models import ToyLM


def test_complex_linear_forward_backward():
    lin = ComplexLinear(8, 8)
    z = torch.randn(2, 10, 8, dtype=torch.complex64, requires_grad=True)
    out = lin(z)
    assert out.shape == z.shape and torch.is_complex(out)
    out.sum().abs().backward()
    assert z.grad is not None


def test_complex_layernorm():
    ln = ComplexLayerNorm(8)
    z = torch.randn(2, 10, 8, dtype=torch.complex64)
    out = ln(z)
    assert out.shape == z.shape and torch.is_complex(out)
    # RMS modulus roughly unit per position after normalisation (scale=1 init)
    rms = out.abs().square().mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones(2, 10), atol=1e-2)


def test_complex_rmsnorm():
    rn = ComplexRMSNorm(8)
    z = torch.randn(2, 10, 8, dtype=torch.complex64)
    out = rn(z)
    assert out.shape == z.shape and torch.is_complex(out)
    rms = out.abs().square().mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones(2, 10), atol=1e-2)


def test_complex_ffn_forward_backward():
    ff = ComplexFFN(8, expansion=4)
    z = torch.randn(2, 10, 8, dtype=torch.complex64, requires_grad=True)
    out = ff(z)
    assert out.shape == z.shape and torch.is_complex(out)
    out.sum().abs().backward()
    assert z.grad is not None
    assert ff.fc1.Wr.grad is not None


def test_vec_cr_attention_forward_backward():
    p = 5
    N = p ** 3
    d = 8
    attn = VecCRAttention(d_model=d, p=p, gate=True, mix=True)
    z = torch.randn(2, N, d, dtype=torch.complex64, requires_grad=True)
    out = attn(z)
    assert out.shape == z.shape and torch.is_complex(out)
    out.sum().abs().backward()
    assert z.grad is not None
    assert attn.channel_mix.Wr.grad is not None


def test_vec_cr_block_forward_backward():
    p = 5
    N = p ** 3
    block = VecCRBlock(d_model=8, p=p, gate=True, mix=True)
    z = torch.randn(2, N, 8, dtype=torch.complex64, requires_grad=True)
    out = block(z)
    assert out.shape == z.shape and torch.is_complex(out)
    out.sum().abs().backward()
    assert z.grad is not None
    assert block.ffn.fc1.Wr.grad is not None


def test_toylm_cr_vec_forward_backward():
    p = 5
    N = p ** 3
    model = ToyLM(vocab=32, d_model=16, n_layers=2, p=p,
                  block_type="cr-vec")
    x = torch.randint(0, 32, (2, N))
    logits, hidden = model(x, return_hidden=True)
    assert logits.shape == (2, N, 32)
    assert torch.is_complex(hidden)
    grid = model.hidden_to_grid(hidden)
    assert grid.shape == (2 * 16, p, p, p)
    tgt = torch.randint(0, 32, (2, N))
    tgrid = model.embed_target_grid(tgt)
    assert tgrid.shape == grid.shape and torch.is_complex(tgrid)
    loss = logits.sum().abs() + (grid - tgrid).abs().sum()
    loss.backward()
    assert model.blocks[0].attn.channel_mix.Wr.grad is not None


def test_sobolev_embedding_loss():
    p = 5
    B = 8
    fn = SobolevEmbeddingLoss(ce_weight=0.0, so_weight=1.0, mu=0.0)
    logits = torch.randn(1, p ** 3, 32)
    tokens = torch.randint(0, 32, (1, p ** 3))
    f = torch.randn(B, p, p, p, dtype=torch.complex64, requires_grad=True)
    total, stats = fn(logits, tokens, f, f.detach(), p=p)
    assert total.ndim == 0
    # so term is zero when hidden == target and mu == 0
    assert total.item() < 1e-5
    total.backward()
    assert f.grad is not None


def test_fluid_attention_forward_backward():
    p = 5
    N = p ** 3
    attn = FluidCRAttention(d_model=8, p=p, gate=True, mix=True)
    z = torch.randn(2, N, 8, dtype=torch.complex64, requires_grad=True)
    out = attn(z)
    assert out.shape == z.shape and torch.is_complex(out)
    assert torch.isfinite(out).all()
    out.sum().abs().backward()
    assert z.grad is not None
    assert attn.t.grad is not None


def test_fluid_attention_spectrum_variants():
    for spec in ("full", "mlp", "diffusion"):
        p = 5
        N = p ** 3
        attn = FluidCRAttention(d_model=8, p=p, spectrum=spec)
        z = torch.randn(2, N, 8, dtype=torch.complex64, requires_grad=True)
        out = attn(z)
        assert out.shape == z.shape and torch.isfinite(out).all()
        out.sum().abs().backward()
        assert attn.t.grad is not None
        if spec == "full":
            assert attn.Wr.grad is not None
        elif spec == "mlp":
            assert attn.spec_mlp[-1].weight.grad is not None


def test_fluid_attention_spectral_mix():
    p = 5
    N = p ** 3
    attn = FluidCRAttention(d_model=8, p=p, spectrum="full",
                            spectral_mix=True)
    z = torch.randn(2, N, 8, dtype=torch.complex64, requires_grad=True)
    out = attn(z)
    assert out.shape == z.shape and torch.isfinite(out).all()
    out.sum().abs().backward()
    assert attn.spec_mix.Wr.grad is not None


def test_collapse_loss():
    p = 5
    B = 8
    fn = CollapseLoss(ce_weight=1.0, col_weight=1.0, mu=1e-3)
    logits = torch.randn(1, p ** 3, 32)
    tokens = torch.randint(0, 32, (1, p ** 3))
    f = torch.randn(B, p, p, p, dtype=torch.complex64, requires_grad=True)
    t = torch.randn(B, p, p, p, dtype=torch.complex64)
    total, stats = fn(logits, tokens, f, t, p=p)
    assert total.ndim == 0
    assert set(stats) >= {"ce", "sob", "im", "dbar"}
    total.backward()
    assert f.grad is not None
