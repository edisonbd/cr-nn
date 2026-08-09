"""Smoke test: CR-Attention and CR-Block forward/backward.

Verifies the layer runs end-to-end (no shape errors, autograd flows) at a
small prime p. Numerical correctness of the attention math itself is covered
by the FFT/convolution tests in test_heisenberg.py; this is purely "does it
run as a torch module."
"""
import torch
from crnn.layers import CRAttention, CRBlock


def test_cr_attention_forward_backward():
    p = 5
    N = p ** 3
    d = 8
    B = 2
    layer = CRAttention(d_model=d, p=p, M=0, gate=True)
    x = torch.randn(B, N, d, requires_grad=True)
    out = layer(x)
    assert out.shape == (B, N, d), f"shape {out.shape}"
    loss = out.sum()
    loss.backward()
    assert x.grad is not None and x.grad.shape == x.shape
    # learnable params got gradients
    assert layer.out_proj.weight.grad is not None


def test_cr_block_forward_backward():
    p = 5
    N = p ** 3
    d = 8
    B = 2
    block = CRBlock(d_model=d, p=p, M=0, gate=True)
    x = torch.randn(B, N, d, requires_grad=True)
    out = block(x)
    assert out.shape == (B, N, d), f"shape {out.shape}"
    out.sum().backward()
    assert x.grad is not None
    # check a few params have grads
    assert block.attn.out_proj.weight.grad is not None
    assert block.ffn.fc1.weight.grad is not None


def test_cr_attention_padding():
    # sequence length not exactly p^3 -> layer pads
    p = 5
    d = 4
    layer = CRAttention(d_model=d, p=p, M=0, gate=False)
    x = torch.randn(1, 100, d)   # 100 < 125 = p^3
    out = layer(x)
    assert out.shape == (1, 125, d), f"padded shape {out.shape}"
