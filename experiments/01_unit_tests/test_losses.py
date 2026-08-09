"""Smoke test for CR-Sobolev loss: differentiable, sane values, vanishes on target."""
import torch
from crnn.losses import CRSobolevLoss, dbar_energy, cr_sobolev_norm


def test_loss_differentiable():
    p = 5
    B = 2
    out = torch.randn(B, p, p, p, dtype=torch.complex64, requires_grad=True)
    target = torch.randn(B, p, p, p, dtype=torch.complex64)
    loss_fn = CRSobolevLoss(mu=1e-3, s=1.0, n=1)
    loss = loss_fn(out, target, p=p)
    assert loss.requires_grad and loss.ndim == 0
    loss.backward()
    assert out.grad is not None and out.grad.shape == out.shape


def test_loss_zero_on_match():
    p = 5
    B = 1
    f = torch.randn(B, p, p, p, dtype=torch.complex64)
    loss_fn = CRSobolevLoss(mu=1e-3, s=1.0, n=1)
    loss = loss_fn(f, f, p=p)
    # when out==target, the Sobolev term is 0; only μ‖∂̄_b f‖² remains
    assert loss.item() >= 0
    # with mu=0, loss should be exactly 0
    loss_fn0 = CRSobolevLoss(mu=0.0, s=1.0, n=1)
    assert loss_fn0(f, f, p=p).item() < 1e-10


def test_dbar_energy_positive():
    p = 5
    f = torch.randn(2, p, p, p, dtype=torch.complex64)
    e = dbar_energy(f, p=p)
    assert e.shape == (2,)
    assert (e >= 0).all()


def test_dbar_energy_zero_on_constant():
    # CR functions on the torus are constants; ∂̄_b of a constant vanishes.
    p = 5
    f = torch.full((1, p, p, p), 3.0 + 0j, dtype=torch.complex64)
    e = dbar_energy(f, p=p)
    assert e.item() < 1e-8
