"""M5 curvature-perturbation tests (original cr + fully-complex cr-vec)."""

import torch

from crnn.curvature.perturbation import delta_b_powers
from crnn.layers import CRAttention, VecCRAttention


def _dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_delta_b_powers():
    p = 5
    f = torch.randn(2, p, p, p, dtype=torch.complex64, device=_dev())
    powers = delta_b_powers(f, M=3)
    assert len(powers) == 3
    for g in powers:
        assert g.shape == f.shape and torch.is_complex(g)


def test_cr_attention_perturbation_m0_equals_flat():
    # eps is init 0 -> soft-constrained eps_eff = 0 -> M>0 must equal M=0.
    p = 5
    N = p ** 3
    d = 8
    torch.manual_seed(0)
    flat = CRAttention(d_model=d, p=p, M=0, gate=False)
    curved = CRAttention(d_model=d, p=p, M=2, gate=False)
    flat = flat.to(_dev())
    curved = curved.to(_dev())
    # share every weight the two architectures have in common (eps stays 0)
    flat.load_state_dict(curved.state_dict(), strict=False)
    x = torch.randn(2, N, d, device=_dev())
    with torch.no_grad():
        y0 = flat(x)
        y1 = curved(x)
    assert torch.allclose(y0, y1, atol=1e-5)


def test_cr_attention_perturbation_backward():
    p = 5
    N = p ** 3
    attn = CRAttention(d_model=8, p=p, M=2, gate=False)
    attn = attn.to(_dev())
    x = torch.randn(2, N, 8, device=_dev(), requires_grad=True)
    out = attn(x)
    out.sum().backward()
    assert attn.eps.grad is not None
    assert attn.eps.grad.shape == (2,)
    assert x.grad is not None


def test_vec_cr_attention_perturbation_backward():
    p = 5
    N = p ** 3
    attn = VecCRAttention(d_model=8, p=p, M=2, gate=False, mix=True)
    attn = attn.to(_dev())
    z = torch.randn(2, N, 8, dtype=torch.complex64, device=_dev(),
                    requires_grad=True)
    out = attn(z)
    out.sum().abs().backward()
    assert attn.eps.grad is not None
    assert attn.eps.grad.shape == (2,)
    assert z.grad is not None


def test_log_correction_runs_and_backward():
    p = 5
    N = p ** 3
    attn = CRAttention(d_model=8, p=p, M=1, gate=False,
                       log_correction=True)
    attn = attn.to(_dev())
    x = torch.randn(2, N, 8, device=_dev(), requires_grad=True)
    out = attn(x)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert attn.eps_log.grad is not None


def test_vec_log_correction_runs():
    p = 5
    N = p ** 3
    attn = VecCRAttention(d_model=8, p=p, M=1, gate=False, mix=True,
                          log_correction=True)
    attn = attn.to(_dev())
    z = torch.randn(2, N, 8, dtype=torch.complex64, device=_dev())
    out = attn(z)
    assert torch.isfinite(out).all()


def test_perturbation_m3_finite_with_normalization():
    # RMS-normalised Delta_b powers must keep M=3 finite (raw M=3 explodes).
    p = 5
    N = p ** 3
    attn = CRAttention(d_model=8, p=p, M=3, gate=False).to(_dev())
    x = torch.randn(2, N, 8, device=_dev())
    out = attn(x)
    assert torch.isfinite(out).all()
    vec = VecCRAttention(d_model=8, p=p, M=3, gate=False, mix=True).to(_dev())
    z = torch.randn(2, N, 8, dtype=torch.complex64, device=_dev())
    assert torch.isfinite(vec(z)).all()
