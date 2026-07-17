"""Tests for sld.objectives.jepa: EMATarget and JEPALoss.

Seeds are explicit local generators (torch.Generator) throughout; no global
RNG state is mutated.
"""

from __future__ import annotations

import torch
from torch import nn

from sld.objectives.jepa import EMATarget, JEPALoss


def test_ema_gold_gradient_never_leaks_into_target() -> None:
    """The silent-bug lock: grad must flow through the prediction path only."""
    gen = torch.Generator().manual_seed(0)
    online = nn.Linear(3, 4)
    target = EMATarget(online, tau=0.9)
    pred_net = nn.Linear(4, 4)

    x = torch.randn(5, 3, generator=gen)
    z_online = online(x)
    h_hat = pred_net(z_online)
    z_target = target(x)

    loss = JEPALoss()(h_hat, z_target)["total"]
    loss.backward()

    for p in target.target.parameters():
        assert p.grad is None

    for module in (online, pred_net):
        for p in module.parameters():
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()
    grads = [p.grad for module in (online, pred_net) for p in module.parameters()]
    assert any((g != 0).any() for g in grads)


def test_ema_update_math_and_buffer_verbatim_copy() -> None:
    class ToyWithBuffer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(2, 2)
            self.register_buffer("running_stat", torch.zeros(3))

    online = ToyWithBuffer().double()
    with torch.no_grad():
        online.lin.weight.fill_(1.0)
        online.lin.bias.fill_(2.0)
        online.running_stat.fill_(0.0)

    tau = 0.9
    target = EMATarget(online, tau=tau)

    # move the online params/buffer to new values *after* the EMA snapshot,
    # so target holds the old value and online holds the new one.
    with torch.no_grad():
        online.lin.weight.fill_(3.0)
        online.lin.bias.fill_(5.0)
        online.running_stat.copy_(torch.tensor([7.0, 8.0, 9.0], dtype=torch.float64))

    target.update(online)

    expected_weight = torch.full((2, 2), tau * 1.0 + (1 - tau) * 3.0, dtype=torch.float64)
    expected_bias = torch.full((2,), tau * 2.0 + (1 - tau) * 5.0, dtype=torch.float64)
    torch.testing.assert_close(target.target.lin.weight, expected_weight, atol=1e-12, rtol=0)
    torch.testing.assert_close(target.target.lin.bias, expected_bias, atol=1e-12, rtol=0)

    # buffers are copied verbatim, not EMA'd
    torch.testing.assert_close(
        target.target.running_stat,
        torch.tensor([7.0, 8.0, 9.0], dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )


def test_consistency_stops_grad_at_target_but_not_lift() -> None:
    # Wave 3: lift (psi) trains THROUGH consistency -- only the target
    # embedding itself must receive no gradient (the classic JEPA leak).
    gen = torch.Generator().manual_seed(1)
    pred_z = torch.randn(5, 4, generator=gen, requires_grad=True)
    target_z = torch.randn(5, 4, generator=gen).requires_grad_(True)
    h_hat = torch.randn(5, 6, generator=gen, requires_grad=True)
    lift = nn.Linear(4, 6)

    loss = JEPALoss(lambda_c=1.0)(pred_z, target_z, h_hat=h_hat, lift=lift)["total"]
    loss.backward()

    assert target_z.grad is None or torch.all(target_z.grad == 0)

    for p in lift.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
        assert (p.grad != 0).any()

    for leaf in (pred_z, h_hat):
        assert leaf.grad is not None
        assert torch.isfinite(leaf.grad).all()


def test_loss_is_zero_when_prediction_matches_target_exactly() -> None:
    z = torch.randn(4, 4, generator=torch.Generator().manual_seed(2))
    out = JEPALoss()(z, z)
    assert out["total"].item() == 0.0


def test_two_arg_call_consistency_zero_total_equals_pred() -> None:
    pred_z = torch.randn(4, 4, generator=torch.Generator().manual_seed(4))
    target_z = torch.randn(4, 4, generator=torch.Generator().manual_seed(5))
    out = JEPALoss()(pred_z, target_z)
    assert out["consistency"].item() == 0.0
    assert out["total"].item() == out["pred"].item()


def test_consistency_term_active_with_lambda_c() -> None:
    gen = torch.Generator().manual_seed(3)
    pred_z = torch.randn(6, 4, generator=gen)
    target_z = torch.randn(6, 4, generator=gen)
    h_hat = torch.randn(6, 5, generator=gen)
    lift = nn.Linear(4, 5)

    lambda_c = 0.5
    out = JEPALoss(lambda_c=lambda_c)(pred_z, target_z, h_hat=h_hat, lift=lift)
    assert out["consistency"].item() > 0.0
    expected_total = out["pred"] + lambda_c * out["consistency"]
    torch.testing.assert_close(out["total"], expected_total, atol=0.0, rtol=0.0)


def test_exactly_one_of_h_hat_lift_raises() -> None:
    # silent half-wiring is the classic JEPA bug: refuse partial consistency wiring
    import pytest

    jepa = JEPALoss()
    lift = nn.Linear(4, 6)
    with pytest.raises(ValueError):
        jepa(torch.randn(10, 4), torch.randn(10, 4), lift=lift)
    with pytest.raises(ValueError):
        jepa(torch.randn(10, 4), torch.randn(10, 4), h_hat=torch.randn(10, 6))


def test_tau_domain_validated():
    # adversarial-review lock: the EMA update is a convex combination only
    # for tau in [0, 1]; outside it silently extrapolates
    import pytest
    from torch import nn

    from sld.objectives.jepa import EMATarget

    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            EMATarget(nn.Linear(2, 2), tau=bad)
    EMATarget(nn.Linear(2, 2), tau=0.0)  # boundaries are legal
    EMATarget(nn.Linear(2, 2), tau=1.0)


def test_shape_mismatch_rejected_not_broadcast():
    # adversarial-review lock: F.mse_loss silently broadcasts mismatched
    # time/batch dims into a wrong comparison — the loss must refuse
    import pytest
    import torch

    from sld.objectives.jepa import JEPALoss

    jepa = JEPALoss()
    with pytest.raises(ValueError):
        jepa(torch.randn(10, 4), torch.randn(10, 1))
    with pytest.raises(ValueError):
        jepa(torch.randn(9, 4), torch.randn(10, 4))
    # lifted path: h_hat must match lift(target_z), not target_z's own shape
    lift = torch.nn.Linear(4, 6)
    with pytest.raises(ValueError):
        jepa(torch.randn(10, 4), torch.randn(10, 4), h_hat=torch.randn(10, 4), lift=lift)
    out = jepa(torch.randn(10, 4), torch.randn(10, 4), h_hat=torch.randn(10, 6), lift=lift)
    assert out["pred"].shape == ()
    assert out["consistency"].shape == ()


def test_target_eval_mode_is_persistent():
    # nn.Module.train() recurses into children: without the override an
    # enclosing model's .train() would silently re-arm the target's train
    # mode (dropout/BN batch-stats) — eval must survive train() calls
    from torch import nn

    from sld.objectives.jepa import EMATarget

    ema = EMATarget(nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.5)))
    assert not ema.target.training
    ema.train()
    assert ema.training
    assert not ema.target.training
    ema.train(False)
    ema.train(True)
    assert not ema.target.training
