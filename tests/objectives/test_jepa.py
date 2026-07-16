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


def test_lifted_path_stops_grad_at_lift() -> None:
    gen = torch.Generator().manual_seed(1)
    online = nn.Linear(3, 4)
    target = EMATarget(online, tau=0.9)
    pred_net6 = nn.Linear(4, 6)
    lift = nn.Linear(4, 6)

    x = torch.randn(5, 3, generator=gen)
    z_online = online(x)
    h_hat6 = pred_net6(z_online)
    z_target = target(x)

    loss = JEPALoss()(h_hat6, z_target, lift=lift)["total"]
    loss.backward()

    for p in lift.parameters():
        assert p.grad is None

    for module in (online, pred_net6):
        for p in module.parameters():
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()


def test_loss_is_zero_when_prediction_matches_target_exactly() -> None:
    z = torch.randn(4, 4, generator=torch.Generator().manual_seed(2))
    out = JEPALoss()(z, z)
    assert out["total"].item() == 0.0
