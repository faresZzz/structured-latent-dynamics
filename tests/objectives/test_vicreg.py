"""Tests for sld.objectives.vicreg: variance_loss and covariance_loss.

Seeds are explicit local generators (torch.Generator) throughout; no global
RNG state is mutated.
"""

from __future__ import annotations

import torch

from sld.objectives.vicreg import covariance_loss, variance_loss


def test_variance_loss_constant_batch_hits_the_hinge_ceiling() -> None:
    z = torch.ones(10, 4)  # all rows identical -> std_j == 0 exactly
    loss = variance_loss(z, gamma=1.0, eps=1e-4)
    expected = 1.0 - (1e-4) ** 0.5  # 0.99
    torch.testing.assert_close(loss, torch.tensor(expected), atol=1e-6, rtol=0)


def test_isotropic_gaussian_has_low_variance_and_covariance_loss() -> None:
    gen = torch.Generator().manual_seed(0)
    z = torch.randn(20_000, 8, generator=gen)
    assert variance_loss(z).item() < 0.05
    assert covariance_loss(z).item() < 0.01


def test_perfectly_correlated_dims_spike_covariance_loss() -> None:
    gen = torch.Generator().manual_seed(0)
    z = torch.randn(20_000, 8, generator=gen)
    isotropic_cov = covariance_loss(z).item()

    z_corr = z.clone()
    z_corr[:, 1] = z_corr[:, 0]  # dim 1 perfectly correlated with dim 0
    correlated_cov = covariance_loss(z_corr).item()

    assert correlated_cov > 10 * isotropic_cov


def test_both_losses_are_differentiable() -> None:
    gen = torch.Generator().manual_seed(3)
    z = torch.randn(32, 6, generator=gen, requires_grad=True)

    (variance_loss(z) + covariance_loss(z)).backward()

    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_singleton_batch_raises():
    # adversarial-review lock: N=1 makes unbiased variance / covariance
    # undefined (silent NaN) — both losses must refuse it explicitly
    import pytest
    import torch

    from sld.objectives.vicreg import covariance_loss, variance_loss

    z = torch.randn(1, 4)
    with pytest.raises(ValueError):
        variance_loss(z)
    with pytest.raises(ValueError):
        covariance_loss(z)
