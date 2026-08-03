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


def test_covariance_loss_matches_hand_derived_value() -> None:
    """Fixed N=2, d=2 input. The convention ACTUALLY implemented
    (src/sld/objectives/vicreg.py:35-48) is::

        C = z_centered.T @ z_centered / (N - 1)
        result = (sum(C**2) - sum(diag(C)**2)) / d

    Derivation, done here independently of the source before comparing
    (REGLES 7.2)::

        z = [[1, 2], [3, 0]]            -> mean = [2, 1]
        z_centered = [[-1, 1], [1, -1]]
        C = z_centered.T @ z_centered / (2 - 1) = [[2, -2], [-2, 2]]
        off-diagonal sum of squares = (-2)^2 + (-2)^2 = 8   (diag 2^2+2^2=8 excluded)
        result = 8 / d = 8 / 2 = 4.0

    Every intermediate value is a small integer or an exact power-of-two
    division, so float32 introduces no rounding at all here (measured error
    running this test is exactly 0.0); atol=1e-6 is representational
    headroom (float32 tier), not real slack.
    """
    z = torch.tensor([[1.0, 2.0], [3.0, 0.0]])
    expected = 4.0
    actual = covariance_loss(z)
    print(f"[covariance_loss exact] actual={actual.item()} expected={expected}")
    torch.testing.assert_close(actual, torch.tensor(expected), atol=1e-6, rtol=0)


def test_covariance_loss_rejects_wrong_normalization_conventions() -> None:
    """Same z as above. Three plausible-but-wrong conventions -- each
    computed independently here, never by calling covariance_loss -- must
    NOT reproduce its actual output; otherwise the positive test above would
    pass for a broken implementation too (REGLES 7.1's "would a wrong
    implementation fail this test?").
    """
    z = torch.tensor([[1.0, 2.0], [3.0, 0.0]])
    correct = covariance_loss(z).item()
    n, d = z.shape
    z_c = z - z.mean(dim=0, keepdim=True)
    cov_correct = (z_c.T @ z_c) / (n - 1)

    # (a) /N instead of /(N - 1)
    cov_a = (z_c.T @ z_c) / n
    off_a = cov_a.pow(2).sum() - cov_a.diagonal().pow(2).sum()
    wrong_a = (off_a / d).item()

    # (b) diagonal included in the penalized sum
    wrong_b = (cov_correct.pow(2).sum() / d).item()

    # (c) missing the /d normalization
    off_correct = cov_correct.pow(2).sum() - cov_correct.diagonal().pow(2).sum()
    wrong_c = off_correct.item()

    for label, wrong in (
        ("/N instead of /(N-1)", wrong_a),
        ("diagonal included", wrong_b),
        ("missing /d", wrong_c),
    ):
        gap = abs(wrong - correct)
        print(f"[covariance_loss negative: {label}] wrong={wrong} correct={correct} gap={gap:.6f}")
        assert gap > 1e-3, f"{label} ({wrong}) must NOT match the correct value ({correct})"


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
