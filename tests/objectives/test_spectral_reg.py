"""Tests for sld.objectives.spectral_reg.SpectralSeparationReg.

Seeds are explicit local generators (torch.Generator) throughout; no global
RNG state is mutated.
"""

from __future__ import annotations

import math

import torch

from sld.objectives.spectral_reg import SpectralSeparationReg


def test_zero_loss_exactly_at_the_two_wells() -> None:
    rho_d = 0.9
    reg = SpectralSeparationReg(rho_d=rho_d, beta=1.0)
    moduli = torch.tensor([1.0, rho_d], dtype=torch.float64)
    loss = reg(moduli)
    torch.testing.assert_close(loss, torch.zeros((), dtype=torch.float64), atol=1e-12, rtol=0)


def test_hinge_and_double_well_at_modulus_above_one() -> None:
    rho_d = 0.9
    beta = 1.0
    reg = SpectralSeparationReg(rho_d=rho_d, beta=beta)
    modulus = 1.05
    moduli = torch.tensor([modulus], dtype=torch.float64)
    loss = reg(moduli)

    log_m = math.log(modulus)
    log_rho_d = math.log(rho_d)
    expected = (modulus - 1.0) ** 2 + beta * min(log_m**2, (log_m - log_rho_d) ** 2)
    torch.testing.assert_close(loss, torch.tensor(expected, dtype=torch.float64), atol=1e-10, rtol=0)


def test_mushy_middle_is_the_double_well_maximum() -> None:
    rho_d = 0.9
    beta = 1.0
    reg = SpectralSeparationReg(rho_d=rho_d, beta=beta)
    modulus = math.sqrt(rho_d)  # log midpoint between the two wells
    moduli = torch.tensor([modulus], dtype=torch.float64)
    loss = reg(moduli)

    log_rho_d = math.log(rho_d)
    expected = beta * (log_rho_d / 2) ** 2  # hinge is 0 here (modulus < 1)
    torch.testing.assert_close(loss, torch.tensor(expected, dtype=torch.float64), atol=1e-10, rtol=0)


def test_differentiable_wrt_moduli() -> None:
    reg = SpectralSeparationReg()
    moduli = torch.tensor([0.5, 1.05, 0.95], requires_grad=True)
    reg(moduli).backward()
    assert moduli.grad is not None
    assert torch.isfinite(moduli.grad).all()


def test_rho_d_domain_validated():
    # adversarial-review lock: rho_d outside (0, 1) silently breaks the
    # "zero exactly at the two wells" contract, so it must raise
    import pytest

    from sld.objectives.spectral_reg import SpectralSeparationReg

    for bad in (0.0, 1.0, 1.3, -0.2):
        with pytest.raises(ValueError):
            SpectralSeparationReg(rho_d=bad)
