"""Spectral-separation regularizer for the rotation-scaling predictor family.

This regularizer is an ABLATION of the spectral predictor family only: it
must never enter the common constraint factor shared across families (that
factor is the total-Jacobian penalty), and a spectrum shaped by it is a
display object, never the stability verdict.
"""

from __future__ import annotations

import torch
from torch import nn


class SpectralSeparationReg(nn.Module):
    """Double-well regularizer on the spectral moduli of a rotation-scaling block.

    Consumes the spectral-moduli vector ``moduli = |a_bar_i| > 0`` of a
    structured predictor (Wave 2: produced by
    ``RotationScalingBlocks.spectral_moduli()``; the coupling here is via a
    plain tensor, not a model dependency).

    forward(moduli):
        hinge = sum_i max(0, moduli_i - 1)^2          # nothing outside the unit circle
        log_m = log(clamp(moduli, min=1e-12))
        double_well = beta * sum_i min(log_m_i^2, (log_m_i - log(rho_d))^2)
        return hinge + double_well

    The double well parks each modulus near 1 (marginal/neutral mode) or near
    rho_d (contracting mode), penalizing the mushy middle between the two.
    """

    def __init__(self, rho_d: float = 0.9, beta: float = 1.0) -> None:
        super().__init__()
        # the two wells must be distinct and the rho well strictly inside the
        # unit circle, else the "zero exactly at {1, rho_d}" contract is false
        if not 0.0 < rho_d < 1.0:
            raise ValueError(f"rho_d must be in (0, 1), got {rho_d}")
        self.rho_d = rho_d
        self.beta = beta

    def forward(self, moduli: torch.Tensor) -> torch.Tensor:
        hinge = torch.clamp(moduli - 1.0, min=0.0).pow(2).sum()

        log_m = torch.clamp(moduli, min=1e-12).log()
        log_rho_d = torch.log(torch.as_tensor(self.rho_d, dtype=moduli.dtype, device=moduli.device))
        well_at_one = log_m.pow(2)
        well_at_rho_d = (log_m - log_rho_d).pow(2)
        double_well = self.beta * torch.minimum(well_at_one, well_at_rho_d).sum()

        return hinge + double_well
