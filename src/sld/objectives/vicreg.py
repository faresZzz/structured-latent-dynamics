"""VICReg variance and covariance regularizers.

Bardes, Ponce & LeCun, "VICReg: Variance-Invariance-Covariance Regularization
for Self-Supervised Learning", arXiv:2105.04906
(https://arxiv.org/abs/2105.04906). Used here as the anti-collapse terms:
variance PENALIZES collapsed dimensions, covariance decorrelates them.

Honest limit (adversarial-review note): at an EXACTLY constant coordinate
the variance-hinge gradient is exactly zero (d var/dz vanishes at equality),
so exact collapse is a zero-gradient stationary point these terms cannot
escape — they make collapse unattractive, they do not guarantee escape from
it. Exact collapse requires pre-activations exactly constant across the
batch (zero weights, saturated activations, or data in a weight nullspace);
the end-to-end gate checks the trained latent variance directly.
"""

from __future__ import annotations

import torch


def variance_loss(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """Hinge loss on the per-dimension standard deviation.

    z: (..., d), any leading dims are flattened into the batch dimension N.
    Returns mean_j max(0, gamma - sqrt(var_j(z) + eps)).
    """
    z = z.reshape(-1, z.shape[-1])
    if z.shape[0] < 2:
        raise ValueError("variance_loss needs N >= 2 samples (unbiased var is NaN at N=1)")
    std = torch.sqrt(z.var(dim=0, unbiased=True) + eps)
    return torch.clamp(gamma - std, min=0.0).mean()


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """Off-diagonal penalty on the centered covariance matrix.

    z: (..., d), any leading dims are flattened into the batch dimension N.
    C = (z - mean)^T (z - mean) / (N - 1); returns (1/d) * sum_{i != j} C_ij^2.
    """
    z = z.reshape(-1, z.shape[-1])
    n, d = z.shape
    if n < 2:
        raise ValueError("covariance_loss needs N >= 2 samples (C is undefined at N=1)")
    z_centered = z - z.mean(dim=0, keepdim=True)
    cov = (z_centered.T @ z_centered) / (n - 1)
    off_diag_sq = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    return off_diag_sq / d
