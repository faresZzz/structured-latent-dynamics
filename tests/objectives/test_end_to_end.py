"""End-to-end plumbing proof (Wave 2 gate): JEPA + a minimal latent MLP
predictor overfit one pendulum trajectory.

The predictor family modules arrive in Wave 3; the predictor here is a
test-local minimal MLP in z-space (h = z, no lifting). The test asserts BOTH
the prediction loss floor (< 1e-4) and a latent-variance floor — without the
second assert, a collapsed constant encoder would pass the first trivially,
which is exactly the failure mode `test_collapse_without_vicreg_is_trivial`
demonstrates constructively below.
"""

import numpy as np
import torch
from torch import nn

from sld.data.regimes import OURegime
from sld.objectives.jepa import EMATarget, JEPALoss
from sld.objectives.vicreg import covariance_loss, variance_loss
from sld.systems.damped_pendulum import DampedPendulum


def _one_pendulum_trajectory():
    system = DampedPendulum()
    rng = np.random.default_rng(0)
    x0, u_seq = OURegime().sample_episode(system, T=200, dt=0.02, rng=rng)
    traj = system.integrate(x0, u_seq, dt=0.02)
    x = torch.as_tensor(traj, dtype=torch.float32)          # (201, 2)
    u = torch.as_tensor(u_seq, dtype=torch.float32)         # (200, 1)
    return x, u


class _Encoder(nn.Module):
    def __init__(self, z_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, z_dim))

    def forward(self, x):
        return self.net(x)


class _ResidualMLPPredictor(nn.Module):
    # minimal z-space predictor: z_hat_{t+1} = z_t + MLP([z_t; u_t])
    def __init__(self, z_dim: int = 8, u_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + u_dim, 64), nn.Tanh(), nn.Linear(64, z_dim)
        )

    def forward(self, z, u):
        return z + self.net(torch.cat([z, u], dim=-1))


def test_jepa_mlp_overfits_one_trajectory():
    torch.manual_seed(0)
    x, u = _one_pendulum_trajectory()

    encoder = _Encoder()
    predictor = _ResidualMLPPredictor()
    target = EMATarget(encoder, tau=0.99)
    jepa = JEPALoss()
    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()), lr=2e-3
    )

    pred_value = float("inf")
    for _ in range(4000):
        opt.zero_grad()
        z = encoder(x)                                   # (201, z)
        h_hat = predictor(z[:-1], u)                     # one-step prediction
        z_target = target(x[1:])
        losses = jepa(h_hat, z_target)
        # anti-collapse terms on the online embeddings (weights declared)
        total = losses["total"] + 0.1 * variance_loss(z) + 0.01 * covariance_loss(z)
        total.backward()
        opt.step()
        target.update(encoder)
        pred_value = losses["pred"].item()

    latent_std = encoder(x).std(dim=0).mean().item()
    print(f"[end-to-end] pred={pred_value:.2e} latent_std={latent_std:.3f}")
    # the Wave 2 gate: plumbing overfits one trajectory below 1e-4 while the
    # latent space stays alive (a collapsed encoder would fail the std floor)
    assert pred_value < 1e-4
    assert latent_std > 0.1


def test_collapse_without_vicreg_is_trivial():
    # constructive doctrine test: with a CONSTANT encoder (zeroed weights)
    # and no variance/covariance terms, the bare predictive loss reaches ~0
    # while the latent variance is exactly 0 — collapse minimizes the naked
    # loss, which is why VICReg terms guard every arm of the factorial
    torch.manual_seed(0)
    x, u = _one_pendulum_trajectory()

    encoder = _Encoder()
    with torch.no_grad():
        for p in encoder.parameters():
            p.zero_()
    for p in encoder.parameters():
        p.requires_grad_(False)

    predictor = _ResidualMLPPredictor()
    target = EMATarget(encoder, tau=0.99)
    jepa = JEPALoss()
    opt = torch.optim.Adam(predictor.parameters(), lr=1e-2)

    for _ in range(200):
        opt.zero_grad()
        z = encoder(x)
        h_hat = predictor(z[:-1], u)
        losses = jepa(h_hat, target(x[1:]))
        losses["total"].backward()
        opt.step()

    assert losses["pred"].item() < 1e-6
    assert encoder(x).std(dim=0).mean().item() == 0.0
