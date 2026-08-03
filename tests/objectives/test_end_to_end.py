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
from sld.encoders.frame_mlp import FrameMLPEncoder
from sld.predictors.base import PredictorState
from sld.predictors.mlp import MLPPredictor


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

    # every reported number evaluated on the SAME final model state
    # (adversarial-review fix: loss was recorded pre-update, stats post-)
    with torch.no_grad():
        z = encoder(x)
        pred_value = jepa(predictor(z[:-1], u), target(x[1:]))["pred"].item()
        # identity baseline = the EXACT loss of a zero-residual predictor:
        # online embeddings passed through unchanged, regressed on the same
        # targets (adversarial-review fix: using target(x[:-1]) here would
        # only coincide with this when online == EMA target)
        identity_baseline = jepa(z[:-1], target(x[1:]))["pred"].item()
        # action sensitivity: a predictor that actually uses u must degrade
        # when u is shuffled in time (a u-ignoring interpolator would not)
        u_shuffled = u[torch.randperm(len(u), generator=torch.Generator().manual_seed(0))]
        pred_shuffled = jepa(predictor(z[:-1], u_shuffled), target(x[1:]))["pred"].item()
        per_dim_std = z.std(dim=0)
        # participation ratio of the latent covariance spectrum: a rank-one
        # embedding (all dims copies of one signal) scores 1, a spread one
        # scores near the latent dim — kills the collapse construction that
        # passed the earlier mean-std check (adversarial review)
        z_centered = z - z.mean(dim=0, keepdim=True)
        eigvals = torch.linalg.eigvalsh(z_centered.T @ z_centered / (len(z) - 1))
        participation_ratio = eigvals.sum().pow(2) / eigvals.pow(2).sum()

    print(
        f"[end-to-end] pred={pred_value:.2e} identity={identity_baseline:.2e} "
        f"ratio={pred_value / identity_baseline:.3f} "
        f"shuffled-u={pred_shuffled:.2e} min_std={per_dim_std.min():.3f} "
        f"PR={participation_ratio:.2f}"
    )
    # the Wave 2 gate, hardened (scope: single-trajectory PLUMBING gate,
    # deliberately overfit — generalization is Wave 3+ territory): the
    # predictor must beat the exact zero-residual identity baseline by 10x
    # (identity solutions score ratio 1), degrade by 3x when actions are
    # time-shuffled (a u-ignoring interpolator would not), keep every
    # latent dimension alive, and spread covariance across dimensions
    assert pred_value < 1e-4
    assert pred_value < 0.1 * identity_baseline
    assert pred_shuffled > 3.0 * pred_value
    assert per_dim_std.min().item() > 0.1
    assert participation_ratio.item() > 4.0


GAMMA_JEPA = 10.0  # declared before measuring anything (REGLES 7.2), never revised


def test_jepa_overfits_one_trajectory_through_production_classes():
    """Same overfit as ``test_jepa_mlp_overfits_one_trajectory`` above, but
    through the PRODUCTION classes (REGLES 7.1, gap closed 2026-08-03).

    That test builds ``_Encoder`` and ``_ResidualMLPPredictor`` locally, so it
    proves the OBJECTIVE is optimizable in principle and nothing about the code
    path the project actually runs. Here the encoder is
    ``sld.encoders.frame_mlp.FrameMLPEncoder`` and the predictor is
    ``sld.predictors.mlp.MLPPredictor``, driven through its real
    ``step(h, u, PredictorState())`` contract. The local test is kept: it
    isolates the objective, this one exercises the wiring.

    Gate: the JEPA prediction loss must beat the IDENTITY baseline -- the exact
    loss of a zero-residual predictor on the same targets -- by GAMMA_JEPA.
    An absolute threshold would say nothing about whether the model beat doing
    nothing; the same defect this session found in the four families' overfit
    gates, where persistence itself cleared the declared bar on koopman_ssm.
    """
    torch.manual_seed(0)
    x, u = _one_pendulum_trajectory()

    encoder = FrameMLPEncoder(d_x=2, d_z=8, hidden=(32,))
    predictor = MLPPredictor(d_h=8, d_u=1, hidden=(64,))
    target = EMATarget(encoder, tau=0.99)
    jepa = JEPALoss()
    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()), lr=2e-3
    )

    for _ in range(4000):
        opt.zero_grad()
        z = encoder(x)
        h_hat, _ = predictor.step(z[:-1], u, PredictorState())
        losses = jepa(h_hat, target(x[1:]))
        total = losses["total"] + 0.1 * variance_loss(z) + 0.01 * covariance_loss(z)
        total.backward()
        opt.step()
        target.update(encoder)

    with torch.no_grad():
        z = encoder(x)
        h_hat, _ = predictor.step(z[:-1], u, PredictorState())
        pred_value = jepa(h_hat, target(x[1:]))["pred"].item()
        identity_baseline = jepa(z[:-1], target(x[1:]))["pred"].item()
        u_shuffled = u[torch.randperm(len(u), generator=torch.Generator().manual_seed(0))]
        h_shuf, _ = predictor.step(z[:-1], u_shuffled, PredictorState())
        pred_shuffled = jepa(h_shuf, target(x[1:]))["pred"].item()

    factor = identity_baseline / max(pred_value, 1e-30)
    print(
        f"[jepa-production] pred={pred_value:.4e} identity={identity_baseline:.4e} "
        f"factor={factor:.1f}x shuffled-u={pred_shuffled:.4e}"
    )
    assert pred_value < identity_baseline / GAMMA_JEPA

    # Negative 1: the identity baseline itself must FAIL the gate -- it scores
    # exactly 1x by construction, which is precisely the point.
    assert not (identity_baseline < identity_baseline / GAMMA_JEPA)
    # Negative 2: a predictor that ignores u would be insensitive to shuffling
    # it in time. Degradation proves the action path is live.
    assert pred_shuffled > pred_value


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
