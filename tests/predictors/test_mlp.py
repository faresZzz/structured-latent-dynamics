import numpy as np
import pytest
import torch

from sld.data.regimes import OURegime
from sld.predictors.base import PredictorState
from sld.predictors.mlp import MLPPredictor
from sld.systems.damped_pendulum import DampedPendulum

D_H, D_U = 2, 1


def _make(seed: int) -> MLPPredictor:
    # nn.Module init reads torch's global RNG internally with no generator
    # hook; fork_rng() seeds it locally for this construction only and
    # restores the prior global state on exit, so tests stay independent.
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return MLPPredictor(d_h=D_H, d_u=D_U)


def test_step_shapes_and_state_stays_none():
    model = _make(0)
    B = 5
    gen = torch.Generator().manual_seed(1)
    h = torch.randn(B, D_H, generator=gen)
    u = torch.randn(B, D_U, generator=gen)
    h_next, state = model.step(h, u, PredictorState())
    assert h_next.shape == (B, D_H)
    assert state.m is None


def test_rollout_shape_and_state_never_touched():
    model = _make(1)
    B, T = 3, 7
    gen = torch.Generator().manual_seed(2)
    h0 = torch.randn(B, D_H, generator=gen)
    u_seq = torch.randn(T, B, D_U, generator=gen)
    out = model.rollout(h0, u_seq)
    assert out.shape == (T, B, D_H)
    # rolling out manually with an explicitly driven state must also keep m=None
    h, state = h0, PredictorState()
    for t in range(T):
        h, state = model.step(h, u_seq[t], state)
        assert state.m is None


def test_local_operator_is_generic_none():
    model = _make(2)
    gen = torch.Generator().manual_seed(3)
    h = torch.randn(2, D_H, generator=gen)
    u = torch.randn(2, D_U, generator=gen)
    assert model.local_operator(h, u, PredictorState()) is None


def test_long_rollout_has_no_cache_to_overflow():
    # no state means no cache to overflow -- a long rollout must just work
    model = _make(3)
    B, T = 2, 300
    gen = torch.Generator().manual_seed(4)
    h0 = torch.randn(B, D_H, generator=gen)
    u_seq = torch.randn(T, B, D_U, generator=gen)
    out = model.rollout(h0, u_seq)
    assert out.shape == (T, B, D_H)
    assert torch.all(torch.isfinite(out))


def _one_pendulum_trajectory(T: int = 100, dt: float = 0.05, seed: int = 0):
    system = DampedPendulum()
    rng = np.random.default_rng(seed)  # local Generator, no global numpy RNG touched
    x0, u_seq_np = OURegime().sample_episode(system, T=T, dt=dt, rng=rng)
    traj = system.integrate(x0, u_seq_np, dt=dt)  # (T+1, state_dim), float64
    h = torch.as_tensor(traj, dtype=torch.float32)
    u = torch.as_tensor(u_seq_np, dtype=torch.float32)
    return h, u


def test_overfit_one_trajectory():
    h, u = _one_pendulum_trajectory()
    model = _make(seed=0)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)

    loss = torch.tensor(float("inf"))
    for _ in range(600):
        opt.zero_grad()
        h_pred, _ = model.step(h[:-1], u, PredictorState())  # teacher-forced one-step
        loss = torch.mean((h_pred - h[1:]) ** 2)
        loss.backward()
        opt.step()

    print(f"[overfit-mlp] final teacher-forced one-step MSE = {loss.item():.3e}")
    assert loss.item() < 1e-3


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
