"""Contract, property, and overfit-gate tests for the Mamba-style minimal
selective SSM predictor (``sld.predictors.min_ssm``).
"""

from __future__ import annotations

import time

import numpy as np
import torch

from sld.predictors.base import PredictorState
from sld.predictors.min_ssm import SelectiveSSMPredictor
from sld.systems.damped_pendulum import DampedPendulum

D_H, D_U, D_STATE = 2, 1, 8


def _make_model(seed: int, **kwargs) -> SelectiveSSMPredictor:
    # torch.random.fork_rng scopes the seed to module construction only --
    # the caller's global RNG state is restored on exit (never mutated)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return SelectiveSSMPredictor(**kwargs)


def test_shapes_and_state_threading():
    gen = torch.Generator().manual_seed(0)
    B, T = 4, 5
    model = _make_model(1, d_h=D_H, d_u=D_U, d_state=D_STATE)

    h0 = torch.randn(B, D_H, generator=gen)
    u_seq = torch.randn(T, B, D_U, generator=gen)

    # PredictorState(m=None) must be equivalent to an explicit zero state
    h_none, state_none = model.step(h0, u_seq[0], PredictorState(m=None))
    h_zero, state_zero = model.step(h0, u_seq[0], PredictorState(m=torch.zeros(B, D_STATE)))
    torch.testing.assert_close(h_none, h_zero)
    torch.testing.assert_close(state_none.m, state_zero.m)
    assert state_none.m.shape == (B, D_STATE)

    # m evolves across steps (not stuck at a fixed point for generic input)
    h1, state1 = model.step(h0, u_seq[1], state_none)
    assert not torch.allclose(state_none.m, state1.m)

    rollout_out = model.rollout(h0, u_seq)
    assert rollout_out.shape == (T, B, D_H)


def test_delta_and_a_bar_bounds():
    gen = torch.Generator().manual_seed(2)
    B = 6
    model = _make_model(3, d_h=D_H, d_u=D_U, d_state=D_STATE)

    h = torch.randn(B, D_H, generator=gen)
    m = torch.randn(B, D_STATE, generator=gen)
    u = torch.randn(B, D_U, generator=gen)
    A, _ = model.local_operator(h, u, PredictorState(m=m))
    A_bar = torch.diagonal(A, dim1=-2, dim2=-1)  # (B, d_state)

    assert torch.all(A_bar > 0.0) and torch.all(A_bar < 1.0)

    # invert A_bar_t = exp(delta_t * a_cont) to recover delta_t and check the
    # clamp bounds, without re-deriving delta_t through W_c/W_delta directly
    a_cont = -torch.exp(model.a_log.detach())
    delta_recovered = torch.log(A_bar) / a_cont
    assert torch.all(delta_recovered >= model.delta_min - 1e-9)
    assert torch.all(delta_recovered <= model.delta_max + 1e-9)


def test_selectivity_varies_with_state():
    gen = torch.Generator().manual_seed(4)
    model = _make_model(5, d_h=D_H, d_u=D_U, d_state=D_STATE)

    m = torch.randn(1, D_STATE, generator=gen)
    u = torch.randn(1, D_U, generator=gen)
    h_a = torch.randn(1, D_H, generator=gen)
    h_b = torch.randn(1, D_H, generator=gen)

    A_a, _ = model.local_operator(h_a, u, PredictorState(m=m))
    A_b, _ = model.local_operator(h_b, u, PredictorState(m=m))
    assert not torch.allclose(A_a, A_b), "A_bar_t must vary with h_t (LPV field, not fixed)"


def test_local_operator_coherence():
    """A, B from local_operator must match the autograd Jacobian of the
    frozen recurrence (c held constant), relative error < 1e-5 -- the wave
    gate. Uses B=1 so torch.func.jacrev's batch Jacobian reduces to the
    single-sample block directly (no cross-batch terms to discard).
    """
    gen = torch.Generator().manual_seed(6)
    model = _make_model(7, d_h=D_H, d_u=D_U, d_state=D_STATE).double()

    h = torch.randn(1, D_H, generator=gen, dtype=torch.float64)
    m = torch.randn(1, D_STATE, generator=gen, dtype=torch.float64)
    u = torch.randn(1, D_U, generator=gen, dtype=torch.float64)
    c = model.W_c(torch.cat([h, m], dim=-1)).detach()

    jac_A = torch.func.jacrev(lambda m_: model._frozen_recurrence(m_, h, u, c))(m)
    jac_B = torch.func.jacrev(lambda u_: model._frozen_recurrence(m, h, u_, c))(u)
    jac_A = jac_A[0, :, 0, :]
    jac_B = jac_B[0, :, 0, :]

    A, B = model.local_operator(h, u, PredictorState(m=m))
    err_A = (jac_A - A[0]).abs().max().item() / A[0].abs().max().item()
    err_B = (jac_B - B[0]).abs().max().item() / B[0].abs().max().item()
    print(f"[coherence] err_A={err_A:.3e} err_B={err_B:.3e}")
    assert err_A < 1e-5
    assert err_B < 1e-5


def test_frozen_differs_from_raw_autograd():
    # sanity check: freezing c must actually change the Jacobian w.r.t. m --
    # otherwise local_operator's "frozen-context" label would be vacuous
    gen = torch.Generator().manual_seed(8)
    model = _make_model(9, d_h=D_H, d_u=D_U, d_state=D_STATE).double()

    h = torch.randn(1, D_H, generator=gen, dtype=torch.float64)
    m = torch.randn(1, D_STATE, generator=gen, dtype=torch.float64)
    u = torch.randn(1, D_U, generator=gen, dtype=torch.float64)

    def full_step_m(m_):
        c = model.W_c(torch.cat([h, m_], dim=-1))
        return model._frozen_recurrence(m_, h, u, c)

    jac_full = torch.func.jacrev(full_step_m)(m)[0, :, 0, :]
    A, _ = model.local_operator(h, u, PredictorState(m=m))
    diff = (jac_full - A[0]).abs().max().item()
    print(f"[frozen-vs-raw] max abs diff = {diff:.3e}")
    assert diff > 1e-6


def _pendulum_trajectory(seed: int, T: int = 100, dt: float = 0.05):
    system = DampedPendulum()
    rng = np.random.default_rng(seed)
    x0 = system.sample_ic(rng)
    # iid clipped actions: normal draws clipped to the actuator range
    u_seq = np.clip(rng.normal(scale=1.0, size=(T, 1)), -system.u_max, system.u_max)
    traj = system.integrate(x0, u_seq, dt=dt)  # (T+1, 2)
    h_true = torch.as_tensor(traj, dtype=torch.float32).unsqueeze(1)  # (T+1, 1, 2)
    u = torch.as_tensor(u_seq, dtype=torch.float32).unsqueeze(1)  # (T, 1, 1)
    return h_true, u


def _teacher_forced_scan(model: SelectiveSSMPredictor, h_true: torch.Tensor, u: torch.Tensor):
    # teacher forcing on h (ground truth fed at every step) while m is
    # threaded across the scan (never reset) -- see test docstring below
    state = PredictorState()
    preds = []
    for t in range(u.shape[0]):
        h_next, state = model.step(h_true[t], u[t], state)
        preds.append(h_next)
    return torch.stack(preds, dim=0)


def test_overfit_one_pendulum_trajectory():
    """Overfit gate (wave requirement): free-running ``model.rollout`` was
    tried first and diverges to NaN within a few epochs at this learning
    rate (the residual readout ``h_next = h + W_out(m_next)`` compounds
    error over T=100 steps with nothing bounding ``h``), so this test uses
    the documented fallback -- teacher forcing on h (the true state is fed
    at every step) while the recurrent state m is threaded across the whole
    scan, never reset per step. This still exercises the selective-SSM
    recurrence end to end (m's evolution depends on the real trajectory
    history through c_t), it just removes h-space error compounding from
    the loss the optimizer sees.
    """
    h_true, u = _pendulum_trajectory(seed=0)
    model = _make_model(10, d_h=D_H, d_u=D_U, d_state=D_STATE)

    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    start = time.time()
    loss = torch.tensor(float("inf"))
    for _ in range(300):
        opt.zero_grad()
        preds = _teacher_forced_scan(model, h_true, u)
        loss = torch.nn.functional.mse_loss(preds, h_true[1:])
        loss.backward()
        opt.step()
        if loss.item() < 1e-3:
            break
    elapsed = time.time() - start

    print(f"[overfit] final_loss={loss.item():.3e} elapsed={elapsed:.1f}s")
    assert elapsed < 60.0
    assert loss.item() < 1e-3


def test_bounds_validated_and_default_init_finite_rollout():
    import pytest

    # adversarial review round 1 (two locks):
    # 1) NaN / inverted / non-positive step-size bounds must raise.
    for kwargs in (
        {"delta_min": float("nan")},
        {"delta_max": float("inf")},
        {"delta_min": 0.0},
        {"delta_min": 2.0, "delta_max": 1.0},
    ):
        with pytest.raises(ValueError):
            SelectiveSSMPredictor(d_h=2, d_u=1, d_state=8, **kwargs)
    # 2) the DEFAULT construction is finite over a long free rollout: W_out
    # is zero-initialized (h_next == h at init), fixing the expansive
    # residual loop the old default init produced.
    with torch.random.fork_rng():
        torch.manual_seed(7)
        model = SelectiveSSMPredictor(d_h=4, d_u=2, d_state=16)
    gen = torch.Generator().manual_seed(8)
    h0 = torch.randn(5, 4, generator=gen)
    u_seq = torch.randn(200, 5, 2, generator=gen)
    # thread step manually so the INTERNAL state m is checked too: h alone
    # can look finite while m overflows (0 * inf -> NaN one step later) --
    # adversarial review round 2. Claim locked here: 200 steps at default
    # init, full state (h, m); free-running stability beyond this is an
    # audit object, not a construction guarantee.
    h, state = h0, PredictorState()
    for t in range(u_seq.shape[0]):
        h, state = model.step(h, u_seq[t], state)
        assert torch.isfinite(h).all(), f"h nonfinite at step {t}"
        assert torch.isfinite(state.m).all(), f"m nonfinite at step {t}"
    print(
        f"[default-init] 200-step max |h| = {h.abs().max().item():.3e}, "
        f"max |m| = {state.m.abs().max().item():.3e}"
    )
