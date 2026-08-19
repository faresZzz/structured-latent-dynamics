"""Tests for sld.experiments.gate: Wave D3a's free-running evaluation path
-- from a trained run/frozen representation to the e_h curve in TRUE-STATE
coordinates (PROTOCOL.md sections 12, 14).

This module reuses sld.audit.rollout's own primitives (normalized_state_error,
persistence_rollout, linear_extrapolation_rollout, vpt) as the reference for
"gate.py wired them correctly" -- R2's independent reference for the PHYSICS
itself is the hand-built exact-ZOH oracle below: a tiny linear system whose
discrete update is written in closed form directly in this test file, never
by calling anything under test.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from sld.audit.rollout import (
    evaluate_positive_control,
    fit_train_state_scale,
    linear_extrapolation_rollout,
    normalized_state_error,
    persistence_rollout,
)
from sld.audit.rollout import vpt as _vpt_reducer
from sld.encoders.pretrain import (
    load_frozen_representation,
    pretrain_representation,
    save_frozen_representation,
)
from sld.experiments import protocol as protocol_mod
from sld.experiments.factorial import _build_dataset_for, _stratum_representation_path
from sld.experiments.factorial import main as factorial_main
from sld.experiments.gate import (
    FreeRolloutResult,
    RolloutDivergence,
    evaluate_free_rollout,
    evaluate_gate_campaign,
    evaluate_run_free_rollout,
    read_gate_artifact,
    write_gate_artifact,
)
from sld.predictors.base import PredictorModule, PredictorState
from sld.training.trainer import data_fingerprint


class _IdentityEncoder(nn.Module):
    """obs == state, exactly -- no parameters, so it passes the audit's
    CPU-float64 check vacuously regardless of caller dtype."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _LinearOracle(PredictorModule):
    """``h_next = h @ A.T + u @ B.T``, fixed non-trainable float64 A/B --
    the hand-built reference for the exact-ZOH test, independent of
    sld.audit.rollout (R2). Inherits ``rollout`` unmodified from
    ``PredictorModule`` (base.py): sequential ``step`` calls, no lift/project.
    """

    def __init__(self, A: np.ndarray, B: np.ndarray):
        super().__init__()
        self.A = nn.Parameter(torch.as_tensor(A, dtype=torch.float64), requires_grad=False)
        self.B = nn.Parameter(torch.as_tensor(B, dtype=torch.float64), requires_grad=False)

    def step(self, h, u, state):
        return h @ self.A.T + u @ self.B.T, state


class _NaNPredictor(PredictorModule):
    def step(self, h, u, state):
        return h * float("nan"), state


class _GrowingPredictor(PredictorModule):
    """``h_next = h * growth`` every step -- deterministic explosive growth,
    independent of the toy dataset's random ``h0``, used to check
    ``RolloutDivergence.first_step`` against a hand-computed reference
    (REGLES section 7.1's corollary: the reference is made by hand, never
    the same code path as the object under test)."""

    def __init__(self, growth: float):
        super().__init__()
        self.growth = growth

    def step(self, h, u, state):
        return h * self.growth, state


def _toy_lti_split(A: np.ndarray, B: np.ndarray, *, n_episodes: int, T: int, seed: int) -> dict:
    """A tiny, hand-built LTI dataset: ``z_true[t+1] = A @ z_true[t] + B @ u[t]``,
    generated directly in float64 by the SAME exact matrix update a
    ZOH-discretized linear system has -- the update IS the closed-form
    discrete solution, not an approximation to one. ``obs == state``
    (identity), matching ``_IdentityEncoder`` above.
    """
    d_state = A.shape[0]
    d_u = B.shape[1]
    rng = np.random.default_rng(seed)
    z_true = np.empty((n_episodes, T + 1, d_state), dtype=np.float64)
    u = rng.normal(scale=0.3, size=(n_episodes, T, d_u))
    z_true[:, 0, :] = rng.normal(size=(n_episodes, d_state))
    for t in range(T):
        z_true[:, t + 1, :] = z_true[:, t, :] @ A.T + u[:, t, :] @ B.T
    return {"x": z_true.copy(), "u": u, "z_true": z_true}


def _identity_readout(d_state: int, bias: np.ndarray | None = None) -> np.ndarray:
    """``(d_state+1, d_state)``: identity decode block + a bias row (zero
    unless given)."""
    W = np.zeros((d_state + 1, d_state), dtype=np.float64)
    W[:d_state, :] = np.eye(d_state)
    if bias is not None:
        W[d_state, :] = bias
    return W


_A_TRUE = np.array([[0.85, 0.05], [0.0, 0.80]])
_B_TRUE = np.array([[0.4], [0.2]])
_SIGMA = np.array([1.0, 1.0])


def _kwargs(**overrides) -> dict:
    base = dict(horizon=8, tau=0.5, h_gate=5, gamma=5.0, tau_1=0.1)
    base.update(overrides)
    return base


# --------------------------------------------------------------------------
# Positive
# --------------------------------------------------------------------------


def test_exact_zoh_oracle_gives_zero_error():
    # R2: the reference is the closed-form matrix update itself (_toy_lti_split
    # / _LinearOracle above), built independently of sld.audit.rollout -- this
    # validates the plumbing against physics, not against a second
    # implementation of the same code.
    split = _toy_lti_split(_A_TRUE, _B_TRUE, n_episodes=5, T=12, seed=1)
    predictor = _LinearOracle(_A_TRUE, _B_TRUE)
    result = evaluate_free_rollout(
        _IdentityEncoder(), predictor, split, W=_identity_readout(2), sigma_train=_SIGMA, **_kwargs()
    )
    max_e = float(np.max(np.abs(result.e_h)))
    max_e_per_episode = float(np.max(np.abs(result.per_episode_e_h)))
    print(f"[gate] exact-ZOH oracle: max|e_h|={max_e:.3e} max|per_episode_e_h|={max_e_per_episode:.3e}")
    assert max_e < 1e-12
    assert max_e_per_episode < 1e-12


def test_perfect_model_verdict_learns_vpt_censored_every_episode():
    split = _toy_lti_split(_A_TRUE, _B_TRUE, n_episodes=5, T=12, seed=2)
    predictor = _LinearOracle(_A_TRUE, _B_TRUE)
    result = evaluate_free_rollout(
        _IdentityEncoder(), predictor, split, W=_identity_readout(2), sigma_train=_SIGMA, **_kwargs()
    )
    print(f"[gate] perfect model verdict={result.positive_control.verdict} vpt_status={result.vpt_status}")
    assert result.positive_control.verdict == "learns"
    assert result.positive_control.lower_bound
    assert result.positive_control.upper_bound
    assert np.all(result.vpt_status == "censored")
    assert np.all(np.isinf(result.vpt_value))
    assert np.isinf(result.vpt_median)
    assert result.vpt_censored == result.vpt_value.shape[0]


def test_baselines_match_direct_calls_on_the_same_inputs():
    split = _toy_lti_split(_A_TRUE, _B_TRUE, n_episodes=4, T=12, seed=3)
    predictor = _LinearOracle(_A_TRUE, _B_TRUE)
    kw = _kwargs()
    result = evaluate_free_rollout(
        _IdentityEncoder(), predictor, split, W=_identity_readout(2), sigma_train=_SIGMA, **kw
    )
    z_true = split["z_true"]
    x_minus_1, x0 = z_true[:, 0, :], z_true[:, 1, :]
    target = z_true[:, 2 : 2 + kw["horizon"], :]
    pers = normalized_state_error(persistence_rollout(x0, kw["horizon"]), target, _SIGMA)
    lin = normalized_state_error(
        linear_extrapolation_rollout(x_minus_1, x0, kw["horizon"]), target, _SIGMA
    )
    np.testing.assert_array_equal(result.persistence_e_h, pers)
    np.testing.assert_array_equal(result.linear_e_h, lin)


def test_unit_invariance_under_adversarial_per_coordinate_rescale():
    split = _toy_lti_split(_A_TRUE, _B_TRUE, n_episodes=4, T=12, seed=4)
    predictor = _LinearOracle(_A_TRUE, _B_TRUE)
    kw = _kwargs()
    base = evaluate_free_rollout(
        _IdentityEncoder(), predictor, split, W=_identity_readout(2), sigma_train=_SIGMA, **kw
    )

    scale = np.array([7.3, 0.0021])  # adversarial: wildly different per-coordinate magnitudes
    scaled_split = dict(split)
    scaled_split["z_true"] = split["z_true"] * scale  # state-space coordinates only; x/u untouched
    scaled_W = _identity_readout(2) * scale[None, :]  # each OUTPUT (state) column rescaled
    scaled_sigma = _SIGMA * scale

    scaled = evaluate_free_rollout(
        _IdentityEncoder(), predictor, scaled_split, W=scaled_W, sigma_train=scaled_sigma, **kw
    )
    print(f"[gate] unit invariance: base.e_h={base.e_h[:3]} scaled.e_h={scaled.e_h[:3]}")
    np.testing.assert_allclose(scaled.e_h, base.e_h, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(scaled.persistence_e_h, base.persistence_e_h, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(scaled.linear_e_h, base.linear_e_h, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(scaled.vpt_value, base.vpt_value, atol=0, rtol=0)


def test_per_episode_e_h_is_the_source_e_h_is_derived_from():
    # requirement 7 ("one e_h table, one source"): the aggregate e_h is the
    # exact RMS-combination of per_episode_e_h over episodes -- structurally
    # one array, not two independently computed curves.
    split = _toy_lti_split(_A_TRUE, _B_TRUE, n_episodes=6, T=12, seed=9)
    predictor = _LinearOracle(np.array([[0.7, 0.1], [0.0, 0.6]]), _B_TRUE)  # NOT exact: e_h != 0
    result = evaluate_free_rollout(
        _IdentityEncoder(), predictor, split, W=_identity_readout(2), sigma_train=_SIGMA, **_kwargs()
    )
    combined = np.sqrt(np.mean(result.per_episode_e_h**2, axis=0))
    np.testing.assert_allclose(combined, result.e_h, atol=1e-10, rtol=1e-10)


def test_rollout_is_called_exactly_once(monkeypatch):
    # class-level spy (not instance-level): evaluate_run_free_rollout would
    # deep-copy the predictor before rolling out, so an instance-level spy
    # would silently observe zero calls on the ORIGINAL object and prove
    # nothing; this function does not deep-copy, but the spy is written at
    # the class level so it also covers that caller.
    import sld.predictors.base as base_mod

    calls = []
    real_rollout = base_mod.PredictorModule.rollout

    def _counting_rollout(self, h0, u_seq):
        calls.append(1)
        return real_rollout(self, h0, u_seq)

    monkeypatch.setattr(base_mod.PredictorModule, "rollout", _counting_rollout)

    split = _toy_lti_split(_A_TRUE, _B_TRUE, n_episodes=4, T=12, seed=10)
    predictor = _LinearOracle(_A_TRUE, _B_TRUE)
    evaluate_free_rollout(
        _IdentityEncoder(), predictor, split, W=_identity_readout(2), sigma_train=_SIGMA, **_kwargs()
    )
    print(f"[gate] predictor.rollout call count = {len(calls)}")
    assert len(calls) == 1


# --------------------------------------------------------------------------
# Negative
# --------------------------------------------------------------------------


def test_teacher_forcing_curve_differs_from_free_running():
    # The negative that matters most (REGLES section 7.1): an unstable predictor
    # (larger spectral radius than the TRUE system) compounds every step's
    # error exponentially under FREE running, while teacher-forcing resets
    # to the true state every step and only ever sees ONE step of the
    # mismatch -- the two curves cannot coincide.
    A_wrong = np.array([[1.3, 0.0], [0.0, 1.25]])
    split = _toy_lti_split(_A_TRUE, _B_TRUE, n_episodes=4, T=12, seed=5)
    predictor = _LinearOracle(A_wrong, _B_TRUE)
    kw = _kwargs(tau_1=10.0, gamma=1.0)
    H = kw["horizon"]

    fr = evaluate_free_rollout(
        _IdentityEncoder(), predictor, split, W=_identity_readout(2), sigma_train=_SIGMA, **kw
    )

    # Independent teacher-forced curve, threaded over the TRUE prefix -- the
    # exact call-site pattern Trainer._composite_loss's TF term uses
    # (predictor.step looped over h_true[t]); built ONLY for this comparison,
    # never inside gate.py, and never accepted as its input.
    z_true = split["z_true"]
    u = split["u"]
    h_true = torch.as_tensor(z_true[:, 1 : 1 + H, :], dtype=torch.float64)
    u_t = torch.as_tensor(u[:, 1 : 1 + H, :], dtype=torch.float64)
    state = PredictorState()
    preds = []
    for t in range(H):
        h_next, state = predictor.step(h_true[:, t, :], u_t[:, t, :], state)
        preds.append(h_next)
    tf_pred_x = np.transpose(torch.stack(preds, dim=0).detach().numpy(), (1, 0, 2))
    target = z_true[:, 2 : 2 + H, :]
    tf_e_h = normalized_state_error(tf_pred_x, target, _SIGMA)

    print(f"[gate] teacher-forced e_h={tf_e_h}\n       free-running   e_h={fr.e_h}")
    assert not np.allclose(tf_e_h, fr.e_h, atol=1e-6)
    assert fr.e_h[-1] > tf_e_h[-1] * 5  # free-running is the exponentially worse one


def test_split_shorter_than_horizon_is_refused_not_truncated():
    split = _toy_lti_split(_A_TRUE, _B_TRUE, n_episodes=3, T=5, seed=6)  # T+1 = 6 steps
    predictor = _LinearOracle(_A_TRUE, _B_TRUE)
    with pytest.raises(ValueError, match="fewer than the"):
        evaluate_free_rollout(
            _IdentityEncoder(),
            predictor,
            split,
            W=_identity_readout(2),
            sigma_train=_SIGMA,
            **_kwargs(horizon=8),
        )


def test_frozen_readout_and_sigma_are_used_never_refit_on_eval_split():
    split = _toy_lti_split(_A_TRUE, _B_TRUE, n_episodes=5, T=12, seed=7)
    predictor = _LinearOracle(_A_TRUE, _B_TRUE)  # exact oracle: decoded pred_z == target exactly
    bias = np.array([5.0, -3.0])  # deliberately WRONG constant offset
    W_frozen = _identity_readout(2, bias=bias)
    kw = _kwargs(tau_1=100.0)  # loose enough the offset doesn't block the gate arithmetic

    result = evaluate_free_rollout(
        _IdentityEncoder(), predictor, split, W=W_frozen, sigma_train=_SIGMA, **kw
    )
    target = split["z_true"][:, 2 : 2 + kw["horizon"], :]

    # W: the decoded prediction must carry the FROZEN bias exactly -- an OLS
    # refit on this split (state == observation, truly zero bias) would find
    # ~zero, not this constant.
    offset = result.model_pred_x - target
    np.testing.assert_allclose(offset, np.broadcast_to(bias, offset.shape), atol=1e-9)

    # sigma: result.e_h must equal normalized_state_error computed with the
    # SUPPLIED sigma, and must differ from what a refit-on-eval-split sigma
    # would give -- the two are shown to actually disagree, so matching the
    # frozen one is informative, not a coincidence.
    expected = normalized_state_error(result.model_pred_x, target, _SIGMA)
    np.testing.assert_array_equal(result.e_h, expected)
    refit_sigma = fit_train_state_scale(target.reshape(-1, target.shape[-1]))
    assert not np.allclose(_SIGMA, refit_sigma)
    refit_e_h = normalized_state_error(result.model_pred_x, target, refit_sigma)
    assert not np.allclose(result.e_h, refit_e_h)


def test_non_finite_curve_is_refused():
    # CORR-2/15.3: the model's own rollout going non-finite is now TYPED
    # (RolloutDivergence, a ValueError subclass -- this pytest.raises still
    # catches it), with the first diverged step recorded structurally rather
    # than only in the message.
    split = _toy_lti_split(_A_TRUE, _B_TRUE, n_episodes=3, T=12, seed=8)
    predictor = _NaNPredictor()
    with pytest.raises(ValueError, match="non-finite") as excinfo:
        evaluate_free_rollout(
            _IdentityEncoder(), predictor, split, W=_identity_readout(2), sigma_train=_SIGMA, **_kwargs()
        )
    assert isinstance(excinfo.value, RolloutDivergence)
    assert excinfo.value.first_step == 1  # h_next is nan from the FIRST step onward


def test_typed_divergence_first_step_matches_a_hand_computed_reference():
    """An explosive (but not immediately NaN) predictor: RolloutDivergence's
    ``first_step`` must match the step at which the SAME growth law,
    replayed by hand directly in this test (never via the predictor class or
    gate.py), first leaves float64's finite range."""
    growth = 10.0**40
    split = _toy_lti_split(_A_TRUE, _B_TRUE, n_episodes=3, T=20, seed=20)
    predictor = _GrowingPredictor(growth)
    kw = _kwargs(horizon=12)

    with pytest.raises(RolloutDivergence) as excinfo:
        evaluate_free_rollout(
            _IdentityEncoder(), predictor, split, W=_identity_readout(2), sigma_train=_SIGMA, **kw
        )

    # hand-computed reference: model_pred_x[b, h, :] == x0_obs[b, :] * growth**(h+1)
    # exactly (identity encoder, no lift, identity readout) -- replay that
    # growth law directly with plain numpy, independent of the predictor
    # class and of gate.py's own code path.
    x0_obs = split["x"][:, 1, :].astype(np.float64)
    val = x0_obs.copy()
    expected_first_step = None
    for step in range(1, kw["horizon"] + 1):
        val = val * growth
        if not np.isfinite(val).all():
            expected_first_step = step
            break
    assert expected_first_step is not None, "positive control: growth must actually overflow within horizon"
    print(f"[gate] typed divergence: first_step={excinfo.value.first_step} expected={expected_first_step}")
    assert excinfo.value.first_step == expected_first_step


# --------------------------------------------------------------------------
# evaluate_run_free_rollout: the run_dir/representation wrapper
# --------------------------------------------------------------------------


def _run_config(tmp_path: Path, **overrides) -> dict:
    config = {
        "exp_name": "gate_test",
        "seed": 0,
        "family": "mlp",
        "d_x": 2,
        "d_z": 4,
        "d_u": 1,
        "encoder_hidden": [8],
        "target_params": 2_000,
        "epochs": 2,
        "batch_episodes": 4,
        "lr": 5e-2,
        "h_fr": 5,
        "results_root": str(tmp_path / "results"),
        "stratum_system": "pendulum",
        "stratum_regime": "ou",
    }
    config.update(overrides)
    return config


def _small_representation(tmp_path, data, *, seed: int, name: str):
    artifact = pretrain_representation(
        data, system="pendulum", regime="ou", d_z=4, encoder_hidden=(8,), seed=seed, epochs=2,
        batch_size=16,
    )
    path = tmp_path / f"{name}.pt"
    save_frozen_representation(artifact, path)
    return load_frozen_representation(path)


# ==============================================================================
# gate.json persistence + campaign (Wave: gate persistence). Everything above
# this point tests the pure evaluation core; everything below tests
# write_gate_artifact/read_gate_artifact/evaluate_gate_campaign -- the
# persistence half of gate.py.
# ==============================================================================


def _trained_run_dir(tmp_path: Path, *, exp_name: str, seed: int = 0) -> Path:
    """A cheap, REAL completed run dir (valid run_complete.json + config.yaml
    on disk) for write_gate_artifact/read_gate_artifact tests. Deliberately
    tiny (T=10) -- these tests attach their OWN FreeRolloutResult/
    RolloutDivergence objects rather than calling evaluate_run_free_rollout
    on this run, so it never needs to satisfy H_GATE's horizon requirement."""
    from sld.data.generate import build_dataset
    from sld.data.regimes import OURegime
    from sld.systems.damped_pendulum import DampedPendulum
    from sld.training.trainer import Trainer

    data = build_dataset(
        system=DampedPendulum(), regime=OURegime(), obs=None, n_episodes=6, T=10, dt=0.02, seed=0
    )
    art = _small_representation(tmp_path, data, seed=1, name=exp_name)
    trainer = Trainer(_run_config(tmp_path, exp_name=exp_name, seed=seed), representation=art)
    result = trainer.fit(data)
    return Path(result["paths"]["ckpt"]).parent


def _positive_control_result(e_h, persistence_e_h, linear_e_h, *, h_gate, gamma, tau_1, tau=0.5):
    """A REAL ``FreeRolloutResult``, built from hand-picked curves through the
    REAL ``evaluate_positive_control``/``vpt`` (R2: the gate arithmetic is
    never reimplemented here) -- lets a test pick exactly which bound passes
    without needing a predictor whose rollout happens to land there."""
    e_h = np.asarray(e_h, dtype=np.float64)
    persistence_e_h = np.asarray(persistence_e_h, dtype=np.float64)
    linear_e_h = np.asarray(linear_e_h, dtype=np.float64)
    pc = evaluate_positive_control(e_h, persistence_e_h, linear_e_h, h_gate=h_gate, gamma=gamma, tau_1=tau_1)
    per_episode_e_h = e_h[None, :]
    vpt_result = _vpt_reducer(per_episode_e_h, tau)
    return FreeRolloutResult(
        e_h=e_h,
        persistence_e_h=persistence_e_h,
        linear_e_h=linear_e_h,
        per_episode_e_h=per_episode_e_h,
        model_pred_x=np.zeros((1, e_h.shape[0], 1)),
        vpt_value=vpt_result.value,
        vpt_status=vpt_result.status,
        vpt_median=float(np.median(vpt_result.value)),
        vpt_censored=int(np.sum(vpt_result.status == "censored")),
        horizon=e_h.shape[0],
        positive_control=pc,
    )


def _test_protocol(**overrides) -> dict:
    base = dict(
        h_gate=3, gamma=5.0, tau_1=0.01, tau_vpt=0.5, horizon=3,
        gamma_1=2.0, e_1_pers=1.0, e_recon=0.1, attainable=True,
    )
    base.update(overrides)
    return base


def _test_identity(run_dir, **overrides) -> dict:
    base = dict(
        cell_id=0,
        arm={"backbone": "mlp", "structure": "off", "family": "mlp"},
        seed=0,
        stratum_system="pendulum",
        stratum_regime="ou",
        run_dir=str(run_dir),
    )
    base.update(overrides)
    return base


def _mini_gate_config(tmp_path: Path, *, exp_name: str, seeds: list[int], n_episodes: int = 8, T: int = 28):
    """A real, minimal sld-factorial YAML (DampedPendulum/OURegime, one arm,
    a `representation:` block) written to disk -- T large enough (>=26) to
    satisfy H_GATE=25's own horizon requirement, so the campaign tests below
    exercise the PRODUCTION path (protocol.py's real constants), not a
    shrunk stand-in for it."""
    results_root = tmp_path / "results"
    config = {
        "exp_name": exp_name,
        "datasets": {
            "ou": {
                "system": {"name": "DampedPendulum", "params": {}},
                "regime": {"name": "OURegime", "params": {"theta_ou": 1.0}},
                "obs": None,
                "n_episodes": n_episodes,
                "T": T,
                "dt": 0.02,
                "seed": 0,
            }
        },
        "base": {
            "d_x": 2,
            "d_z": 4,
            "d_u": 1,
            "epochs": 1,
            "batch_episodes": 2,
            "target_params": 500,
            "encoder_hidden": [8],
            "results_root": str(results_root),
        },
        "representation": {
            "epochs": 2,
            "lr": 1e-2,
            "seed": 0,
            "w_align": 1.0,
            "w_var": 1.0,
            "w_cov": 1.0,
            "w_recon": 1.0,
            "view_offset": 1,
            "tau_ema": 0.99,
            "d_z": 4,
            "encoder_hidden": [8],
        },
        "arms": [{"backbone": "mlp", "structure": "off", "overrides": {"mu_j": 0.0}}],
        "seeds": seeds,
    }
    path = tmp_path / f"{exp_name}.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(config, f)
    return path, results_root


# -- write_gate_artifact / read_gate_artifact: round trip, atomicity -------


def test_write_then_read_gate_artifact_round_trips_exactly(tmp_path):
    run_dir = _trained_run_dir(tmp_path, exp_name="gate_roundtrip")
    split = _toy_lti_split(_A_TRUE, _B_TRUE, n_episodes=4, T=12, seed=40)
    predictor = _LinearOracle(_A_TRUE, _B_TRUE)
    kw = _kwargs(h_gate=5, gamma=5.0, tau_1=1.0, horizon=8, tau=0.5)
    result = evaluate_free_rollout(
        _IdentityEncoder(), predictor, split, W=_identity_readout(2), sigma_train=_SIGMA, **kw
    )
    protocol = _test_protocol(h_gate=kw["h_gate"], gamma=kw["gamma"], tau_1=kw["tau_1"], tau_vpt=kw["tau"], horizon=kw["horizon"])
    write_gate_artifact(run_dir, result, identity=_test_identity(run_dir), protocol=protocol, split="test")
    payload = read_gate_artifact(run_dir)

    print(f"[gate] round trip: status={payload['status']} verdict={result.positive_control.verdict}")
    np.testing.assert_array_equal(np.asarray(payload["measures"]["e_h"]), result.e_h)
    np.testing.assert_array_equal(np.asarray(payload["measures"]["persistence_e_h"]), result.persistence_e_h)
    np.testing.assert_array_equal(np.asarray(payload["measures"]["linear_e_h"]), result.linear_e_h)
    assert payload["status"] == result.positive_control.verdict
    assert payload["measures"]["m_low"] == result.positive_control.m_low
    assert payload["measures"]["m_high"] == result.positive_control.m_high
    assert payload["measures"]["lower_bound"] == result.positive_control.lower_bound
    assert payload["measures"]["upper_bound"] == result.positive_control.upper_bound
    assert payload["schema"] == "gate/1"
    assert payload["split"] == "test"


def test_gate_json_written_atomically_via_os_replace(tmp_path, monkeypatch):
    # spy at the REAL call site inside sld.experiments.gate -- REGLES 7.1:
    # exercise the production path, never a parallel one that only looks
    # like it (this project has already been bitten by exactly that).
    import sld.experiments.gate as gate_mod

    run_dir = _trained_run_dir(tmp_path, exp_name="gate_atomic")
    result = _positive_control_result([0.05, 0.05, 0.05], [1, 1, 1], [1, 1, 1], h_gate=3, gamma=5.0, tau_1=1.0)
    protocol = _test_protocol(h_gate=3, gamma=5.0, tau_1=1.0, tau_vpt=0.5, horizon=3)

    calls = []
    real_replace = os.replace

    def _spy_replace(src, dst):
        assert Path(src).exists() and Path(src).stat().st_size > 0  # never rename a partial write
        calls.append((Path(src), Path(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(gate_mod.os, "replace", _spy_replace)

    gate_path = write_gate_artifact(run_dir, result, identity=_test_identity(run_dir), protocol=protocol, split="test")
    print(f"[gate] os.replace calls={calls}")
    assert len(calls) == 1
    assert calls[0][1] == gate_path
    assert gate_path.name == "gate.json"
    assert gate_path.exists()
    assert not gate_path.with_name("gate.json.tmp").exists()


# -- negative 1/2: only one bound passing persists does_not_learn ----------


def test_only_lower_bound_passing_persists_does_not_learn(tmp_path):
    run_dir = _trained_run_dir(tmp_path, exp_name="gate_lower_only")
    # baseline e_H (persistence/linear) = 1.0 at h_gate=3, gamma=5 -> passing
    # threshold is e_H < 0.2; e_1 = 0.5 >= tau_1 = 0.01 -> upper bound fails.
    result = _positive_control_result(
        e_h=[0.5, 0.5, 0.1], persistence_e_h=[1, 1, 1], linear_e_h=[1, 1, 1],
        h_gate=3, gamma=5.0, tau_1=0.01,
    )
    assert result.positive_control.lower_bound and not result.positive_control.upper_bound
    protocol = _test_protocol(h_gate=3, gamma=5.0, tau_1=0.01, tau_vpt=0.5, horizon=3)
    write_gate_artifact(run_dir, result, identity=_test_identity(run_dir), protocol=protocol, split="test")
    payload = read_gate_artifact(run_dir)
    print(f"[gate] lower-only: status={payload['status']} measures={payload['measures']['lower_bound']}/{payload['measures']['upper_bound']}")
    assert payload["status"] == "does_not_learn"
    assert payload["measures"]["lower_bound"] is True
    assert payload["measures"]["upper_bound"] is False


def test_only_upper_bound_passing_persists_does_not_learn(tmp_path):
    run_dir = _trained_run_dir(tmp_path, exp_name="gate_upper_only")
    result = _positive_control_result(
        e_h=[0.05, 0.9, 0.9], persistence_e_h=[1, 1, 1], linear_e_h=[1, 1, 1],
        h_gate=3, gamma=5.0, tau_1=1.0,
    )
    assert result.positive_control.upper_bound and not result.positive_control.lower_bound
    protocol = _test_protocol(h_gate=3, gamma=5.0, tau_1=1.0, tau_vpt=0.5, horizon=3)
    write_gate_artifact(run_dir, result, identity=_test_identity(run_dir), protocol=protocol, split="test")
    payload = read_gate_artifact(run_dir)
    print(f"[gate] upper-only: status={payload['status']} measures={payload['measures']['lower_bound']}/{payload['measures']['upper_bound']}")
    assert payload["status"] == "does_not_learn"
    assert payload["measures"]["lower_bound"] is False
    assert payload["measures"]["upper_bound"] is True


# -- negative 3: non-finite value in the chain -> refused, never fabricated -


def test_divergence_writes_null_measures_never_a_fabricated_number(tmp_path):
    run_dir = _trained_run_dir(tmp_path, exp_name="gate_divergence")
    split = _toy_lti_split(_A_TRUE, _B_TRUE, n_episodes=3, T=20, seed=30)
    predictor = _GrowingPredictor(10.0**40)
    kw = _kwargs(horizon=12)
    with pytest.raises(RolloutDivergence) as excinfo:
        evaluate_free_rollout(
            _IdentityEncoder(), predictor, split, W=_identity_readout(2), sigma_train=_SIGMA, **kw
        )
    divergence = excinfo.value

    protocol = _test_protocol(
        h_gate=kw["h_gate"], gamma=kw["gamma"], tau_1=kw["tau_1"], tau_vpt=kw["tau"], horizon=kw["horizon"]
    )
    write_gate_artifact(run_dir, divergence, identity=_test_identity(run_dir), protocol=protocol, split="test")
    payload = read_gate_artifact(run_dir)
    print(f"[gate] divergence: status={payload['status']} first_step={payload['first_step']}")
    assert payload["status"] == "divergence"
    assert payload["first_step"] == divergence.first_step
    assert payload["measures"] is None  # never a fabricated 0 or inf


def test_read_gate_artifact_refuses_non_finite_measure_never_fabricated(tmp_path):
    # a mutant/corruption that smuggles a NaN into a persisted curve must be
    # refused on READ -- write a valid artifact, hand-corrupt the bytes on
    # disk (simulating exactly that), and check the validator bites.
    run_dir = _trained_run_dir(tmp_path, exp_name="gate_nonfinite")
    result = _positive_control_result([0.05, 0.05, 0.05], [1, 1, 1], [1, 1, 1], h_gate=3, gamma=5.0, tau_1=1.0)
    protocol = _test_protocol(h_gate=3, gamma=5.0, tau_1=1.0, tau_vpt=0.5, horizon=3)
    write_gate_artifact(run_dir, result, identity=_test_identity(run_dir), protocol=protocol, split="test")

    gate_path = run_dir / "gate.json"
    payload = json.loads(gate_path.read_text())
    payload["measures"]["e_h"][0] = float("nan")
    gate_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="e_h"):
        read_gate_artifact(run_dir)


# -- write_gate_artifact: closed schema + sentinel refusal ------------------


def test_write_gate_artifact_refuses_incomplete_identity(tmp_path):
    run_dir = _trained_run_dir(tmp_path, exp_name="gate_bad_identity")
    result = _positive_control_result([0.05] * 3, [1] * 3, [1] * 3, h_gate=3, gamma=5.0, tau_1=1.0)
    protocol = _test_protocol(h_gate=3, gamma=5.0, tau_1=1.0, tau_vpt=0.5, horizon=3)
    bad_identity = _test_identity(run_dir)
    del bad_identity["seed"]
    with pytest.raises(ValueError, match="identity"):
        write_gate_artifact(run_dir, result, identity=bad_identity, protocol=protocol, split="test")


def test_write_gate_artifact_refuses_incomplete_protocol(tmp_path):
    run_dir = _trained_run_dir(tmp_path, exp_name="gate_bad_protocol")
    result = _positive_control_result([0.05] * 3, [1] * 3, [1] * 3, h_gate=3, gamma=5.0, tau_1=1.0)
    protocol = _test_protocol(h_gate=3, gamma=5.0, tau_1=1.0, tau_vpt=0.5, horizon=3)
    del protocol["gamma_1"]
    with pytest.raises(ValueError, match="protocol"):
        write_gate_artifact(run_dir, result, identity=_test_identity(run_dir), protocol=protocol, split="test")


def test_write_gate_artifact_refuses_run_without_valid_sentinel(tmp_path):
    incomplete_dir = tmp_path / "not_a_real_run"
    incomplete_dir.mkdir()
    result = _positive_control_result([0.05] * 3, [1] * 3, [1] * 3, h_gate=3, gamma=5.0, tau_1=1.0)
    protocol = _test_protocol(h_gate=3, gamma=5.0, tau_1=1.0, tau_vpt=0.5, horizon=3)
    with pytest.raises(ValueError, match="completion sentinel"):
        write_gate_artifact(
            incomplete_dir, result, identity=_test_identity(incomplete_dir), protocol=protocol, split="test"
        )


# -- negative 7: a threshold changed after the run is written --------------


def test_read_gate_artifact_refuses_when_threshold_changed_since_write(tmp_path):
    run_dir = _trained_run_dir(tmp_path, exp_name="gate_threshold_moved")
    result = _positive_control_result([0.05] * 3, [1] * 3, [1] * 3, h_gate=3, gamma=5.0, tau_1=1.0)
    protocol = _test_protocol(h_gate=3, gamma=5.0, tau_1=1.0, tau_vpt=0.5, horizon=3)
    write_gate_artifact(run_dir, result, identity=_test_identity(run_dir), protocol=protocol, split="test")

    with pytest.raises(ValueError, match="h_gate"):
        read_gate_artifact(run_dir, expected_protocol={**protocol, "h_gate": 999})
    # an unmoved threshold must NOT be refused
    read_gate_artifact(run_dir, expected_protocol=protocol)


def test_read_gate_artifact_refuses_when_digest_changed_since_write(tmp_path):
    run_dir = _trained_run_dir(tmp_path, exp_name="gate_digest_moved")
    result = _positive_control_result([0.05] * 3, [1] * 3, [1] * 3, h_gate=3, gamma=5.0, tau_1=1.0)
    protocol = _test_protocol(h_gate=3, gamma=5.0, tau_1=1.0, tau_vpt=0.5, horizon=3)
    write_gate_artifact(run_dir, result, identity=_test_identity(run_dir), protocol=protocol, split="test")
    payload = read_gate_artifact(run_dir)
    real_config_hash = payload["digests"]["config_hash"]

    with pytest.raises(ValueError, match="config_hash"):
        read_gate_artifact(run_dir, expected_digests={"config_hash": real_config_hash + "dead"})
    # the real digest must NOT be refused
    read_gate_artifact(run_dir, expected_digests={"config_hash": real_config_hash})


# -- evaluate_gate_campaign: full grid, end to end --------------------------


def test_campaign_end_to_end_reloadable_gate_json_matches_recomputed_verdict(tmp_path):
    cfg_path, results_root = _mini_gate_config(tmp_path, exp_name="gate_e2e", seeds=[0])
    rc = factorial_main(["--config", str(cfg_path)])
    assert rc == 0

    results = evaluate_gate_campaign(cfg_path)
    assert len(results) == 1
    assert results[0]["status"] in ("learns", "does_not_learn")
    run_dir = Path(results[0]["run_dir"])
    payload = read_gate_artifact(run_dir)
    assert payload["status"] == results[0]["status"]
    assert payload["protocol"]["h_gate"] == protocol_mod.H_GATE
    assert payload["protocol"]["gamma"] == protocol_mod.GAMMA
    assert payload["protocol"]["tau_vpt"] == protocol_mod.TAU_VPT
    assert payload["protocol"]["horizon"] == protocol_mod.H_GATE

    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    data = _build_dataset_for(config["datasets"]["ou"])
    data_hash = data_fingerprint(data)
    rep_path = _stratum_representation_path(results_root, "gate_e2e", "DampedPendulum", "OURegime", data_hash)
    artifact = load_frozen_representation(rep_path)

    # independent re-derivation: reload the frozen representation + the SAME
    # test split from disk, call evaluate_run_free_rollout again, and check
    # the persisted curves match EXACTLY (atol=rtol=0) -- the numbers in
    # gate.json ARE the computed ones, not a rounded or re-summarized copy.
    recomputed = evaluate_run_free_rollout(
        run_dir, artifact, data["test"],
        horizon=protocol_mod.H_GATE, tau=protocol_mod.TAU_VPT,
        h_gate=protocol_mod.H_GATE, gamma=protocol_mod.GAMMA, tau_1=payload["protocol"]["tau_1"],
    )
    print(f"[gate] campaign e2e: persisted verdict={payload['status']} recomputed verdict={recomputed.positive_control.verdict}")
    np.testing.assert_array_equal(np.asarray(payload["measures"]["e_h"]), recomputed.e_h)
    np.testing.assert_array_equal(np.asarray(payload["measures"]["persistence_e_h"]), recomputed.persistence_e_h)
    np.testing.assert_array_equal(np.asarray(payload["measures"]["linear_e_h"]), recomputed.linear_e_h)
    assert payload["status"] == recomputed.positive_control.verdict

    # gate_index.csv covers the (single-cell) grid too
    index_rows = list(csv.DictReader((results_root / "gate_e2e" / "gate_index.csv").open()))
    assert len(index_rows) == 1
    assert index_rows[0]["status"] == payload["status"]


def test_missing_seed_is_refused_named_and_shows_missing_in_index(tmp_path):
    # negative 4: the campaign declares 2 seeds; only seed 0 is ever trained
    # (--shard 1/2 trains exactly cell index 0) -- seed 1's run_dir never
    # exists at all. evaluate_gate_campaign must still return an entry for
    # BOTH cells (never silently drop the untrained one), status "missing"
    # for the untrained cell, and gate_index.csv must show it explicitly.
    cfg_path, results_root = _mini_gate_config(tmp_path, exp_name="gate_missing_seed", seeds=[0, 1])
    rc = factorial_main(["--config", str(cfg_path), "--shard", "1/2"])
    assert rc == 0

    results = evaluate_gate_campaign(cfg_path)
    print(f"[gate] missing-seed campaign results: {results}")
    assert len(results) == 2
    by_index = {r["index"]: r for r in results}
    assert by_index[0]["status"] in ("learns", "does_not_learn")
    assert by_index[1]["status"] == "missing"

    rows = list(csv.DictReader((results_root / "gate_missing_seed" / "gate_index.csv").open()))
    assert len(rows) == 2
    missing_rows = [r for r in rows if r["seed"] == "1"]
    assert len(missing_rows) == 1
    assert missing_rows[0]["status"] == "missing"


def test_gate_index_covers_full_grid_even_under_shard(tmp_path):
    # negative 4's mirror: BOTH seeds are trained, but the GATE campaign
    # itself only evaluates a shard -- gate_index.csv must still cover both
    # cells (never just the shard's slice), reporting "missing" for the cell
    # this invocation never gated (never confusing "trained but not gated"
    # with a genuinely evaluated verdict).
    cfg_path, results_root = _mini_gate_config(tmp_path, exp_name="gate_shard_index", seeds=[0, 1])
    rc = factorial_main(["--config", str(cfg_path)])
    assert rc == 0

    results = evaluate_gate_campaign(cfg_path, shard=(1, 2))
    assert len(results) == 1
    assert results[0]["index"] == 0

    rows = list(csv.DictReader((results_root / "gate_shard_index" / "gate_index.csv").open()))
    assert len(rows) == 2
    by_seed = {int(r["seed"]): r["status"] for r in rows}
    assert by_seed[0] in ("learns", "does_not_learn")
    assert by_seed[1] == "missing"


def test_campaign_refuses_when_split_is_shorter_than_h_gate(tmp_path):
    # negative 5: H_GATE is fixed at 25 (protocol.py) and never overridable
    # from the campaign config -- a dataset too short for it must be a named
    # refusal, never a silent min(H, T).
    cfg_path, results_root = _mini_gate_config(tmp_path, exp_name="gate_too_short", seeds=[0], T=10)
    rc = factorial_main(["--config", str(cfg_path)])
    assert rc == 0
    with pytest.raises(ValueError, match="fewer than"):
        evaluate_gate_campaign(cfg_path)


def test_campaign_propagates_non_divergence_valueerrors_uncaught(tmp_path, monkeypatch):
    """Negative 6 (representation digest mismatch), wired at the CAMPAIGN
    level.

    A naive version of this test would overwrite the stratum's cached
    representation file on disk with a DIFFERENT artifact (same stratum,
    different pretraining seed) after training, then call
    evaluate_gate_campaign expecting evaluate_run_free_rollout's own digest
    check to fire. It does not: representation_digest is baked into
    config_hash (Trainer.__init__), which is baked into run_dir's own name
    -- so a changed on-disk representation makes the campaign's preflight
    resolve a DIFFERENT (never-trained) run_dir, and the cell is reported
    "missing", not "digest mismatch". The digest check inside
    evaluate_run_free_rollout is therefore UNREACHABLE through the
    campaign's own normal resolution flow; it guards a different calling
    pattern (a caller handing it a mismatched artifact directly, as
    test_factorial.py's own test_representation_digest_mismatch_is_refused
    already exercises).

    What IS this module's own job to prove: that evaluate_gate_campaign
    propagates a ValueError from evaluate_run_free_rollout UNCAUGHT --
    only RolloutDivergence is caught per cell; any other refusal (a digest
    mismatch, a schema error) is a campaign-wide identity/data problem, not
    a per-seed fluke, so it must abort the whole campaign rather than being
    silently recorded as "missing".
    """
    import sld.experiments.gate as gate_mod

    cfg_path, results_root = _mini_gate_config(tmp_path, exp_name="gate_digest_wiring", seeds=[0])
    rc = factorial_main(["--config", str(cfg_path)])
    assert rc == 0

    def _fake_evaluate(*args, **kwargs):
        raise ValueError("representation digest mismatch: fake, for this test")

    monkeypatch.setattr(gate_mod, "evaluate_run_free_rollout", _fake_evaluate)

    with pytest.raises(ValueError, match="digest mismatch"):
        evaluate_gate_campaign(cfg_path)


def test_representation_digest_mismatch_is_refused(tmp_path):
    from sld.data.generate import build_dataset
    from sld.data.regimes import OURegime
    from sld.systems.damped_pendulum import DampedPendulum
    from sld.training.trainer import Trainer

    data = build_dataset(
        system=DampedPendulum(), regime=OURegime(), obs=None, n_episodes=6, T=10, dt=0.02, seed=0
    )
    art_a = _small_representation(tmp_path, data, seed=1, name="a")
    art_b = _small_representation(tmp_path, data, seed=2, name="b")  # different seed -> different digest
    assert art_a.manifest["digest"] != art_b.manifest["digest"]

    trainer = Trainer(_run_config(tmp_path), representation=art_a)
    result = trainer.fit(data)
    run_dir = Path(result["paths"]["ckpt"]).parent

    with pytest.raises(ValueError, match="digest mismatch"):
        evaluate_run_free_rollout(
            run_dir, art_b, data["test"], horizon=5, tau=0.5, h_gate=3, gamma=5.0, tau_1=1.0
        )


def test_vpt_is_reduced_from_the_returned_e_h_table_not_a_second_array():
    """Requirement 7 (one e_h table, one source), made falsifiable.

    A mutant that computes the VPT from a SECOND array -- built alongside the
    returned one, against a shifted target -- left this whole file GREEN,
    because every other test only checks that ``per_episode_e_h`` and ``e_h``
    agree with EACH OTHER, never that the VPT came from the returned table.
    The check that bites is re-deriving the VPT from the RESULT and demanding
    bit-equality, which no separately-built array can satisfy.
    """
    # a model that is WRONG enough to cross tau on some episodes: an
    # all-censored vector would make the equality below vacuous
    split = _toy_lti_split(_A_TRUE, _B_TRUE, n_episodes=6, T=12, seed=11)
    predictor = _LinearOracle(np.array([[0.2, 0.4], [0.3, 0.1]]), _B_TRUE)
    result = evaluate_free_rollout(
        _IdentityEncoder(), predictor, split, W=_identity_readout(2), sigma_train=_SIGMA, **_kwargs()
    )
    re_derived = _vpt_reducer(result.per_episode_e_h, _kwargs()["tau"])
    print(
        f"[gate] vpt re-derived from returned table: value={re_derived.value} "
        f"crossed={int(np.sum(re_derived.status == 'crossed'))}/{re_derived.value.shape[0]}"
    )
    # positive control FIRST: the assertion is only meaningful if some episode crosses
    assert np.isfinite(re_derived.value).any(), "positive control: some episode must cross tau"
    np.testing.assert_array_equal(result.vpt_value, re_derived.value)
    np.testing.assert_array_equal(result.vpt_status, re_derived.status)
    assert result.vpt_median == float(np.median(re_derived.value))
    assert result.vpt_censored == int(np.sum(re_derived.status == "censored"))


def test_too_few_STEPS_is_refused_even_when_actions_are_long_enough():
    """The steps guard has its own test because the actions guard shadows it.

    ``evaluate_free_rollout`` refuses on two counts -- too few physical steps
    and too few actions. A mutation removing ONLY the steps guard left the
    whole file green, because the existing refusal test supplies a split that
    is short in BOTH and so still trips the actions guard. This test hands it
    a split with plenty of actions and too few steps, so only the steps guard
    can catch it, and asserts on that guard's own message.
    """
    split = _toy_lti_split(_A_TRUE, _B_TRUE, n_episodes=3, T=12, seed=13)
    horizon = _kwargs()["horizon"]
    # keep every action, truncate ONLY the state arrays to one step too few
    short = dict(split)
    short["z_true"] = split["z_true"][:, : 1 + 1 + horizon - 1, :]
    short["x"] = split["x"][:, : 1 + 1 + horizon - 1, :]
    assert short["u"].shape[1] >= 1 + horizon, "positive control: actions must NOT be the limiter"
    with pytest.raises(ValueError, match="physical steps per episode"):
        evaluate_free_rollout(
            _IdentityEncoder(), _LinearOracle(_A_TRUE, _B_TRUE), short,
            W=_identity_readout(2), sigma_train=_SIGMA, **_kwargs(),
        )


def test_too_few_ACTIONS_is_refused_even_when_steps_are_long_enough():
    """Mirror of the steps-guard test above: the actions guard has its own
    test because a split short in BOTH counts trips the (first-checked)
    steps guard before the actions guard's own message is ever exercised.
    This test hands it a split with plenty of physical steps and too few
    actions, so only the actions guard can catch it.
    """
    split = _toy_lti_split(_A_TRUE, _B_TRUE, n_episodes=3, T=12, seed=14)
    horizon = _kwargs()["horizon"]
    required_actions = 1 + horizon  # gate.py's _T0 + horizon
    required_steps = 1 + 1 + horizon  # gate.py's _LOOKBACK + 1 + horizon
    # keep every physical step, truncate ONLY actions to one action too few
    short = dict(split)
    short["u"] = split["u"][:, : required_actions - 1, :]
    assert short["z_true"].shape[1] >= required_steps, "positive control: steps must NOT be the limiter"
    with pytest.raises(ValueError, match="actions per episode"):
        evaluate_free_rollout(
            _IdentityEncoder(), _LinearOracle(_A_TRUE, _B_TRUE), short,
            W=_identity_readout(2), sigma_train=_SIGMA, **_kwargs(),
        )
