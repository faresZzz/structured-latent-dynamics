"""Tests for the D3 persistence layer added to ``sld.experiments.oscillator_gold``
(PROTOCOL.md section 19.4's D3 analogue of ``gate.py``'s ``gate.json``/
``evaluate_gate_campaign``): ``write_d3_gold_artifact``/``read_d3_gold_artifact``,
``score_oscillator_campaign``, and section 19.3's non-gating diagnostic
(``measure_frozen_chain_cost``/``write_frozen_chain_cost_artifact``).

``tests/experiments/test_learned_oscillator_gold.py`` already covers the pure
primitives (``discrete_reference``, ``eig_relative_error``,
``action_relative_error``, ``local_jacobians``, ``oscillator_bounds``) this
module's four bounds are built from -- this file never re-derives those, it
tests the WIRING and the PERSISTENCE arithmetic (bound aggregation, schema,
atomicity, drift detection) on top of them, same split ``test_gate.py`` draws
between ``gate.py``'s pure core and its persistence half.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from sld.audit.rollout import evaluate_positive_control
from sld.audit.rollout import vpt as _vpt_reducer
from sld.data.generate import build_dataset
from sld.data.regimes import OURegime
from sld.encoders.pretrain import build_identity_representation, load_frozen_representation, save_frozen_representation
from sld.experiments import protocol as protocol_mod
from sld.experiments.factorial import main as factorial_main
from sld.experiments.gate import FreeRolloutResult, RolloutDivergence
from sld.experiments.oscillator_gold import (
    _probe_states_from_split,
    measure_frozen_chain_cost,
    read_d3_gold_artifact,
    score_oscillator_campaign,
    write_d3_gold_artifact,
    write_frozen_chain_cost_artifact,
)
from sld.systems.coupled_oscillators import CoupledOscillators
from sld.training.trainer import Trainer

DELTA = 0.05


# --------------------------------------------------------------------------
# Fixtures / hand-built objects
# --------------------------------------------------------------------------


def _tiny_d3_dataset(*, n_episodes: int = 6, T: int = 10, seed: int = 0) -> dict:
    return build_dataset(CoupledOscillators(), OURegime(theta_ou=1.0), None, n_episodes, T, DELTA, seed)


def _tiny_d3_trainer_config(tmp_path: Path, *, exp_name: str, seed: int = 0, epochs: int = 2) -> dict:
    return {
        "exp_name": exp_name,
        "seed": seed,
        "family": "mlp_k",
        "d_x": 4,
        "d_z": 4,
        "d_u": 1,
        "delta": DELTA,
        "r": 2,
        "encoder_hidden": [],
        "target_params": 300,
        "epochs": epochs,
        "batch_episodes": 2,
        "lr": 1e-2,
        "h_fr": 3,
        "results_root": str(tmp_path / "results_d3"),
        "stratum_system": "CoupledOscillators",
        "stratum_regime": "OURegime",
    }


def _trained_d3_run_dir(tmp_path: Path, *, exp_name: str, seed: int = 0, T: int = 10) -> Path:
    """A cheap, REAL completed D3 run dir (valid run_complete.json +
    config.yaml on disk), identity representation, structured ``mlp_k``
    family -- for write/read_d3_gold_artifact tests, which attach their OWN
    hand-built FreeRolloutResult/bounds_field rather than calling
    evaluate_run_free_rollout/oscillator_bounds on this run, so it never
    needs to satisfy bound 1's H=100 horizon requirement."""
    data = _tiny_d3_dataset(T=T, seed=seed)
    art = build_identity_representation(data, system="CoupledOscillators", regime="OURegime", d_z=4)
    rep_path = tmp_path / f"{exp_name}_identity.pt"
    save_frozen_representation(art, rep_path)
    art = load_frozen_representation(rep_path)
    trainer = Trainer(_tiny_d3_trainer_config(tmp_path, exp_name=exp_name, seed=seed), representation=art)
    result = trainer.fit(data)
    return Path(result["paths"]["ckpt"]).parent


def _d3_free_rollout_result(e_h, persistence_e_h, linear_e_h, *, h_gate, gamma, tau_1, tau=0.5) -> FreeRolloutResult:
    """A REAL ``FreeRolloutResult``, built from hand-picked curves through
    the REAL ``evaluate_positive_control``/``vpt`` (R2: gate arithmetic is
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


def _bounds_field(eig_max: float, action_max: float, *, n: int = 3) -> dict:
    eig_vals = [float(eig_max)] + [float(eig_max) * 0.1] * (n - 1)
    action_vals = [float(action_max)] + [float(action_max) * 0.1] * (n - 1)
    return {
        "eig_relative_error": max(eig_vals),
        "eig_relative_error_mean": float(np.mean(eig_vals)),
        "eig_relative_error_std": float(np.std(eig_vals)),
        "eig_relative_error_values": eig_vals,
        "action_relative_error": max(action_vals),
        "action_relative_error_mean": float(np.mean(action_vals)),
        "action_relative_error_std": float(np.std(action_vals)),
        "action_relative_error_values": action_vals,
    }


def _test_thresholds(**overrides) -> dict:
    base = dict(
        state_h100=1e-3,
        spectral=1e-3,
        action=1e-3,
        horizon=3,
        delta=DELTA,
        h_gate=3,
        gamma=5.0,
        tau_vpt=0.5,
        tau_1=0.01,
        gamma_1=2.0,
        e_1_pers=1.0,
        e_recon=0.1,
        attainable=True,
    )
    base.update(overrides)
    return base


def _test_identity(run_dir, **overrides) -> dict:
    base = dict(
        cell_id=0,
        arm={"backbone": "mlp", "structure": "on", "family": "mlp_k"},
        seed=0,
        stratum_system="CoupledOscillators",
        stratum_regime="OURegime",
        run_dir=str(run_dir),
    )
    base.update(overrides)
    return base


def _mini_d3_config(tmp_path: Path, *, exp_name: str, seeds: list[int], n_episodes: int = 10, T: int = 105):
    """A real, minimal ``sld-factorial`` YAML matching
    ``learned_oscillator_gold.yaml``'s own schema (CoupledOscillators/
    OURegime, identity representation, one structured arm) -- ``T`` large
    enough (>=101) to satisfy bound 1's H=100 horizon requirement, so the
    campaign tests below exercise the PRODUCTION path, not a shrunk
    stand-in for it."""
    results_root = tmp_path / "results_d3"
    config = {
        "exp_name": exp_name,
        "datasets": {
            "oscillator": {
                "system": {"name": "CoupledOscillators", "params": {}},
                "regime": {"name": "OURegime", "params": {"theta_ou": 1.0}},
                "obs": None,
                "n_episodes": n_episodes,
                "T": T,
                "dt": DELTA,
                "seed": 0,
            }
        },
        "base": {
            "d_x": 4,
            "d_z": 4,
            "d_u": 1,
            "delta": DELTA,
            "r": 2,
            "encoder_hidden": [],
            "target_params": 300,
            "epochs": 1,
            "batch_episodes": 2,
            "h_fr": 3,
            "results_root": str(results_root),
        },
        "representation": {"kind": "identity", "d_z": 4},
        "arms": [{"backbone": "mlp", "structure": "on", "overrides": {"mu_j": 0.0, "lr": 1e-2}}],
        "seeds": seeds,
    }
    path = tmp_path / f"{exp_name}.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(config, f)
    return path, results_root


# --------------------------------------------------------------------------
# _probe_states_from_split
# --------------------------------------------------------------------------


def test_probe_states_drawn_from_the_test_split_at_the_declared_index():
    # positive: hand-built z_true/u with KNOWN values -- the probes returned
    # must be EXACTLY column-index 1 of each, one row per episode, never a
    # second, independently invented sampling scheme.
    z_true = np.arange(2 * 5 * 4, dtype=np.float64).reshape(2, 5, 4)
    u = np.arange(2 * 5 * 1, dtype=np.float64).reshape(2, 5, 1) * -1.0
    split = {"z_true": z_true, "u": u, "x": z_true.copy()}
    probe_states, probe_actions = _probe_states_from_split(split)
    print(f"[oscillator_gold] probes: states={probe_states.tolist()} actions={probe_actions.tolist()}")
    np.testing.assert_array_equal(probe_states.numpy(), z_true[:, 1, :])
    np.testing.assert_array_equal(probe_actions.numpy(), u[:, 1, :])


def test_probe_states_refused_when_too_few_STEPS_even_when_actions_are_long_enough():
    z_true = np.zeros((2, 1, 4), dtype=np.float64)  # only index 0 -- no index 1
    u = np.zeros((2, 5, 1), dtype=np.float64)  # plenty of actions
    with pytest.raises(ValueError, match="physical steps per episode"):
        _probe_states_from_split({"z_true": z_true, "u": u, "x": z_true.copy()})


def test_probe_states_refused_when_too_few_ACTIONS_even_when_steps_are_long_enough():
    z_true = np.zeros((2, 5, 4), dtype=np.float64)  # plenty of steps
    u = np.zeros((2, 1, 1), dtype=np.float64)  # only index 0
    with pytest.raises(ValueError, match="actions per episode"):
        _probe_states_from_split({"z_true": z_true, "u": u, "x": z_true.copy()})


def test_probe_states_refused_on_empty_split():
    z_true = np.zeros((0, 5, 4), dtype=np.float64)
    u = np.zeros((0, 5, 1), dtype=np.float64)
    with pytest.raises(ValueError, match="no episodes"):
        _probe_states_from_split({"z_true": z_true, "u": u, "x": z_true.copy()})


# --------------------------------------------------------------------------
# write_d3_gold_artifact / read_d3_gold_artifact
# --------------------------------------------------------------------------


def test_all_four_bounds_passing_round_trips_as_status_passes(tmp_path):
    run_dir = _trained_d3_run_dir(tmp_path, exp_name="d3_all_pass")
    result = _d3_free_rollout_result(
        [1e-4, 1e-4, 1e-4], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], h_gate=3, gamma=5.0, tau_1=0.01
    )
    bounds_field = _bounds_field(eig_max=1e-4, action_max=1e-4)
    thresholds = _test_thresholds()
    path = write_d3_gold_artifact(
        run_dir, result, identity=_test_identity(run_dir), thresholds=thresholds, split="test",
        bounds_field=bounds_field,
    )
    payload = read_d3_gold_artifact(run_dir)
    print(f"[d3_gold] all-pass round trip: status={payload['status']} bounds={payload['bounds']}")
    assert payload["status"] == "passes"
    assert path.name == "d3_gold.json"
    assert payload["schema"] == "d3_gold/1"
    assert payload["split"] == "test"
    for name in ("state_h100", "spectral", "action", "gain"):
        assert payload["bounds"][name]["passes"] is True
    assert payload["dispersion"]["n_probes"] == 3
    assert payload["dispersion"]["spectral_values"] == bounds_field["eig_relative_error_values"]
    assert payload["digests"]["config_hash"] == run_dir.name.split("-")[0]


def test_exactly_one_bound_passing_is_fails_never_passes(tmp_path):
    """REGLES section 7.1's own negative, made concrete: state_h100 alone
    passes (e_h at the shared h_gate=horizon index is tiny), spectral and
    action both fail, and the GAIN bound also fails despite e_h being tiny --
    engineered by giving the trivial baselines an even SMALLER error at that
    index, so ``e_h < baseline / gamma`` is false even though ``e_h`` itself
    clears the 1e-3 state bound. Exactly one of four bounds passes; status
    must be "fails", never "passes"."""
    run_dir = _trained_d3_run_dir(tmp_path, exp_name="d3_one_pass")
    result = _d3_free_rollout_result(
        [0.01, 0.01, 0.0005], [0.01, 0.01, 0.002], [0.01, 0.01, 0.003],
        h_gate=3, gamma=5.0, tau_1=1.0,
    )
    assert result.positive_control.lower_bound is False, "test setup: gain bound must fail"
    assert float(result.e_h[2]) < 1e-3, "test setup: state_h100 bound must pass"
    bounds_field = _bounds_field(eig_max=5e-3, action_max=5e-3)
    thresholds = _test_thresholds(tau_1=1.0)
    write_d3_gold_artifact(
        run_dir, result, identity=_test_identity(run_dir), thresholds=thresholds, split="test",
        bounds_field=bounds_field,
    )
    payload = read_d3_gold_artifact(run_dir)
    passing = [name for name, b in payload["bounds"].items() if b["passes"]]
    print(f"[d3_gold] exactly-one-passing: passing={passing} status={payload['status']}")
    assert passing == ["state_h100"]
    assert payload["status"] == "fails"


def test_one_bound_failing_among_four_is_fails(tmp_path):
    run_dir = _trained_d3_run_dir(tmp_path, exp_name="d3_almost_pass")
    result = _d3_free_rollout_result(
        [1e-4, 1e-4, 1e-4], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], h_gate=3, gamma=5.0, tau_1=0.01
    )
    # action bound alone fails (5e-3 > 1e-3 threshold); spectral/state/gain pass
    bounds_field = _bounds_field(eig_max=1e-4, action_max=5e-3)
    write_d3_gold_artifact(
        run_dir, result, identity=_test_identity(run_dir), thresholds=_test_thresholds(), split="test",
        bounds_field=bounds_field,
    )
    payload = read_d3_gold_artifact(run_dir)
    assert payload["bounds"]["action"]["passes"] is False
    assert payload["status"] == "fails"


def test_divergence_writes_null_bounds_never_a_fabricated_number(tmp_path):
    run_dir = _trained_d3_run_dir(tmp_path, exp_name="d3_divergence")
    exc = RolloutDivergence(first_step=7, message="synthetic divergence for this test")
    write_d3_gold_artifact(
        run_dir, exc, identity=_test_identity(run_dir), thresholds=_test_thresholds(), split="test"
    )
    payload = read_d3_gold_artifact(run_dir)
    print(f"[d3_gold] divergence: status={payload['status']} first_step={payload['first_step']}")
    assert payload["status"] == "divergence"
    assert payload["bounds"] is None
    assert payload["dispersion"] is None
    assert payload["first_step"] == 7


def test_d3_gold_written_atomically_via_os_replace(tmp_path, monkeypatch):
    # spy at the REAL call site inside sld.experiments.oscillator_gold --
    # REGLES 7.1: exercise the production path, never a parallel one.
    import sld.experiments.oscillator_gold as og_mod

    run_dir = _trained_d3_run_dir(tmp_path, exp_name="d3_atomic")
    result = _d3_free_rollout_result(
        [1e-4, 1e-4, 1e-4], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], h_gate=3, gamma=5.0, tau_1=0.01
    )
    bounds_field = _bounds_field(eig_max=1e-4, action_max=1e-4)

    calls = []
    real_replace = os.replace

    def _spy_replace(src, dst):
        assert Path(src).exists() and Path(src).stat().st_size > 0  # never rename a partial write
        calls.append((Path(src), Path(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(og_mod.os, "replace", _spy_replace)
    path = write_d3_gold_artifact(
        run_dir, result, identity=_test_identity(run_dir), thresholds=_test_thresholds(), split="test",
        bounds_field=bounds_field,
    )
    print(f"[d3_gold] os.replace calls={calls}")
    assert len(calls) == 1
    assert calls[0][1] == path
    assert path.exists()


def test_write_d3_gold_artifact_refuses_incomplete_identity(tmp_path):
    run_dir = _trained_d3_run_dir(tmp_path, exp_name="d3_bad_identity")
    result = _d3_free_rollout_result(
        [1e-4, 1e-4, 1e-4], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], h_gate=3, gamma=5.0, tau_1=0.01
    )
    identity = _test_identity(run_dir)
    del identity["seed"]
    with pytest.raises(ValueError, match="identity must carry exactly"):
        write_d3_gold_artifact(
            run_dir, result, identity=identity, thresholds=_test_thresholds(), split="test",
            bounds_field=_bounds_field(1e-4, 1e-4),
        )


def test_write_d3_gold_artifact_refuses_incomplete_thresholds(tmp_path):
    run_dir = _trained_d3_run_dir(tmp_path, exp_name="d3_bad_thresholds")
    result = _d3_free_rollout_result(
        [1e-4, 1e-4, 1e-4], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], h_gate=3, gamma=5.0, tau_1=0.01
    )
    thresholds = _test_thresholds()
    del thresholds["gamma_1"]
    with pytest.raises(ValueError, match="thresholds must carry exactly"):
        write_d3_gold_artifact(
            run_dir, result, identity=_test_identity(run_dir), thresholds=thresholds, split="test",
            bounds_field=_bounds_field(1e-4, 1e-4),
        )


def test_write_d3_gold_artifact_refuses_run_without_valid_sentinel(tmp_path):
    run_dir = tmp_path / "never_trained"
    run_dir.mkdir()
    result = _d3_free_rollout_result(
        [1e-4, 1e-4, 1e-4], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], h_gate=3, gamma=5.0, tau_1=0.01
    )
    with pytest.raises(ValueError, match="no valid completion sentinel"):
        write_d3_gold_artifact(
            run_dir, result, identity=_test_identity(run_dir), thresholds=_test_thresholds(), split="test",
            bounds_field=_bounds_field(1e-4, 1e-4),
        )


def test_write_d3_gold_artifact_refuses_missing_bounds_field_on_free_rollout_result(tmp_path):
    run_dir = _trained_d3_run_dir(tmp_path, exp_name="d3_no_bounds_field")
    result = _d3_free_rollout_result(
        [1e-4, 1e-4, 1e-4], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], h_gate=3, gamma=5.0, tau_1=0.01
    )
    with pytest.raises(ValueError, match="bounds_field"):
        write_d3_gold_artifact(
            run_dir, result, identity=_test_identity(run_dir), thresholds=_test_thresholds(), split="test",
        )


def test_write_d3_gold_artifact_refuses_wrong_result_type(tmp_path):
    run_dir = _trained_d3_run_dir(tmp_path, exp_name="d3_wrong_type")
    with pytest.raises(ValueError, match="must be a FreeRolloutResult"):
        write_d3_gold_artifact(
            run_dir, {"not": "a result"}, identity=_test_identity(run_dir), thresholds=_test_thresholds(),
            split="test",
        )


def test_read_d3_gold_artifact_refuses_when_threshold_changed_since_write(tmp_path):
    run_dir = _trained_d3_run_dir(tmp_path, exp_name="d3_threshold_drift")
    result = _d3_free_rollout_result(
        [1e-4, 1e-4, 1e-4], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], h_gate=3, gamma=5.0, tau_1=0.01
    )
    write_d3_gold_artifact(
        run_dir, result, identity=_test_identity(run_dir), thresholds=_test_thresholds(), split="test",
        bounds_field=_bounds_field(1e-4, 1e-4),
    )
    with pytest.raises(ValueError, match="threshold 'spectral' changed"):
        read_d3_gold_artifact(run_dir, expected_thresholds={"spectral": 5e-4})
    # the real, unchanged threshold must NOT be refused
    read_d3_gold_artifact(run_dir, expected_thresholds={"spectral": 1e-3})


def test_read_d3_gold_artifact_refuses_when_digest_changed_since_write(tmp_path):
    run_dir = _trained_d3_run_dir(tmp_path, exp_name="d3_digest_drift")
    result = _d3_free_rollout_result(
        [1e-4, 1e-4, 1e-4], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], h_gate=3, gamma=5.0, tau_1=0.01
    )
    write_d3_gold_artifact(
        run_dir, result, identity=_test_identity(run_dir), thresholds=_test_thresholds(), split="test",
        bounds_field=_bounds_field(1e-4, 1e-4),
    )
    real_config_hash = run_dir.name.split("-")[0]
    with pytest.raises(ValueError, match="digest 'config_hash' changed"):
        read_d3_gold_artifact(run_dir, expected_digests={"config_hash": real_config_hash + "dead"})
    read_d3_gold_artifact(run_dir, expected_digests={"config_hash": real_config_hash})


def test_read_d3_gold_artifact_refuses_missing_file(tmp_path):
    run_dir = tmp_path / "no_such_run"
    run_dir.mkdir()
    with pytest.raises(ValueError, match="no d3_gold.json"):
        read_d3_gold_artifact(run_dir)


# --------------------------------------------------------------------------
# score_oscillator_campaign: full grid, end to end
# --------------------------------------------------------------------------


def test_campaign_end_to_end_reloadable_d3_gold_matches_recomputed_bounds(tmp_path):
    from sld.experiments.factorial import _build_dataset_for, _stratum_representation_path
    from sld.experiments.gate import evaluate_run_free_rollout, load_run_double
    from sld.experiments.oscillator_gold import oscillator_bounds
    from sld.encoders.pretrain import load_frozen_representation
    from sld.training.trainer import data_fingerprint

    cfg_path, results_root = _mini_d3_config(tmp_path, exp_name="d3_e2e", seeds=[0])
    rc = factorial_main(["--config", str(cfg_path)])
    assert rc == 0

    results = score_oscillator_campaign(cfg_path)
    assert len(results) == 1
    assert results[0]["status"] in ("passes", "fails")
    run_dir = Path(results[0]["run_dir"])
    payload = read_d3_gold_artifact(run_dir)
    assert payload["status"] == results[0]["status"]
    assert payload["thresholds"]["horizon"] == 100
    assert payload["thresholds"]["state_h100"] == 1e-3
    assert payload["thresholds"]["spectral"] == 1e-3
    assert payload["thresholds"]["action"] == 1e-3
    assert payload["dispersion"]["n_probes"] >= 1

    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    data = _build_dataset_for(config["datasets"]["oscillator"])
    data_hash = data_fingerprint(data)
    rep_path = _stratum_representation_path(results_root, "d3_e2e", "CoupledOscillators", "OURegime", data_hash)
    artifact = load_frozen_representation(rep_path)

    recomputed = evaluate_run_free_rollout(
        run_dir, artifact, data["test"],
        horizon=100, tau=protocol_mod.TAU_VPT, h_gate=protocol_mod.H_GATE, gamma=protocol_mod.GAMMA,
        tau_1=payload["thresholds"]["tau_1"],
    )
    print(
        f"[d3_gold] campaign e2e: persisted status={payload['status']} "
        f"recomputed e_h[99]={recomputed.e_h[99]:.3e}"
    )
    np.testing.assert_array_equal(np.asarray(recomputed.e_h[99]), payload["bounds"]["state_h100"]["value"])

    _encoder64, predictor64, run_cfg = load_run_double(run_dir, artifact)
    probe_states, probe_actions = _probe_states_from_split(data["test"])
    recomputed_bounds = oscillator_bounds(
        predictor64, CoupledOscillators(), delta=run_cfg["delta"],
        probe_states=probe_states, probe_actions=probe_actions,
    )
    assert recomputed_bounds["eig_relative_error"] == payload["bounds"]["spectral"]["value"]
    assert recomputed_bounds["action_relative_error"] == payload["bounds"]["action"]["value"]

    index_rows = list(csv.DictReader((results_root / "d3_e2e" / "d3_gold_index.csv").open()))
    assert len(index_rows) == 1
    assert index_rows[0]["status"] == payload["status"]


def test_missing_seed_is_refused_named_and_shows_missing_in_index(tmp_path):
    cfg_path, results_root = _mini_d3_config(tmp_path, exp_name="d3_missing_seed", seeds=[0, 1])
    rc = factorial_main(["--config", str(cfg_path), "--shard", "1/2"])
    assert rc == 0

    results = score_oscillator_campaign(cfg_path)
    print(f"[d3_gold] missing-seed campaign results: {results}")
    assert len(results) == 2
    by_index = {r["index"]: r for r in results}
    assert by_index[0]["status"] in ("passes", "fails")
    assert by_index[1]["status"] == "missing"
    assert "d3_gold_path" not in by_index[1]

    rows = list(csv.DictReader((results_root / "d3_missing_seed" / "d3_gold_index.csv").open()))
    assert len(rows) == 2
    missing_rows = [r for r in rows if r["seed"] == "1"]
    assert len(missing_rows) == 1
    assert missing_rows[0]["status"] == "missing"
    # the never-trained cell never got a d3_gold.json written for it
    missing_run_dir = Path(missing_rows[0]["run_dir"])
    assert not (results_root / missing_run_dir / "d3_gold.json").exists()


def test_d3_gold_index_covers_full_grid_even_under_shard(tmp_path):
    cfg_path, results_root = _mini_d3_config(tmp_path, exp_name="d3_shard_index", seeds=[0, 1])
    rc = factorial_main(["--config", str(cfg_path)])
    assert rc == 0

    results = score_oscillator_campaign(cfg_path, shard=(1, 2))
    assert len(results) == 1
    assert results[0]["index"] == 0

    rows = list(csv.DictReader((results_root / "d3_shard_index" / "d3_gold_index.csv").open()))
    assert len(rows) == 2
    by_seed = {int(r["seed"]): r["status"] for r in rows}
    assert by_seed[0] in ("passes", "fails")
    assert by_seed[1] == "missing"


def test_campaign_refuses_when_split_is_shorter_than_bound1_horizon(tmp_path):
    # bound 1 is fixed at H=100 and never overridable from the campaign
    # config -- a dataset too short for it must be a named refusal, never a
    # silent min(H, T).
    cfg_path, results_root = _mini_d3_config(tmp_path, exp_name="d3_too_short", seeds=[0], T=10)
    rc = factorial_main(["--config", str(cfg_path)])
    assert rc == 0
    with pytest.raises(ValueError, match="fewer than"):
        score_oscillator_campaign(cfg_path)


def test_campaign_propagates_non_divergence_valueerrors_uncaught(tmp_path, monkeypatch):
    """Mirrors gate.py's own
    test_campaign_propagates_non_divergence_valueerrors_uncaught: only
    RolloutDivergence is caught per cell; any other refusal from
    evaluate_run_free_rollout is a campaign-wide identity/data problem and
    must abort the whole campaign rather than being recorded as "missing"."""
    import sld.experiments.oscillator_gold as og_mod

    cfg_path, results_root = _mini_d3_config(tmp_path, exp_name="d3_digest_wiring", seeds=[0], T=10)
    rc = factorial_main(["--config", str(cfg_path)])
    assert rc == 0

    def _fake_evaluate(*args, **kwargs):
        raise ValueError("representation digest mismatch: fake, for this test")

    monkeypatch.setattr(og_mod, "evaluate_run_free_rollout", _fake_evaluate)

    with pytest.raises(ValueError, match="digest mismatch"):
        score_oscillator_campaign(cfg_path)


def test_campaign_requires_a_representation_block(tmp_path):
    cfg_path, results_root = _mini_d3_config(tmp_path, exp_name="d3_no_rep", seeds=[0], T=10)
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    del config["representation"]
    with open(cfg_path, "w") as f:
        yaml.safe_dump(config, f)
    with pytest.raises(ValueError, match="requires a `representation:` block"):
        score_oscillator_campaign(cfg_path)


# --------------------------------------------------------------------------
# Section 19.3's non-gating diagnostic
# --------------------------------------------------------------------------


def test_measure_frozen_chain_cost_is_finite_and_non_gating():
    result = measure_frozen_chain_cost(
        n_episodes=8, T=12, encoder_hidden=(8,), epochs=2, lr=1e-2,
    )
    print(f"[d3_diagnostic] identity={result['identity']} pretrained={result['pretrained']}")
    assert result["non_gating"] is True
    assert result["schema"] == "d3_frozen_chain_cost/1"
    assert result["system"] == "CoupledOscillators"
    for key in ("identity", "pretrained"):
        assert math.isfinite(result[key]["e_recon"])
        assert math.isfinite(result[key]["e_1_pers"])
        assert math.isfinite(result[key]["e_recon_over_threshold"])
    # positive control: the identity representation's e_recon sits at the
    # float32 noise floor (section 19.1) -- far below the 1e-3 bound, at any
    # data size, since it is an exact linear identity fit, never a trained
    # approximation.
    assert result["identity"]["e_recon"] < 1e-3


def test_write_frozen_chain_cost_artifact_atomic_and_round_trips(tmp_path, monkeypatch):
    import sld.experiments.oscillator_gold as og_mod

    result = measure_frozen_chain_cost(n_episodes=8, T=12, encoder_hidden=(8,), epochs=2, lr=1e-2)
    path = tmp_path / "diagnostic" / "frozen_chain_cost.json"

    calls = []
    real_replace = os.replace

    def _spy_replace(src, dst):
        assert Path(src).exists() and Path(src).stat().st_size > 0
        calls.append((Path(src), Path(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(og_mod.os, "replace", _spy_replace)
    written = write_frozen_chain_cost_artifact(result, path)
    assert len(calls) == 1
    assert written == path
    reloaded = json.loads(path.read_text())
    assert reloaded == result
