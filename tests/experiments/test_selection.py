"""Tests for sld.experiments.selection: PROTOCOL.md sections 2 (LR sweep)
and 5 (mu_j calibration), the four-phase orchestration, and the report
writer.

Three tiers, per the task brief:
1. Pure decision logic (``_select_lr``, ``_select_mu_j``,
   ``_refinement_candidate``, and the phase-4 bounded iteration) -- no
   training, canned/mocked inputs, exact expected branches.
2. Empirical measurement functions (``_measure_fr_val_loss``, the raw
   ``_measure_sigma_max_median``, and PROTOCOL section 10.3's declared
   ``estimate_gauge`` + ``measure_sigma_max_median_normalized``) on a tiny
   real trained checkpoint -- finite, sane-range floats, not gold values
   (these are empirical measurements, not closed-form quantities).
3. One minimal end-to-end integration test (``run_family_selection`` +
   ``run_all_families``) on a single family, tiny epochs, tiny dataset.

Small/fast by design: n_episodes<=10, T<=6, epochs<=2, target_params=10_000
(transformer's budget search needs this repo-standard value to converge
within +/-5% at these tiny dims) -- otherwise mirrors
tests/training/test_trainer.py's declared scale.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sld.experiments import selection
from sld.objectives.jac_reg import Gauge, gauge_digest
from sld.predictors.base import PredictorState

_D_X, _D_Z, _D_U = 2, 4, 1

# A stand-in for the frozen phase-OFF gauge in the mocked decision-logic
# tests: identity-checked by ``is``, never applied, so its contents are
# irrelevant -- what those tests assert is that ONE object reaches every
# measurement, not what it scales.
_GAUGE = Gauge(s_h=torch.ones(1, _D_Z, dtype=torch.float64))


def _tiny_calibration_config(families=("mlp",)) -> dict:
    arms = []
    for family in families:
        off_overrides = {"mu_j": 0.0, "lr": None}
        on_overrides = {"mu_j": None, "lr": None}
        if family == "koopman_ssm":
            off_overrides.update({"constrained": False, "delta": 0.02})
            on_overrides.update({"constrained": True, "delta": 0.02})
        arms.append({"family": family, "constraint": "off", "overrides": off_overrides})
        arms.append({"family": family, "constraint": "on", "overrides": on_overrides})
    return {
        "exp_name": "pendulum_core_selection",
        "datasets": {
            "ou": {
                "system": {"name": "DampedPendulum", "params": {}},
                "regime": {"name": "OURegime", "params": {"theta_ou": 1.0}},
                "obs": None,
                "n_episodes": 10,
                "T": 6,
                "dt": 0.02,
                "seed": 0,
            },
            "action_conditioned": {
                "system": {"name": "DampedPendulum", "params": {}},
                "regime": {
                    "name": "ActionConditionedRegime",
                    "params": {"K_gain": [[0.8, 0.4]], "sigma": 0.3},
                },
                "obs": None,
                "n_episodes": 10,
                "T": 6,
                "dt": 0.02,
                "seed": 0,
            },
        },
        "base": {
            "d_x": _D_X, "d_z": _D_Z, "d_u": _D_U,
            "target_params": 10_000, "encoder_hidden": [8],
        },
        "arms": arms,
        "seeds": [0],
    }


def _train_tiny(
    tmp_path: Path, family: str, epochs: int = 2, *, n_episodes: int | None = None
) -> tuple[dict, dict]:
    """Trains ``family`` for ``epochs`` on a tiny OU dataset and returns
    ``(load_run(...) dict, data["val"])``.

    ``n_episodes`` overrides the config's default (10) when given -- the
    default 0.8/0.1/0.1 split leaves only 1 val episode at n_episodes=10,
    which is enough to pool std over (episode, timestep) for h/fixed-size
    m, but NOT enough to estimate a growing-memory gauge's cross-episode,
    per-position std (needs >= 2 val episodes for a finite Bessel-corrected
    std).
    """
    cfg = _tiny_calibration_config((family,))
    if n_episodes is not None:
        for ds_cfg in cfg["datasets"].values():
            ds_cfg["n_episodes"] = n_episodes
    dataset = selection._build_dataset_for(cfg["datasets"]["ou"])
    config = selection._build_config(
        cfg, family, "off", lr=1e-2, mu_j=0.0, epochs=epochs,
    )
    run, _run_dir = selection._fit_and_load(config, dataset, tmp_path / "results")
    return run, dataset["val"]


# --------------------------------------------------------------------------
# tier 1: pure decision logic
# --------------------------------------------------------------------------


def test_select_lr_tie_break_smallest():
    scores = {1e-4: 0.5, 3e-4: 0.2, 1e-3: 0.2}
    assert selection._select_lr(scores) == 3e-4


def test_select_lr_unique_minimum():
    scores = {1e-4: 0.9, 3e-4: 0.1, 1e-3: 0.5}
    assert selection._select_lr(scores) == 3e-4


def test_select_mu_j_smallest_in_band():
    medians = {1e-3: 1.5, 1e-2: 0.95, 1e-1: 0.92, 1.0: 0.5}
    mu_j, verdict = selection._select_mu_j(medians)
    assert (mu_j, verdict) == (1e-2, "selected")


def test_select_mu_j_none_in_band():
    medians = {1e-3: 1.5, 1e-2: 1.2, 1e-1: 1.1, 1.0: 1.05}
    mu_j, verdict = selection._select_mu_j(medians)
    assert (mu_j, verdict) == (None, "none")


def test_refinement_all_above_band():
    medians = {1e-3: 1.5, 1e-2: 1.4, 1e-1: 1.3, 1.0: 1.05}
    assert selection._refinement_candidate(medians) == pytest.approx(10.0)


def test_refinement_all_below_band():
    medians = {1e-3: 0.8, 1e-2: 0.5, 1e-1: 0.3, 1.0: 0.1}
    assert selection._refinement_candidate(medians) == pytest.approx(1e-4)


def test_refinement_monotone_straddle_geometric_mean():
    # decreasing monotonically, band stepped over between 1e-2 (1.05) and 1e-1 (0.85)
    medians = {1e-3: 1.3, 1e-2: 1.05, 1e-1: 0.85, 1.0: 0.5}
    expected = (1e-2 * 1e-1) ** 0.5
    assert selection._refinement_candidate(medians) == pytest.approx(expected)


def test_refinement_non_monotone_abstains():
    medians = {1e-3: 1.2, 1e-2: 0.8, 1e-1: 1.1, 1.0: 0.7}
    assert selection._refinement_candidate(medians) is None


def test_refinement_monotone_plateau_still_counts_as_monotone():
    # Bug 3 (Wave 5 round 7 review): PROTOCOL.md section 5 step 6 says "the
    # medians decrease monotonically as mu_j increases" -- read verbatim,
    # that is NON-increasing, equality admitted. An exact plateau between
    # two ADJACENT grid points (a real occurrence in 10-epoch proxy runs,
    # see the next test) must not by itself defeat the monotone-straddle
    # branch when the actual bracketing pair lies elsewhere in the grid.
    # Old (strict ``>``) code: monotone_decreasing check fails at the
    # plateau (0.85 > 0.85 is False) -> incorrectly returns None
    # (non-monotone abstention) even though mu_j=1e-2/1e-1 already bracket
    # the band.
    medians = {1e-3: 1.3, 1e-2: 1.05, 1e-1: 0.85, 1.0: 0.85}  # plateau: last two equal
    expected = (1e-2 * 1e-1) ** 0.5
    assert selection._refinement_candidate(medians) == pytest.approx(expected)


def test_refinement_min_ssm_reported_plateau_is_all_above_band_both_ways():
    # The real min_ssm medians named in the round-7 review report --
    # 1.0443, 1.0440, 1.0439, 1.0439, an exact plateau on the last two grid
    # points -- are ALL above BAND[1]=1.00, so this exact input hits the
    # "all four above the band" branch BEFORE the strict-vs-monotone check
    # is ever reached: verified here explicitly, not assumed. The
    # strict/non-strict fix does not change this particular family's
    # outcome; it fixes the (distinct, and real) case exercised by
    # test_refinement_monotone_plateau_still_counts_as_monotone above,
    # where a plateau sits OUTSIDE the bracketing pair in a genuinely
    # straddling grid.
    medians = {1e-3: 1.0443, 1e-2: 1.0440, 1e-1: 1.0439, 1.0: 1.0439}
    assert selection._refinement_candidate(medians) == pytest.approx(10.0)


def test_calibrate_mu_j_uses_refinement_and_reports_band_not_reached(monkeypatch):
    # mock _measure_mu_j_candidate so the calibration loop's decision logic
    # runs without training: first the 4-grid sweep all above band, then
    # the one allowed refinement point also above -> "band not reached"
    calls = []

    def fake_measure(family, lr, mu, epochs, calibration_config, datasets, results_root, gauge, seed):
        calls.append(mu)
        medians_by_mu = {1e-3: 1.5, 1e-2: 1.4, 1e-1: 1.3, 1.0: 1.2, 10.0: 1.1}
        row = {"family": family, "phase": "mu_j_calibration", "mu_j": mu}
        return medians_by_mu[mu], row

    monkeypatch.setattr(selection, "_measure_mu_j_candidate", fake_measure)
    result = selection._calibrate_mu_j("mlp", 1e-3, {}, {}, Path("."), _GAUGE)
    assert result["verdict"] == "band not reached"
    assert result["retained_mu_j"] is None
    # exactly one refinement point (mu_j=10) measured, no further grid growth
    assert calls == [1e-3, 1e-2, 1e-1, 1.0, 10.0]


def test_calibrate_mu_j_refinement_lands_in_band(monkeypatch):
    def fake_measure(family, lr, mu, epochs, calibration_config, datasets, results_root, gauge, seed):
        medians_by_mu = {1e-3: 1.3, 1e-2: 1.05, 1e-1: 0.85, 1.0: 0.5}
        geo_mean = (1e-2 * 1e-1) ** 0.5
        medians_by_mu[geo_mean] = 0.95
        row = {"family": family, "phase": "mu_j_calibration", "mu_j": mu}
        return medians_by_mu[mu], row

    monkeypatch.setattr(selection, "_measure_mu_j_candidate", fake_measure)
    result = selection._calibrate_mu_j("mlp", 1e-3, {}, {}, Path("."), _GAUGE)
    assert result["verdict"] == "selected"
    assert result["retained_mu_j"] == pytest.approx((1e-2 * 1e-1) ** 0.5)


# --------------------------------------------------------------------------
# phase-4 bounded iteration, fully mocked (no training)
# --------------------------------------------------------------------------


def _patch_selection_pipeline(monkeypatch, *, off_lr, on_lr_sequence, calib_sequence, proxy_sequence):
    """Wires ``run_family_selection``'s four training-driving calls to
    scripted return values, so the bounded-iteration control flow (section
    2 phase 4) can be tested without any ``Trainer.fit`` call.

    ``on_lr_sequence``: LRs returned by successive ``_sweep_lr`` calls for
    the ON cell (index 0 = phase ON's initial sweep, index 1+ = each
    cycle's re-sweep).
    ``calib_sequence``: dicts returned by successive ``_calibrate_mu_j``
    calls (index 0 = initial calibration, index 1+ = each cycle's
    recalibration).
    ``proxy_sequence``: medians returned by successive ``_measure_proxy_at``
    calls.

    Returns a ``trace`` dict recording gauge plumbing so the callers can
    check it without a second harness: ``trace["gauge_estimations"]`` is the
    list of ``(family, encoder_tag)`` pairs ``estimate_gauge`` was called
    with (it must have length 1 per family, on the OFF checkpoint),
    ``trace["gauges_seen"]`` is every gauge object handed to a downstream
    measurement, and ``trace["sweep_gauges"]`` is every ``(constraint,
    gauge)`` pair handed to an LR sweep -- the TRAINING side of the same
    plumbing.
    """
    monkeypatch.setattr(selection, "_build_dataset_for", lambda ds_cfg: {"val": {}})

    on_lrs = iter(on_lr_sequence)
    calibs = iter(calib_sequence)
    proxies = iter(proxy_sequence)
    trace = {"gauge_estimations": [], "gauges_seen": [], "sweep_gauges": [], "order": []}

    def fake_sweep_lr(
        family, constraint, mu_j, epochs, calibration_config, datasets, results_root,
        gauge=None, seed=0,
    ):
        lr = off_lr if constraint == "off" else next(on_lrs)
        trace["order"].append(f"sweep_{constraint}")
        # the ON sweeps TRAIN with a live penalty, so they must receive the
        # frozen gauge; the OFF sweep is what produces the checkpoint it is
        # estimated from and cannot have one yet
        trace["sweep_gauges"].append((constraint, gauge))
        return {
            "retained_lr": lr, "scores": {}, "runs": [],
            # tagged so a test can prove the gauge was estimated on the OFF
            # cell's checkpoint and not on some ON (penalized) run's
            "retained_ou_run": {"encoder": f"{constraint}_encoder", "predictor": object()},
        }

    def fake_estimate_gauge(encoder, predictor, lifted, family, val_split):
        trace["gauge_estimations"].append((family, encoder))
        trace["order"].append("estimate_gauge")
        return _GAUGE

    def fake_calibrate_mu_j(family, lr, calibration_config, datasets, results_root, gauge, epochs=10, seed=0):
        trace["gauges_seen"].append(gauge)
        trace["order"].append("calibrate")
        return next(calibs)

    def fake_measure_proxy_at(
        family, lr, mu_j, calibration_config, datasets, results_root, gauge, seed=0, epochs=10
    ):
        trace["gauges_seen"].append(gauge)
        trace["order"].append("proxy")
        return next(proxies), {"family": family, "phase": "phase4_remeasure"}

    monkeypatch.setattr(selection, "_sweep_lr", fake_sweep_lr)
    monkeypatch.setattr(selection, "estimate_gauge", fake_estimate_gauge)
    monkeypatch.setattr(selection, "_calibrate_mu_j", fake_calibrate_mu_j)
    monkeypatch.setattr(selection, "_measure_proxy_at", fake_measure_proxy_at)
    return trace


def test_phase4_lr_on_equals_lr_off_stops_immediately(monkeypatch):
    _patch_selection_pipeline(
        monkeypatch,
        off_lr=3e-4,
        on_lr_sequence=[3e-4],
        calib_sequence=[{"retained_mu_j": 1e-2, "verdict": "selected", "medians": {}, "runs": []}],
        proxy_sequence=[],  # must never be consulted
    )
    result = selection.run_family_selection("mlp", {"datasets": {"ou": {}, "action_conditioned": {}}}, Path("."))
    assert result == {
        "family": "mlp", "lr_off": 3e-4, "lr_on": 3e-4, "mu_j": 1e-2,
        "verdict": "selected", "runs": result["runs"],
    }


def test_phase4_remeasure_in_band_stops(monkeypatch):
    _patch_selection_pipeline(
        monkeypatch,
        off_lr=3e-4,
        on_lr_sequence=[1e-3],
        calib_sequence=[{"retained_mu_j": 1e-2, "verdict": "selected", "medians": {}, "runs": []}],
        proxy_sequence=[0.95],
    )
    result = selection.run_family_selection("mlp", {"datasets": {"ou": {}, "action_conditioned": {}}}, Path("."))
    assert result["verdict"] == "selected"
    assert result["lr_on"] == 1e-3
    assert result["mu_j"] == 1e-2


def test_phase4_cycle_recalibration_fails_band_not_reached(monkeypatch):
    _patch_selection_pipeline(
        monkeypatch,
        off_lr=3e-4,
        on_lr_sequence=[1e-3],
        calib_sequence=[
            {"retained_mu_j": 1e-2, "verdict": "selected", "medians": {}, "runs": []},  # initial
            {"retained_mu_j": None, "verdict": "band not reached", "medians": {}, "runs": []},  # cycle 1
        ],
        proxy_sequence=[1.5],  # initial re-measure: outside band
    )
    result = selection.run_family_selection("mlp", {"datasets": {"ou": {}, "action_conditioned": {}}}, Path("."))
    assert result["verdict"] == "band not reached"


def test_phase4_re_sweep_lr_unchanged_stops(monkeypatch):
    _patch_selection_pipeline(
        monkeypatch,
        off_lr=3e-4,
        on_lr_sequence=[1e-3, 1e-3],  # cycle 1's re-sweep returns the SAME lr
        calib_sequence=[
            {"retained_mu_j": 1e-2, "verdict": "selected", "medians": {}, "runs": []},
            {"retained_mu_j": 5e-2, "verdict": "selected", "medians": {}, "runs": []},
        ],
        proxy_sequence=[1.5],  # initial re-measure: outside band
    )
    result = selection.run_family_selection("mlp", {"datasets": {"ou": {}, "action_conditioned": {}}}, Path("."))
    assert result["verdict"] == "selected"
    assert result["lr_on"] == 1e-3
    assert result["mu_j"] == 5e-2


def test_phase4_exhausts_two_cycles_not_converged(monkeypatch):
    _patch_selection_pipeline(
        monkeypatch,
        off_lr=3e-4,
        on_lr_sequence=[1e-3, 3e-4, 1e-3],  # sweep, cycle1 re-sweep, cycle2 re-sweep -- always moves
        calib_sequence=[
            {"retained_mu_j": 1e-2, "verdict": "selected", "medians": {}, "runs": []},
            {"retained_mu_j": 5e-2, "verdict": "selected", "medians": {}, "runs": []},
            {"retained_mu_j": 2e-2, "verdict": "selected", "medians": {}, "runs": []},
        ],
        proxy_sequence=[1.5, 1.5, 1.5],  # initial + cycle1 + cycle2 re-measures, always outside band
    )
    result = selection.run_family_selection("mlp", {"datasets": {"ou": {}, "action_conditioned": {}}}, Path("."))
    assert result["verdict"] == "selection not converged"


# --------------------------------------------------------------------------
# PROTOCOL section 10.3: the frozen gauge is wired into the DECISION path
# --------------------------------------------------------------------------


def test_gauge_estimated_once_per_family_on_the_off_checkpoint_before_the_sweep(monkeypatch):
    # PROTOCOL section 10.3 "Reference scales": ONE estimation per family,
    # on the retained phase-OFF/OU/seed-0 checkpoint, BEFORE the mu_j sweep,
    # frozen for everything after it. Counted, not read: the worst-case path
    # (two full bounded-iteration cycles) drives 3 calibrations + 3 phase-4
    # re-measurements, and all six must receive the SAME object.
    trace = _patch_selection_pipeline(
        monkeypatch,
        off_lr=3e-4,
        on_lr_sequence=[1e-3, 3e-4, 1e-3],
        calib_sequence=[
            {"retained_mu_j": 1e-2, "verdict": "selected", "medians": {}, "runs": []},
            {"retained_mu_j": 5e-2, "verdict": "selected", "medians": {}, "runs": []},
            {"retained_mu_j": 2e-2, "verdict": "selected", "medians": {}, "runs": []},
        ],
        proxy_sequence=[1.5, 1.5, 1.5],
    )
    selection.run_family_selection("mlp", {"datasets": {"ou": {}, "action_conditioned": {}}}, Path("."))

    assert len(trace["gauge_estimations"]) == 1
    family, encoder = trace["gauge_estimations"][0]
    assert family == "mlp"
    assert encoder == "off_encoder"  # never an ON (penalized) run's checkpoint

    # ordering: OFF sweep -> gauge -> everything else. No re-estimation later.
    assert trace["order"][:2] == ["sweep_off", "estimate_gauge"]
    assert trace["order"].count("estimate_gauge") == 1

    # every downstream measurement got the SAME frozen object
    assert len(trace["gauges_seen"]) == 6
    assert all(g is _GAUGE for g in trace["gauges_seen"])


def test_two_mu_j_candidates_are_measured_with_the_identical_gauge_object(monkeypatch):
    # a per-candidate gauge would be a MOVING gauge -- identity (``is``), not
    # equality, is the property that rules it out.
    seen = {}

    def fake_measure(family, lr, mu, epochs, calibration_config, datasets, results_root, gauge, seed):
        seen[mu] = gauge
        return {1e-3: 1.5, 1e-2: 0.95, 1e-1: 0.9, 1.0: 0.5}[mu], {"mu_j": mu}

    monkeypatch.setattr(selection, "_measure_mu_j_candidate", fake_measure)
    selection._calibrate_mu_j("mlp", 1e-3, {}, {}, Path("."), _GAUGE)

    assert set(seen) == set(selection.MU_J_GRID)
    assert seen[1e-3] is seen[1.0] is _GAUGE


def test_measure_mu_j_candidate_decides_on_the_normalized_measure_not_the_raw_one(monkeypatch):
    # the bug this fixes: the calibration path called ``_measure_sigma_max_median``
    # (raw, un-declared coordinates) instead of the section 10.3 metric.
    raw_calls, normalized_calls, trained_under = [], [], []

    class _Predictor:
        pass  # no ``lift`` attribute -> lifted=False, which is all this needs

    monkeypatch.setattr(
        selection, "_fit_and_load",
        lambda config, data, results_root, gauge=None: (
            trained_under.append(gauge) or ({"encoder": "E", "predictor": _Predictor()}, Path("run"))
        ),
    )
    monkeypatch.setattr(
        selection, "_measure_sigma_max_median",
        lambda *a, **k: raw_calls.append(a) or 42.0,
    )
    monkeypatch.setattr(
        selection, "measure_sigma_max_median_normalized",
        lambda enc, pred, lifted, family, val, gauge: normalized_calls.append(gauge) or 0.95,
    )

    cfg = _tiny_calibration_config(("mlp",))
    datasets = {"ou": {"val": {}}}
    median, row = selection._measure_mu_j_candidate(
        "mlp", 1e-3, 1e-2, 1, cfg, datasets, Path("results"), _GAUGE
    )

    assert raw_calls == []  # the raw quantity never enters the decision
    assert normalized_calls == [_GAUGE]
    assert median == 0.95
    assert row["sigma_max_median"] == 0.95
    # ... and the candidate was TRAINED under that same gauge, not merely
    # measured with it (PROTOCOL section 10.3: "applied identically at
    # training time and at audit time")
    assert trained_under == [_GAUGE]
    assert row["gauge_digest"] == gauge_digest(_GAUGE)


# --------------------------------------------------------------------------
# PROTOCOL section 10.3 "Indeterminacy, not repair": terminal verbatim state
# --------------------------------------------------------------------------


def test_indeterminate_grid_point_ends_calibration_verbatim(monkeypatch):
    # one indeterminate candidate leaves the grid's median vector incomplete;
    # the band test, the monotonicity check and the geometric mean are all
    # defined over the COMPLETE vector, so the verdict is terminal and
    # verbatim -- never floored, never a silently dropped candidate, and
    # never a refinement point measured on a partial grid.
    measured = []

    def fake_measure(family, lr, mu, epochs, calibration_config, datasets, results_root, gauge, seed):
        measured.append(mu)
        median = selection.METRIC_INDETERMINATE if mu == 1e-2 else 1.5
        return median, {"mu_j": mu, "sigma_max_median": median}

    monkeypatch.setattr(selection, "_measure_mu_j_candidate", fake_measure)
    result = selection._calibrate_mu_j("mlp", 1e-3, {}, {}, Path("."), _GAUGE)

    assert result["verdict"] == "metric indeterminate"
    assert result["retained_mu_j"] is None
    assert measured == [1e-3, 1e-2]  # stops at the indeterminate point
    assert result["medians"][1e-2] == "metric indeterminate"  # kept verbatim in the trace
    assert result["runs"][-1]["sigma_max_median"] == "metric indeterminate"


def test_indeterminate_refinement_point_ends_calibration_verbatim(monkeypatch):
    def fake_measure(family, lr, mu, epochs, calibration_config, datasets, results_root, gauge, seed):
        grid = {1e-3: 1.5, 1e-2: 1.4, 1e-1: 1.3, 1.0: 1.2}
        median = grid.get(mu, selection.METRIC_INDETERMINATE)  # the refinement point
        return median, {"mu_j": mu, "sigma_max_median": median}

    monkeypatch.setattr(selection, "_measure_mu_j_candidate", fake_measure)
    result = selection._calibrate_mu_j("mlp", 1e-3, {}, {}, Path("."), _GAUGE)

    assert result["verdict"] == "metric indeterminate"
    assert result["retained_mu_j"] is None
    assert result["runs"][-1]["phase"] == "mu_j_calibration_refinement"


def test_run_family_selection_propagates_calibration_indeterminacy_verbatim(monkeypatch):
    # NOT collapsed into "band not reached": the two states mean different
    # things and the protocol reports both verbatim.
    _patch_selection_pipeline(
        monkeypatch,
        off_lr=3e-4,
        on_lr_sequence=[],
        calib_sequence=[
            {"retained_mu_j": None, "verdict": selection.METRIC_INDETERMINATE, "medians": {}, "runs": []}
        ],
        proxy_sequence=[],
    )
    result = selection.run_family_selection("mlp", {"datasets": {"ou": {}, "action_conditioned": {}}}, Path("."))
    assert result["verdict"] == "metric indeterminate"
    assert result["mu_j"] is None


def test_run_family_selection_propagates_phase4_indeterminacy_verbatim(monkeypatch):
    # the phase-4 re-measurement is the other place a median meets the band
    _patch_selection_pipeline(
        monkeypatch,
        off_lr=3e-4,
        on_lr_sequence=[1e-3],
        calib_sequence=[{"retained_mu_j": 1e-2, "verdict": "selected", "medians": {}, "runs": []}],
        proxy_sequence=[selection.METRIC_INDETERMINATE],
    )
    result = selection.run_family_selection("mlp", {"datasets": {"ou": {}, "action_conditioned": {}}}, Path("."))
    assert result["verdict"] == "metric indeterminate"


def test_decision_functions_refuse_an_indeterminate_median():
    # structural guarantee, not a convention: every consumer of a median
    # raises on a non-float rather than ordering or averaging one. Without
    # these guards a future caller could feed the string into ``sorted``
    # or ``all(...)`` and get a silent verdict instead of an error.
    poisoned = {1e-3: 1.5, 1e-2: selection.METRIC_INDETERMINATE, 1e-1: 0.9, 1.0: 0.5}
    with pytest.raises(ValueError, match="not a determinate median"):
        selection._select_mu_j(poisoned)
    with pytest.raises(ValueError, match="not a determinate median"):
        selection._refinement_candidate(poisoned)
    with pytest.raises(ValueError, match="not a determinate median"):
        selection._in_band(selection.METRIC_INDETERMINATE)


def test_in_band_matches_the_protocol_band_on_floats():
    assert selection._in_band(0.90) and selection._in_band(1.00) and selection._in_band(0.95)
    assert not selection._in_band(0.8999) and not selection._in_band(1.0001)


# --------------------------------------------------------------------------
# tier 2: empirical measurement functions on tiny real checkpoints
# --------------------------------------------------------------------------


@pytest.mark.parametrize("family", ["mlp", "min_ssm", "transformer"])
def test_measure_fr_val_loss_finite(tmp_path, family):
    run, val = _train_tiny(tmp_path, family)
    loss = selection._measure_fr_val_loss(
        run["encoder"], run["predictor"], hasattr(run["predictor"], "lift"), val
    )
    assert isinstance(loss, float)
    assert loss == pytest.approx(loss)  # not NaN
    assert 0.0 <= loss < 1e6


@pytest.mark.parametrize("family", ["mlp", "min_ssm", "transformer"])
def test_measure_sigma_max_median_finite_positive(tmp_path, family):
    # exercises all three dispatch branches of _sigma_max_at_step:
    # mlp (m always None -> total_jacobian tensor path), min_ssm (fixed-size
    # m -> total_jacobian augmented path), transformer (growing m ->
    # augmented_jacobian_excluding_identity)
    run, val = _train_tiny(tmp_path, family, epochs=1)
    median = selection._measure_sigma_max_median(
        run["encoder"], run["predictor"], hasattr(run["predictor"], "lift"), family, val
    )
    assert isinstance(median, float)
    assert median == pytest.approx(median)
    assert median > 0.0


def test_measure_sigma_max_median_conventional_even_count(monkeypatch):
    # Bug 2 (Wave 5 round 7 review): ``sigmas.sort(); sigmas[len(sigmas)//2]``
    # returns the MEDIAN-HIGH order statistic for an even count (index 2 of
    # 4 sorted values), not the conventional mean of the two central
    # values. Isolated from training: a trivial identity encoder/predictor
    # supplies the right shapes/looping, and ``_sigma_max_at_step`` is
    # monkeypatched to hand back four known, distinct values in visitation
    # order so the median arithmetic itself is what's under test.
    class _StubEncoder(torch.nn.Module):
        def forward(self, x):
            return x  # identity: d_z == d_x, exact values never matter here

    class _StubPredictor(torch.nn.Module):
        def step(self, h, u, state):
            return h, PredictorState()  # m stays None; never read (patched below)

    val = {"x": torch.zeros(1, 5, 1).numpy(), "u": torch.zeros(1, 4, 1).numpy()}  # n_ep=1, T=4
    canned = iter([3.0, 1.0, 4.0, 2.0])
    monkeypatch.setattr(selection, "_sigma_max_at_step", lambda *a, **k: next(canned))

    median = selection._measure_sigma_max_median(_StubEncoder(), _StubPredictor(), False, "mlp", val)
    # sorted([3, 1, 4, 2]) == [1, 2, 3, 4]; conventional median = (2+3)/2 = 2.5.
    # Old code returned sigmas[4 // 2] == sorted[2] == 3.0.
    assert median == pytest.approx(2.5)


# --------------------------------------------------------------------------
# PROTOCOL.md section 10.3 amendment: estimate_gauge / measure_sigma_max_median_normalized
# --------------------------------------------------------------------------


@pytest.mark.parametrize("family", ["mlp", "min_ssm", "transformer"])
def test_estimate_gauge_finite_positive_scale(tmp_path, family):
    # exercises all three Gauge shapes: mlp (h only), min_ssm (fixed-size
    # m), transformer (growing, position-indexed m). transformer needs
    # >= 2 val episodes for a finite cross-episode std -- see _train_tiny.
    n_episodes = 40 if family == "transformer" else None
    run, val = _train_tiny(tmp_path, family, epochs=1, n_episodes=n_episodes)
    gauge = selection.estimate_gauge(
        run["encoder"], run["predictor"], hasattr(run["predictor"], "lift"), family, val
    )
    s_h = gauge.h_scale()
    assert torch.isfinite(s_h).all() and (s_h > 0).all()
    if family == "mlp":
        assert gauge.s_m is None and gauge.s_m_positions is None
    elif family == "min_ssm":
        assert gauge.s_m is not None
        assert torch.isfinite(gauge.s_m).all() and (gauge.s_m > 0).all()
    else:  # transformer
        assert gauge.s_m_positions is not None and gauge.position_axis == 4
        assert torch.isfinite(gauge.s_m_positions).all() and (gauge.s_m_positions > 0).all()


@pytest.mark.parametrize("family", ["mlp", "min_ssm", "transformer"])
def test_measure_sigma_max_median_normalized_finite_positive_and_uses_the_off_gauge(tmp_path, family):
    # the real usage pattern: gauge from the OFF checkpoint, applied when
    # measuring an ON (mu_j > 0) checkpoint's margin -- never re-estimated
    # from the penalized run.
    n_episodes = 40 if family == "transformer" else None
    off_run, val = _train_tiny(tmp_path, family, epochs=1, n_episodes=n_episodes)
    gauge = selection.estimate_gauge(
        off_run["encoder"], off_run["predictor"], hasattr(off_run["predictor"], "lift"), family, val
    )

    cfg = _tiny_calibration_config((family,))
    if n_episodes is not None:
        for ds_cfg in cfg["datasets"].values():
            ds_cfg["n_episodes"] = n_episodes
    dataset = selection._build_dataset_for(cfg["datasets"]["ou"])
    on_overrides = {"mlp": {}, "min_ssm": {}, "transformer": {}}[family]
    on_config = selection._build_config(cfg, family, "on", lr=1e-2, mu_j=0.1, epochs=1)
    on_config.update(on_overrides)
    on_run, _ = selection._fit_and_load(on_config, dataset, tmp_path / "results_on")

    median = selection.measure_sigma_max_median_normalized(
        on_run["encoder"], on_run["predictor"], hasattr(on_run["predictor"], "lift"), family, dataset["val"], gauge
    )
    assert isinstance(median, float), f"expected a finite median, got {median!r}"
    assert median == pytest.approx(median)  # not NaN
    assert median > 0.0

    raw_median = selection._measure_sigma_max_median(
        on_run["encoder"], on_run["predictor"], hasattr(on_run["predictor"], "lift"), family, dataset["val"]
    )
    print(f"[{family}] normalized={median:.6g} raw={raw_median:.6g}")


def test_measure_sigma_max_median_normalized_reports_indeterminate_verbatim(monkeypatch):
    # a zero-variance reference coordinate must surface as the literal
    # string "metric indeterminate", never a float, never a raised
    # exception escaping to the caller (PROTOCOL section 10.3:
    # "Indeterminacy, not repair").
    class _StubEncoder(torch.nn.Module):
        def forward(self, x):
            return x

    class _StubPredictor(torch.nn.Module):
        def step(self, h, u, state):
            return h, PredictorState()

    val = {"x": torch.zeros(1, 3, 2).numpy(), "u": torch.zeros(1, 2, 1).numpy()}  # n_ep=1, T=2, d_h=2
    zero_scale_gauge = Gauge(s_h=torch.tensor([[1.0, 0.0]], dtype=torch.float64))

    result = selection.measure_sigma_max_median_normalized(
        _StubEncoder(), _StubPredictor(), False, "mlp", val, zero_scale_gauge
    )
    assert result == "metric indeterminate"


def test_sigma_max_at_step_memory_creating_step_uses_rectangular_jacobian(tmp_path):
    # Bug 1 (Wave 5 round 7 review): at min_ssm's FIRST TF step, m_t is
    # None but predictor.step POPULATES m_next -- there is no OLD m to
    # exclude, so the audited object must be the RECTANGULAR
    # d(h_next, m_next)/d(h_t), matching what Trainer._jac_state_penalty
    # penalizes at that same step (on_state_steps' own m_t-is-None branch
    # differentiates (h_next, m_next) w.r.t. h_t alone, per
    # jac_reg.TotalJacobianPenalty.on_state_steps). The old code called
    # total_jacobian on h_next ALONE at this step, silently dropping the
    # m_next rows.
    from sld.audit.jacobian import augmented_jacobian_excluding_identity, total_jacobian
    from sld.audit.spectral import operator_norm

    run, val = _train_tiny(tmp_path, "min_ssm", epochs=1)
    predictor_d = copy.deepcopy(run["predictor"]).double().eval()
    encoder_d = copy.deepcopy(run["encoder"]).double().eval()
    x = torch.as_tensor(val["x"][:1], dtype=torch.float64)
    u = torch.as_tensor(val["u"][:1], dtype=torch.float64)
    with torch.no_grad():
        z = encoder_d(x)
        h_true = z[:, : u.shape[1]]
    h0 = h_true[0, 0].unsqueeze(0)
    u0 = u[0, 0].unsqueeze(0)

    def step_fn(h, m, uu):
        h_next, state = predictor_d.step(h, uu, PredictorState(m=m))
        return h_next, state.m

    # reference: the exact rectangular Jacobian of (h_next, m_next) w.r.t. h0
    full = augmented_jacobian_excluding_identity(step_fn, h0, None, u0)
    assert full.n_identity_rows == 0  # no old m exists at a memory-creating step
    d_h = h0.numel()
    assert full.jacobian.shape[0] > d_h  # rows for h_next AND m_next, not h_next alone

    sigma = selection._sigma_max_at_step(predictor_d, "min_ssm", h0, None, u0)
    assert sigma == pytest.approx(full.sigma_max)

    # the OLD (buggy) object: sigma_max(d h_next / d h) alone -- a strictly
    # smaller Jacobian whose spectral norm need not (and, on this trained
    # checkpoint, does not) match the correct rectangular one.
    h_shape, u_shape = h0.shape, u0.shape

    def h_only_step_fn(q, uu):
        h_next, _ = predictor_d.step(q.reshape(h_shape), uu.reshape(u_shape), PredictorState())
        return h_next.reshape(-1)

    h_only_sigma = operator_norm(total_jacobian(h_only_step_fn, h0.reshape(-1), u0.reshape(-1)))
    assert sigma != pytest.approx(h_only_sigma)


def test_sigma_max_at_step_dispatch_matches_family_state_shape(tmp_path):
    # min_ssm's state.m is fixed-size once populated: verify the augmented
    # total_jacobian branch actually runs (not silently falling back) by
    # checking a captured post-t0 step has a tensor m
    run, val = _train_tiny(tmp_path, "min_ssm", epochs=1)
    predictor_d = copy.deepcopy(run["predictor"]).double().eval()
    encoder_d = copy.deepcopy(run["encoder"]).double().eval()
    x = torch.as_tensor(val["x"][:1], dtype=torch.float64)
    u = torch.as_tensor(val["u"][:1], dtype=torch.float64)
    with torch.no_grad():
        z = encoder_d(x)
        h_true = z[:, : u.shape[1]]
    state = PredictorState()
    h0 = h_true[0, 0].unsqueeze(0)
    u0 = u[0, 0].unsqueeze(0)
    assert state.m is None
    _, state = predictor_d.step(h0, u0, state)
    assert state.m is not None  # populated after the first step
    sigma = selection._sigma_max_at_step(predictor_d, "min_ssm", h_true[0, 1].unsqueeze(0), state.m, u[0, 1].unsqueeze(0))
    assert sigma > 0.0


# --------------------------------------------------------------------------
# PROTOCOL section 10.3, "applied identically at training time and at audit
# time": the object PENALIZED and the object AUDITED are one quantity
# --------------------------------------------------------------------------


def _spread_gauge(gauge: Gauge) -> Gauge:
    """``gauge`` under a fixed diagonal reparameterization spanning two
    decades -- still a legitimate frozen gauge (PROTOCOL section 10.3's
    declared domain is all finite invertible diagonal rescalings), just an
    unmistakably anisotropic one. Used to make "the gauge changes the
    penalized object" a decisive check rather than a coin flip on a toy
    checkpoint whose estimated ``S`` happens to be near-isotropic.

    Two decades, not more, on purpose: the spread also conditions the
    stochastic estimator, and a wider one raises the Monte-Carlo variance of
    the check below without making its claim any stronger (the raw and
    normalized objects already differ by orders of magnitude here)."""

    def spread(t: torch.Tensor) -> torch.Tensor:
        k = torch.arange(t.numel(), dtype=torch.float64).reshape(t.shape) % 3 - 1
        return t * (10.0**k)

    assert gauge.s_m_positions is None, "growing-memory gauges are not spread here"
    return Gauge(s_h=spread(gauge.s_h), s_m=None if gauge.s_m is None else spread(gauge.s_m))


@pytest.mark.parametrize("family", ["mlp", "min_ssm"])
def test_trained_penalty_and_audit_measure_the_same_normalized_object(tmp_path, family):
    """THE coherence lock, end to end, on a real trained checkpoint and a
    real estimated gauge -- one family per code path: ``mlp`` (tensor state,
    ``TotalJacobianPenalty.__call__``) and ``min_ssm`` (threaded ``m``,
    ``on_state_steps``).

    What is proved: the stochastic quantity ``Trainer._jac_state_penalty``
    accumulates -- through the penalty the ``Trainer`` ACTUALLY built from
    the out-of-band gauge -- converges to ``||S_out^-1 J^{q,excl} S_in||_F^2``
    of the SAME dense Jacobian the offline audit forms, and the audit's own
    entry point (``augmented_sigma_max_normalized``) reads its ``sigma_max``
    off that same matrix.

    Run under TWO gauges. The first is the real ``estimate_gauge`` output on
    this checkpoint -- the production object, whatever anisotropy it happens
    to carry. The second is that same gauge under a fixed diagonal
    reparameterization spanning four decades (inside PROTOCOL section 10.3's
    declared gauge domain), which makes the normalized object unmistakably
    different from the raw one: on a 1-epoch toy ``mlp`` the estimated
    ``S`` is nearly isotropic (measured: 2.2 % apart), so requiring
    "the gauge changes the number" of the estimated gauge alone would be a
    coin flip. The non-vacuity claim -- an UNGAUGED penalty targets the raw
    object, a materially different number -- is therefore made on the
    stressed gauge, where it is unambiguous.
    """
    from sld.audit.jacobian import (
        augmented_jacobian_excluding_identity,
        augmented_sigma_max_normalized,
    )
    from sld.audit.spectral import operator_norm
    from sld.objectives.jac_reg import gauge_digest, normalize_jacobian
    from sld.training.trainer import Trainer

    # --- the frozen phase-OFF gauge, estimated exactly as selection does ---
    off_run, val = _train_tiny(tmp_path, family, epochs=1)
    lifted = hasattr(off_run["predictor"], "lift")
    gauge = selection.estimate_gauge(off_run["encoder"], off_run["predictor"], lifted, family, val)

    cfg = _tiny_calibration_config((family,))
    on_config = selection._build_config(cfg, family, "on", lr=1e-2, mu_j=0.1, epochs=1)
    on_config["k_probes"] = 8  # variance reduction for the Monte-Carlo check only

    # --- one captured TF step, exactly as Trainer._composite_loss captures ---
    predictor = off_run["predictor"]
    x = torch.as_tensor(val["x"][:1], dtype=torch.float32)
    u = torch.as_tensor(val["u"][:1], dtype=torch.float32)
    with torch.no_grad():
        h_true = off_run["encoder"](x)[:, : u.shape[1]]
    state = PredictorState()
    h_0, u_0 = h_true[0, 0].unsqueeze(0), u[0, 0].unsqueeze(0)
    if family == "min_ssm":
        # advance one step so m_t is a real tensor: that is the step whose
        # augmented (h, m) Jacobian the pytree path penalizes
        with torch.no_grad():
            _, state = predictor.step(h_0, u_0, state)
        assert state.m is not None
        h_t, u_t = h_true[0, 1].unsqueeze(0), u[0, 1].unsqueeze(0)
    else:
        assert state.m is None
        h_t, u_t = h_0, u_0
    m_t = state.m
    jac_steps = [(h_t, m_t, u_t)]

    # --- the audited object: dense, float64, normalized by the same gauge ---
    predictor_d = copy.deepcopy(predictor).double().eval()

    def step_fn(h, m, uu):
        h_next, st = predictor_d.step(h, uu, PredictorState(m=m))
        return h_next, st.m

    h_d = h_t.double()
    m_d = None if m_t is None else m_t.double()
    u_d = u_t.double()
    with torch.no_grad():
        _, m_next_d = step_fn(h_d, m_d, u_d)
    dense = augmented_jacobian_excluding_identity(step_fn, h_d, m_d, u_d)
    raw = float((dense.jacobian**2).sum())

    def audited_frobenius_sq_and_sigma_max(g):
        s_h = g.h_scale().reshape(-1).numpy()
        s_in = s_h if m_d is None else np.concatenate([s_h, g.m_scale(m_d).reshape(-1).numpy()])
        s_out_full = (
            s_h
            if m_next_d is None
            else np.concatenate([s_h, g.m_scale(m_next_d).reshape(-1).numpy()])
        )
        s_out = np.delete(s_out_full, dense.excluded_row_indices)
        M = normalize_jacobian(dense.jacobian, s_in, s_out)
        # the audit's public entry point reads sigma_max off that SAME matrix
        sigma = augmented_sigma_max_normalized(step_fn, h_d, m_d, u_d, g)
        assert sigma == pytest.approx(operator_norm(M), rel=1e-9)
        return float((M**2).sum()), sigma

    def penalized_mean(g, n_draws=400):
        trainer = Trainer(dict(on_config, results_root=str(tmp_path / "on")), gauge=g)
        if g is not None:
            assert trainer._jac_penalty.gauge is g
            assert trainer.config["gauge_digest"] == gauge_digest(g)
        assert trainer._jac_penalty.step_subsample == 1.0  # every step kept: no HT noise
        torch.manual_seed(0)  # the penalty draws from the GLOBAL RNG by design
        draws = [float(trainer._jac_state_penalty(predictor, jac_steps).item()) for _ in range(n_draws)]
        return sum(draws) / len(draws)

    for tag, g in (("estimated", gauge), ("stressed", _spread_gauge(gauge))):
        truth, sigma = audited_frobenius_sq_and_sigma_max(g)
        mean_est = penalized_mean(g)
        rel_err = abs(mean_est - truth) / truth
        print(
            f"[{family}/{tag}] penalized={mean_est:.6g} audited_frobenius_sq={truth:.6g} "
            f"raw_frobenius_sq={raw:.6g} rel_err={rel_err:.3%} sigma_max={sigma:.6g}"
        )
        assert rel_err < 0.05
        if tag == "stressed":
            # non-vacuity: under a genuinely anisotropic gauge the normalized
            # object is NOT the raw one, and an UNGAUGED penalty converges to
            # the raw one instead -- exactly the training/audit mismatch this
            # wiring closes
            assert abs(truth - raw) / raw > 0.5
            raw_mean = penalized_mean(None)
            assert abs(raw_mean - raw) / raw < 0.05
            assert abs(raw_mean - truth) / truth > 0.5


def test_phase_on_sweeps_train_under_the_frozen_gauge(monkeypatch):
    # the training side of the plumbing: the ON-cell LR sweeps are TRAINING
    # runs with a live penalty, so they must be handed the same frozen gauge
    # the measurements use. The OFF sweep must not be (it is what produces
    # the checkpoint the gauge is estimated from).
    trace = _patch_selection_pipeline(
        monkeypatch,
        off_lr=3e-4,
        on_lr_sequence=[1e-3],
        calib_sequence=[{"retained_mu_j": 1e-2, "verdict": "selected", "medians": {}, "runs": []}],
        proxy_sequence=[0.95],
    )
    selection.run_family_selection("mlp", {"datasets": {"ou": {}}}, Path("."))

    by_constraint = dict(trace["sweep_gauges"])
    assert by_constraint["off"] is None
    assert by_constraint["on"] is _GAUGE
    assert all(g is _GAUGE for g in trace["gauges_seen"])


def test_fit_and_load_hands_the_gauge_to_the_trainer(monkeypatch):
    # _fit_and_load is the single seam through which every selection-driven
    # training run is created: prove the gauge reaches Trainer's constructor
    # from there, rather than trusting the call sites one by one.
    seen = {}

    class _FakeTrainer:
        def __init__(self, config, gauge=None):
            seen["gauge"] = gauge

        def fit(self, data):
            return {"paths": {"ckpt": "/nowhere/run/ckpt.pt"}}

    monkeypatch.setattr(selection, "Trainer", _FakeTrainer)
    monkeypatch.setattr(selection, "load_run", lambda run_dir: {"run_dir": run_dir})

    selection._fit_and_load({}, {}, Path("."), _GAUGE)
    assert seen["gauge"] is _GAUGE
    selection._fit_and_load({}, {}, Path("."))
    assert seen["gauge"] is None


# --------------------------------------------------------------------------
# tier 3: minimal end-to-end integration (one family, tiny budgets)
# --------------------------------------------------------------------------


def test_run_family_selection_end_to_end_writes_expected_keys(tmp_path):
    cfg = _tiny_calibration_config(("mlp",))
    result = selection.run_family_selection(
        "mlp", cfg, tmp_path / "results", core_epochs=1, calibration_epochs=1
    )
    assert set(result) == {"family", "lr_off", "lr_on", "mu_j", "verdict", "runs"}
    assert result["family"] == "mlp"
    assert result["lr_off"] in selection.LR_GRID
    assert result["verdict"] in {
        "selected", "band not reached", "selection not converged", selection.METRIC_INDETERMINATE,
    }
    assert len(result["runs"]) >= 6  # at least phase OFF's 3 LR x 2 regimes
    for row in result["runs"]:
        assert row["family"] == "mlp"

    # the DECLARED (gauge-normalized) metric really ran on real checkpoints
    # here -- not just in the mocked decision-logic tests above. Without this,
    # a silently indeterminate metric would let the whole path "pass" while
    # measuring nothing.
    calibration_medians = [
        row["sigma_max_median"] for row in result["runs"] if row["phase"].startswith("mu_j_calibration")
    ]
    assert calibration_medians
    assert all(isinstance(m, float) and m > 0.0 for m in calibration_medians)


def test_run_all_families_writes_report(tmp_path, monkeypatch):
    cfg = _tiny_calibration_config(("mlp",))
    config_path = tmp_path / "calibration.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(cfg))

    result = selection.run_all_families(
        config_path, tmp_path / "results", families=("mlp",), core_epochs=1, calibration_epochs=1
    )
    assert "mlp" in result["summary"]
    report_dir = tmp_path / "results" / "pendulum_core_selection"
    report_json = json.loads((report_dir / "selection_report.json").read_text())
    assert "mlp" in report_json["summary"]
    assert len(report_json["runs"]) == len(result["runs"])
    csv_text = (report_dir / "selection_report.csv").read_text()
    assert "family" in csv_text.splitlines()[0]


# --------------------------------------------------------------------------
# pendulum_core.yaml patcher
# --------------------------------------------------------------------------


def test_apply_selection_to_pendulum_core_patches_lr_and_mu_j(tmp_path):
    import yaml

    yaml_path = tmp_path / "pendulum_core.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "exp_name": "pendulum_core",
                "arms": [
                    {"family": "mlp", "constraint": "off", "overrides": {"mu_j": 0.0, "lr": None}},
                    {"family": "mlp", "constraint": "on", "overrides": {"mu_j": None, "lr": None}},
                ],
            }
        )
    )
    summary = {"mlp": {"lr_off": 3e-4, "lr_on": 1e-3, "mu_j": 1e-2, "verdict": "selected"}}
    selection.apply_selection_to_pendulum_core(summary, yaml_path)

    patched = yaml.safe_load(yaml_path.read_text())
    off_arm = next(a for a in patched["arms"] if a["constraint"] == "off")
    on_arm = next(a for a in patched["arms"] if a["constraint"] == "on")
    assert off_arm["overrides"]["lr"] == 3e-4
    assert on_arm["overrides"]["lr"] == 1e-3
    assert on_arm["overrides"]["mu_j"] == 1e-2


def test_apply_selection_to_pendulum_core_refuses_terminal_verdict(tmp_path):
    import yaml

    # the disk-confirmed defect: koopman_ssm's real selection report
    # terminates in "band not reached" with lr_on/mu_j both null, yet the
    # patcher used to copy them into the ON arm unconditionally.
    yaml_path = tmp_path / "pendulum_core.yaml"
    original_text = yaml.safe_dump(
        {
            "exp_name": "pendulum_core",
            "arms": [
                {
                    "family": "koopman_ssm", "constraint": "off",
                    "overrides": {"mu_j": 0.0, "lr": 3.0e-4},
                },
                {
                    "family": "koopman_ssm", "constraint": "on",
                    "overrides": {"mu_j": None, "lr": None},
                },
            ],
        }
    )
    yaml_path.write_text(original_text)
    summary = {
        "koopman_ssm": {
            "lr_off": 3.0e-4, "lr_on": None, "mu_j": None, "verdict": "band not reached",
        }
    }
    with pytest.raises(ValueError, match="koopman_ssm.*band not reached"):
        selection.apply_selection_to_pendulum_core(summary, yaml_path)

    # the refusal happens before any write -- the file on disk must be
    # byte-identical to what was there before the call, never half-patched
    assert yaml_path.read_text() == original_text
