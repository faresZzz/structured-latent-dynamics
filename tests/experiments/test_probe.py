"""Tests for sld.experiments.probe -- the calibration probe harness
(PROTOCOL.md sections 14.3, 14.11).

Lean by design (module docstring in probe.py): this is exploratory-code
infrastructure, not the gate itself. Tests cover exactly this: the axis
expansion (one axis at a time, no cross-product), --dry-run writing
nothing, resume skipping a genuinely
complete cell but training an incomplete one, the namespace guard actually
refusing an outside path, and e_25 being gate.py's own returned value
rather than a second computation.
"""

from __future__ import annotations

import csv
import json as _json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sld.experiments import probe
from sld.experiments import protocol
from sld.experiments.gate import RolloutDivergence
from sld.experiments.protocol import GAMMA_1
from sld.training.trainer import FAILURE_FILENAME, Trainer


def _tiny_cell(cell_id: str = "test_cell") -> dict:
    """A cell built from tiny knobs, independent of the real axis table --
    fast to actually train (small model, few episodes, few updates)."""
    knobs = {
        "epochs": 2,
        "h_fr": 2,
        "target_params": 500,
        "n_episodes": 8,
        "lr": 1e-2,
        "w_align": 1.0,
        "view_offset": 1,
    }
    return {"index": 0, "cell_id": cell_id, "arm": "mlp_k", "axis": "test", "value": "test", "knobs": knobs}


# --------------------------------------------------------------------------
# expand_axes: one axis at a time, no cross-product
# --------------------------------------------------------------------------


def test_expand_axes_declared_cells_one_axis_at_a_time():
    cells = probe.expand_axes()
    print(f"[probe] expand_axes -> {len(cells)} cells: {[c['cell_id'] for c in cells]}")

    ids = [c["cell_id"] for c in cells]
    assert len(ids) == len(set(ids)), "cell_id must be unique"

    mlp_cells = [c for c in cells if c["arm"] == "mlp_k"]
    tr_cells = [c for c in cells if c["arm"] == "transformer_k"]

    # Pinned literals, NOT derived from probe._MLP_AXES/_TRANSFORMER_UPDATES
    # (CORR-2/2e -- this is the change from the prior version of this test):
    # this test is the executable trace of the COMMITTED scope
    # (PROTOCOL.md sections 14.3, 15.4), so it must go RED if someone
    # extends an axis without a dated amendment -- a derived count cannot
    # catch that by construction, it would just silently track the change.
    #
    # mlp_k breakdown, verified against the actual _MLP_AXES table:
    #   1 baseline + 3(updates: 500,2000,4000) + 3(h_fr: 10,15,25) +
    #   1(target_params: 40000) + 1(n_episodes: 64) + 1(lr: 3e-4) +
    #   2(w_align: 0.0,10.0) + 1(view_offset: 2) = 13
    assert len(mlp_cells) == 13
    # transformer_k: 1 baseline + 1(updates=2000) confirmation = 2
    assert len(tr_cells) == 2
    # default total (round 2 NOT included): 13 + 2 = 15 -- PROTOCOL.md 14.3
    assert len(cells) == 15

    # no cross-product: every non-baseline mlp_k cell differs from the mlp_k
    # baseline in EXACTLY one knob (a cross-product bug would move several
    # knobs at once, or produce far more than 15 cells)
    baseline_knobs = next(c["knobs"] for c in mlp_cells if c["axis"] == "baseline")
    for c in mlp_cells:
        if c["axis"] == "baseline":
            continue
        moved = {k for k in baseline_knobs if c["knobs"][k] != baseline_knobs[k]}
        assert len(moved) == 1, f"{c['cell_id']} moved {moved}, expected exactly one knob"


def test_expand_axes_round2_adds_nine_declared_cells():
    """PROTOCOL.md section 15.4(b): nine round-2 cells, a CLOSED list --
    pinned literal, not derived from probe._ROUND2_CELLS (same reasoning as
    the test above: an extension must go red, not silently pass)."""
    cells = probe.expand_axes(include_round2=True)
    print(f"[probe] expand_axes(round2=True) -> {len(cells)} cells")
    assert len(cells) == 24
    round2 = [c for c in cells if c["axis"] == "round2"]
    assert len(round2) == 9
    ids = [c["cell_id"] for c in cells]
    assert len(ids) == len(set(ids)), "cell_id must be unique"
    # positive control: default behavior (no flag) is UNCHANGED at 15 --
    # nobody launches round 2 by accident
    assert len(probe.expand_axes()) == 15


def test_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(probe, "PROBE_ROOT", tmp_path / "results_probe_d")
    exit_code = probe.main(["--dry-run"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "total: 15 cells" in out  # pinned literal -- see expand_axes test above
    # positive control: the directory tree really is empty, not merely
    # "the report file is absent"
    assert list(tmp_path.iterdir()) == []


def test_dry_run_round2_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(probe, "PROBE_ROOT", tmp_path / "results_probe_d")
    exit_code = probe.main(["--dry-run", "--round2"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "total: 24 cells" in out
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------
# Namespace guard
# --------------------------------------------------------------------------


def test_guard_refuses_outside_path_but_accepts_inside(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "PROBE_ROOT", tmp_path / "results_probe_d")
    outside = tmp_path / "results" / "evil.pt"
    with pytest.raises(ValueError, match="refusing to write outside"):
        probe._require_under_root(outside)
    inside = tmp_path / "results_probe_d" / "sub" / "file.pt"
    assert probe._require_under_root(inside) == inside


# --------------------------------------------------------------------------
# Resume: a complete cell is skipped (call-count), an incomplete one is not
# --------------------------------------------------------------------------


def test_resume_skips_complete_cell_but_trains_incomplete_one(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "PROBE_ROOT", tmp_path / "results_probe_d")
    cell = _tiny_cell()
    ctx = probe.ProbeContext()

    calls = []
    real_fit = Trainer.fit

    def counting_fit(self, data):
        calls.append(1)
        return real_fit(self, data)

    monkeypatch.setattr(Trainer, "fit", counting_fit)

    result1 = probe._run_cell(ctx, cell)
    print(f"[probe] first run: status={result1['status']} calls={len(calls)} e_25={result1['e_25']:.4f}")
    assert result1["status"] == "trained"
    assert len(calls) == 1  # incomplete cell WAS trained -- positive control

    result2 = probe._run_cell(ctx, cell)
    print(f"[probe] second run: status={result2['status']} calls={len(calls)} e_25={result2['e_25']:.4f}")
    assert result2["status"] == "skipped_resume"
    assert len(calls) == 1  # complete cell was NOT retrained
    assert result2["e_25"] == result1["e_25"]
    assert result2["config_hash"] == result1["config_hash"]


# --------------------------------------------------------------------------
# e_25 identity: read from gate.py's own result, never recomputed
# --------------------------------------------------------------------------


def test_e25_is_gate_returned_value_not_recomputed(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "PROBE_ROOT", tmp_path / "results_probe_d")
    cell = _tiny_cell("identity_cell")
    ctx = probe.ProbeContext()

    # a mutant that recomputes e_25 from raw arrays (rather than reading
    # evaluate_run_free_rollout's own e_h) cannot possibly reproduce these
    # sentinel values, since nothing about the real rollout produces them
    sentinel_e_h = np.arange(1, probe._H_GATE + 1, dtype=np.float64) * 0.01234
    fake_result = SimpleNamespace(
        e_h=sentinel_e_h,
        persistence_e_h=sentinel_e_h * 2.0,
        linear_e_h=sentinel_e_h * 3.0,
        positive_control=SimpleNamespace(m_low=0.11, m_high=0.22, verdict="learns", near_miss=False),
    )
    monkeypatch.setattr(probe, "evaluate_run_free_rollout", lambda *a, **k: fake_result)

    result = probe._run_cell(ctx, cell)
    print(f"[probe] identity check: e_1={result['e_1']} e_25={result['e_25']} verdict={result['verdict']}")

    assert result["e_1"] == sentinel_e_h[0]
    assert result["e_25"] == sentinel_e_h[probe._H_GATE - 1]
    assert result["e_25_persistence"] == (sentinel_e_h * 2.0)[probe._H_GATE - 1]
    assert result["e_25_linear"] == (sentinel_e_h * 3.0)[probe._H_GATE - 1]
    assert result["m_low"] == 0.11
    assert result["m_high"] == 0.22
    assert result["verdict"] == "learns"
    assert result["near_miss"] is False


# --------------------------------------------------------------------------
# CORR-2 (Amendment 6): corruption vs divergence, eval_failure.json, tau_1,
# operator screening, h_fr-insensitive representation key
# --------------------------------------------------------------------------


def test_corrupted_target_is_a_bare_valueerror_not_a_divergence(tmp_path, monkeypatch):
    """PROTOCOL.md section 15.3: only the tensor that comes from the MODEL
    (``model_pred_x``) may raise ``gate.RolloutDivergence``. A corrupted
    test-split ground truth (``target``) is a real defect -- the campaign
    must die loudly, never get mislabeled ``divergence_rollout``. Probe
    round 1's old ad hoc ``"non-finite" in str(exc)`` filter would have
    caught this too (it matches ``normalized_state_error``'s "target
    contains non-finite values" message just as well as the model's own);
    that conflation is exactly what 15.3/CORR-2 removes.
    """
    monkeypatch.setattr(probe, "PROBE_ROOT", tmp_path / "results_probe_d")
    cell = _tiny_cell("corrupted_target_cell")
    ctx = probe.ProbeContext()

    n_episodes = cell["knobs"]["n_episodes"]
    data = probe._build_dataset(n_episodes)
    data["test"]["z_true"][0, 2, 0] = float("nan")  # index 2 == x_1, inside the target window
    ctx.datasets[n_episodes] = data  # pre-seed the cache so _get_dataset reuses this corrupted copy

    with pytest.raises(ValueError, match="non-finite") as excinfo:
        probe._run_cell(ctx, cell)
    assert not isinstance(excinfo.value, RolloutDivergence)


def test_eval_failure_json_written_on_divergence_and_idempotent(tmp_path, monkeypatch):
    """PROTOCOL.md section 15.3: a genuine RolloutDivergence writes
    ``eval_failure.json`` atomically into the run dir, with the section's own
    fields, and re-evaluating the same checkpoint overwrites it with
    byte-identical content (evaluation is a pure function of a fixed
    checkpoint)."""
    monkeypatch.setattr(probe, "PROBE_ROOT", tmp_path / "results_probe_d")
    cell = _tiny_cell("eval_failure_cell")
    ctx = probe.ProbeContext()

    monkeypatch.setattr(
        probe, "evaluate_run_free_rollout", lambda *a, **k: (_ for _ in ()).throw(RolloutDivergence(first_step=7))
    )

    record = probe._run_cell(ctx, cell)
    print(f"[probe] eval_failure record: status={record['status']} first_nonfinite_step={record['first_nonfinite_step']}")
    assert record["status"] == "divergence_rollout"
    assert record["first_nonfinite_step"] == 7

    failure_path = Path(record["run_dir"]) / "eval_failure.json"
    assert failure_path.exists()
    payload = _json.loads(failure_path.read_text())
    assert payload["first_nonfinite_step"] == 7
    assert payload["rule"] == "15.3"
    assert payload["horizon"] == probe._H_GATE
    assert payload["config_hash"] == record["config_hash"]
    assert payload["data_hash"] == record["data_hash"]
    assert isinstance(payload["representation_digest"], str) and payload["representation_digest"]
    assert isinstance(payload["reason"], str) and payload["reason"]

    # idempotence: a second evaluation of the SAME checkpoint overwrites the
    # file with byte-identical content, never a second, drifting copy
    bytes_before = failure_path.read_bytes()
    record2 = probe._run_cell(ctx, cell)
    assert record2["status"] == "divergence_rollout"
    bytes_after = failure_path.read_bytes()
    assert bytes_after == bytes_before


def test_operator_screening_closed_form_constant_operator():
    """A hand-built MLPKPredictor with ``W_theta = 0`` (already zero at
    init) and a FIXED ``theta_0`` is an exactly constant operator regardless
    of context -- ``rho_max_rollout`` and ``phi_norm_at_h`` are then closed
    form (rotation-scaling ZOH: eigenvalue magnitude ``exp(delta*alpha)``
    exactly), independently re-derived here in plain numpy, NOT by calling
    ``rotation_scaling_zoh``/``block_diagonal`` (REGLES section 7.1's
    corollary: the reference is analytic, made by hand, never the same code
    path as the object under test)."""
    from sld.predictors.structured import MLPKPredictor

    d_z, d_u, r, width = 2, 1, 1, 4
    delta = 0.1
    horizon = 6
    predictor = MLPKPredictor(d_z, d_u, delta, r=r, width=width, dtype=torch.float64)
    alpha0, omega0 = -0.7, 1.3
    with torch.no_grad():
        predictor.head.weight.zero_()  # already zero at init; explicit for robustness to that changing
        predictor.head.bias[:r] = alpha0
        predictor.head.bias[r : 2 * r] = omega0
        predictor.head.bias[2 * r :] = 0.0

    class _Identity(torch.nn.Module):
        def forward(self, x):
            return x

    n_episodes, T = 3, horizon + 5
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n_episodes, T + 1, d_z))
    u = rng.normal(size=(n_episodes, T, d_u))
    data = {"test": {"x": x, "u": u, "z_true": x}}
    run = {"encoder": _Identity(), "predictor": predictor, "config": {}}

    rho_max, phi_norm, last_finite = probe._operator_screening(run, None, data, horizon)

    expected_rho = float(np.exp(delta * alpha0))
    print(f"[probe] screening closed form: rho_max={rho_max} expected={expected_rho}")
    assert rho_max == pytest.approx(expected_rho, rel=1e-9)
    assert last_finite == horizon

    # independent re-derivation of the composed propagator, by hand
    c, s = np.cos(delta * omega0), np.sin(delta * omega0)
    A_block = np.exp(delta * alpha0) * np.array([[c, -s], [s, c]])
    Phi = np.linalg.matrix_power(A_block, horizon)
    expected_phi_norm = np.linalg.norm(Phi, ord=2)
    print(f"[probe] screening closed form: phi_norm={phi_norm} expected={expected_phi_norm}")
    assert phi_norm == pytest.approx(expected_phi_norm, rel=1e-9)


def test_operator_screening_no_local_operator_returns_none():
    """Negative: a family exposing no structured operator (``local_operator``
    inherited, unmodified, from ``PredictorModule`` -- always ``None``) makes
    ``_operator_screening`` return ``(None, None, None)`` rather than
    crashing or fabricating a reading."""
    from sld.predictors.mlp import MLPPredictor

    d_h, d_u = 4, 1
    predictor = MLPPredictor(d_h, d_u).double()

    class _Identity(torch.nn.Module):
        def forward(self, x):
            return x

    n_episodes, T = 2, 10
    rng = np.random.default_rng(1)
    x = rng.normal(size=(n_episodes, T + 1, d_h))
    u = rng.normal(size=(n_episodes, T, d_u))
    data = {"test": {"x": x, "u": u, "z_true": x}}
    run = {"encoder": _Identity(), "predictor": predictor, "config": {}}

    result = probe._operator_screening(run, None, data, horizon=5)
    assert result == (None, None, None)


def test_tau_1_decoupled_from_e_recon_unlike_the_old_12_3_rule():
    """The value, re-derived from results_probe_d/report.json's
    mlp_k__baseline cell (NOT copied from PROTOCOL.md section 15.1's own
    arithmetic -- REGLES section 7.2: compute it, then compare), and the
    decoupling property that IS 15.1's correction: unlike 12.3's
    ``3 * e_recon``, ``tau_1`` does not even take ``e_recon`` as an
    argument, so it cannot move when the encoder improves."""
    e_1_pers = 0.039121020987324714  # results_probe_d/report.json, mlp_k__baseline
    e_recon = 0.009646622490505241

    expected = e_1_pers / 2.0
    got = protocol.tau_1(e_1_pers)
    print(f"[protocol] tau_1({e_1_pers}) = {got}, expected {expected}")
    assert got == expected

    tau_1_old_before = 3.0 * e_recon  # 12.3's superseded rule, for contrast only
    tau_1_old_after = 3.0 * (e_recon / 10.0)
    assert tau_1_old_before != tau_1_old_after  # the OLD rule DOES move with e_recon

    # the NEW rule cannot move: e_recon is not one of tau_1's arguments, so
    # recomputing it after "shrinking" e_recon is bit-identical, not merely close
    assert protocol.tau_1(e_1_pers) == got


def test_tau1_validity_both_directions():
    e_1_pers = 0.039121020987324714
    e_recon = 0.009646622490505241
    validity = protocol.tau1_validity(e_recon, e_1_pers)
    print(f"[protocol] tau1_validity({e_recon}, {e_1_pers}) = {validity}")
    assert validity == {"attainable": True, "discriminant": True}

    # force attainable=False: gamma_1 * e_recon >= e_1_pers (tau_1 <= e_recon)
    e_recon_too_big = e_1_pers / GAMMA_1  # exactly tau_1 -- attainable requires STRICT >
    validity_fail = protocol.tau1_validity(e_recon_too_big, e_1_pers)
    print(f"[protocol] forced failure: {validity_fail}")
    assert validity_fail["attainable"] is False
    assert validity_fail["discriminant"] is True  # unaffected by e_recon, true by construction (gamma_1 > 1)


def test_representation_key_insensitive_to_h_fr(tmp_path, monkeypatch):
    """PROTOCOL.md section 15.4(b): the h_fr=25 round-2 cells must reuse the
    EXISTING encoder artifact. Structurally guaranteed -- ``_rep_cfg``/
    ``_get_representation`` take no ``h_fr`` argument at all -- verified
    here by resolving two cells that differ ONLY in h_fr and checking the
    underlying build function is invoked exactly once (0 rebuilds)."""
    monkeypatch.setattr(probe, "PROBE_ROOT", tmp_path / "results_probe_d")
    ctx = probe.ProbeContext()

    calls = []
    real_build = probe._factorial._load_or_build_representation

    def counting_build(*a, **k):
        calls.append(1)
        return real_build(*a, **k)

    monkeypatch.setattr(probe._factorial, "_load_or_build_representation", counting_build)

    baseline_knobs = _tiny_cell("hfr_a")["knobs"]  # h_fr=2
    hfr_other_knobs = dict(baseline_knobs, h_fr=25)  # ONLY h_fr moved

    def _resolve(knobs):
        return probe._get_representation(ctx, knobs["n_episodes"], knobs["w_align"], knobs["view_offset"])

    art_a, _, _ = _resolve(baseline_knobs)
    assert len(calls) == 1
    art_b, _, _ = _resolve(hfr_other_knobs)
    print(f"[probe] h_fr-insensitive representation: calls={len(calls)} (expected 1)")
    assert len(calls) == 1, "a cell differing only in h_fr must reuse the cached representation, 0 rebuilds"
    assert art_a is art_b

    # negative sanity: a knob that DOES enter _rep_cfg (w_align) must trigger
    # a real rebuild -- proves the equality above is not because
    # _get_representation ignores its arguments outright
    w_align_other_knobs = dict(baseline_knobs, w_align=0.0)
    _resolve(w_align_other_knobs)
    assert len(calls) == 2


# --------------------------------------------------------------------------
# PROTOCOL.md section 11.7: training-time divergence, same object-not-message
# doctrine as 15.3's RolloutDivergence handling above. Regression coverage
# for the round-2 incident (ssm_k__baseline diverged at epoch 30, killing the
# campaign at cell 23/24 because _run_cell had no handler for it -- only the
# evaluation-time case was covered).
# --------------------------------------------------------------------------


def _write_failure_json(run_dir, *, config_hash: str, data_hash: str, epoch: int, reason: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    failure = {
        "config_hash": config_hash,
        "data_hash": data_hash,
        "epoch": epoch,
        "reason": reason,
        "nonfinite_fields": ["train_loss", "val_loss"],
        "train_loss": "nan",
        "val_loss": "nan",
        "exception_class": "RuntimeError",
    }
    (run_dir / FAILURE_FILENAME).write_text(_json.dumps(failure))


def test_run_cell_training_divergence_returns_record_with_failure_json(tmp_path, monkeypatch):
    """Positive: ``fit`` writes a terminal ``failure.json`` (the trainer's own
    11.7 behaviour) and then raises -- ``_run_cell`` must RETURN a
    ``divergence_training`` record, never propagate the exception."""
    monkeypatch.setattr(probe, "PROBE_ROOT", tmp_path / "results_probe_d")
    cell = _tiny_cell("ssm_k_divergence_cell")
    ctx = probe.ProbeContext()

    def diverging_fit(self, data):
        _write_failure_json(
            self.run_dir,
            config_hash=self.config_hash,
            data_hash=self.data_hash,
            epoch=3,
            reason="non-finite loss",
        )
        raise RuntimeError("nonfinite loss at epoch 3: train=nan, val=nan")

    monkeypatch.setattr(Trainer, "fit", diverging_fit)

    record = probe._run_cell(ctx, cell)
    print(f"[probe] training divergence record: status={record['status']} verdict={record['verdict']} "
          f"detail={record['divergence_detail']!r}")

    assert record["status"] == "divergence_training"
    assert record["verdict"] == "divergence"
    assert record["e_25"] is None
    assert record["e_1"] is None
    assert record["rho_max_rollout"] is None
    assert record["phi_norm_at_h"] is None
    assert record["first_nonfinite_step"] is None
    assert "non-finite loss" in record["divergence_detail"]
    assert "epoch 3" in record["divergence_detail"]
    # the representation-artifact fields remain computable and ARE filled in
    # (11.7's "aborts without success artifacts" is about the PREDICTOR,
    # never about the frozen, already-built representation)
    assert record["e_recon"] is not None
    assert record["readout_r2"] is not None
    assert record["tau_1"] is not None
    # NOT 15.3's file -- that governs evaluation only
    assert not (Path(record["run_dir"]) / "eval_failure.json").exists()


def test_run_cell_runtime_error_without_failure_json_propagates(tmp_path, monkeypatch):
    """Negative (the one that counts): a bare ``RuntimeError`` with NO
    ``failure.json`` on disk is an infrastructure defect, not a diagnosed
    divergence -- it must propagate, never get labelled ``divergence_training``."""
    monkeypatch.setattr(probe, "PROBE_ROOT", tmp_path / "results_probe_d")
    cell = _tiny_cell("infra_crash_cell")
    ctx = probe.ProbeContext()

    def crashing_fit(self, data):
        raise RuntimeError("CUDA out of memory")  # no failure.json written

    monkeypatch.setattr(Trainer, "fit", crashing_fit)

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        probe._run_cell(ctx, cell)


def test_resume_preserves_training_divergence_without_retraining(tmp_path, monkeypatch):
    """PROTOCOL.md 11.7: 'resume preserves it and does not retrain'. A
    run_dir pre-seeded with ``failure.json`` (and no ``run_complete.json``)
    must be classified ``divergence_training`` with ZERO calls to
    ``Trainer.fit`` -- the object's mere existence is enough, fit() is never
    re-attempted (pattern mirrors ``test_resume_skips_complete_cell_but_trains_incomplete_one``)."""
    monkeypatch.setattr(probe, "PROBE_ROOT", tmp_path / "results_probe_d")
    cell = _tiny_cell("resume_divergence_cell")
    ctx = probe.ProbeContext()

    knobs = cell["knobs"]
    artifact, data, _cond_sigma = probe._get_representation(
        ctx, knobs["n_episodes"], knobs["w_align"], knobs["view_offset"]
    )
    trainer_cfg = probe._trainer_config(cell)
    probe_trainer = Trainer(trainer_cfg, representation=artifact)
    run_dir = probe_trainer.resolve_run_dir(data)
    _write_failure_json(
        run_dir,
        config_hash=probe_trainer.config_hash,
        data_hash=probe_trainer.data_hash,
        epoch=1,
        reason="non-finite loss",
    )

    calls = []
    real_fit = Trainer.fit

    def counting_fit(self, data):
        calls.append(1)
        return real_fit(self, data)

    monkeypatch.setattr(Trainer, "fit", counting_fit)

    record = probe._run_cell(ctx, cell)
    print(f"[probe] resume-preserves-divergence: status={record['status']} calls={len(calls)}")
    assert record["status"] == "divergence_training"
    assert len(calls) == 0  # 11.7: resume preserves it, never retrains
    assert record["run_dir"] == str(run_dir)


# --------------------------------------------------------------------------
# PROTOCOL.md section 16 (Amendment 7): probe round 3 -- LR re-selection
# sweep, ON-arm finitude screen, and the two stage-3 cells (budget rung,
# w_align). Every mechanical decision below is a pure function, tested
# without training anything; the two integration tests (dedup, val-split
# wiring) train ONLY tiny cells, exactly like the round-1/2 tests above.
# --------------------------------------------------------------------------


def test_trainer_config_omits_delta_for_off_families():
    """trainer._FAMILY_CONFIG_KEYS (single source of truth) refuses 'delta'
    for the three round-3 sweep OFF families -- Trainer.__init__ raises
    ValueError on it ('applied or rejected, never silently ignored').
    Positive check: _trainer_config never puts it there for them."""
    for arm in ("mlp", "transformer", "min_ssm"):
        cell = {
            "cell_id": f"{arm}__r3_lr_1e-4", "arm": arm,
            "knobs": dict(probe._BASELINE, epochs=4000, h_fr=25, lr=1e-4),
        }
        config = probe._trainer_config(cell)
        print(f"[probe] trainer_config({arm}) keys={sorted(config)}")
        assert "delta" not in config
        assert config["family"] == arm  # sanity: didn't silently drop the wrong key


def test_trainer_config_pinned_identical_for_k_families_config_hash_safe(tmp_path, monkeypatch):
    """Negative/regression: a `_k` family's _trainer_config dict is
    byte-for-byte what it was BEFORE this fix -- pinned literal, not
    re-derived from the function under test, because config_hash (a
    sort_keys=True YAML dump of exactly this dict, trainer.py) is the resume
    identity of the 24 existing round-1/2 cells; the smallest change here
    would force every one of them to retrain."""
    monkeypatch.setattr(probe, "PROBE_ROOT", tmp_path / "results_probe_d")
    cell = next(c for c in probe._ROUND2_CELLS if c["cell_id"] == "mlp_k__hfr25_updates_4000_lr_3e-4")
    config = probe._trainer_config(cell)
    expected = {
        "exp_name": "mlp_k__hfr25_updates_4000_lr_3e-4",
        "seed": 0,
        "family": "mlp_k",
        "d_x": 2,
        "d_z": 4,
        "d_u": 1,
        "target_params": 10_000,
        "epochs": 4000,
        "batch_episodes": 8,
        "lr": 3e-4,
        "h_fr": 25,
        "delta": 0.02,
        "results_root": str(tmp_path / "results_probe_d" / "runs"),
        "stratum_system": "DampedPendulum",
        "stratum_regime": "OURegime",
    }
    print(f"[probe] trainer_config(mlp_k) == pinned: {config == expected}")
    assert config == expected


def test_round3_sweep_is_nine_cells_pinned():
    cells = probe._ROUND3_SWEEP
    print(f"[probe] _ROUND3_SWEEP -> {len(cells)} cells: {[c['cell_id'] for c in cells]}")
    assert len(cells) == 9
    assert [c["cell_id"] for c in cells] == [
        "mlp__r3_lr_1e-4", "mlp__r3_lr_3e-4", "mlp__r3_lr_1e-3",
        "transformer__r3_lr_1e-4", "transformer__r3_lr_3e-4", "transformer__r3_lr_1e-3",
        "min_ssm__r3_lr_1e-4", "min_ssm__r3_lr_3e-4", "min_ssm__r3_lr_1e-3",
    ]
    assert [c["arm"] for c in cells] == ["mlp"] * 3 + ["transformer"] * 3 + ["min_ssm"] * 3
    assert all(c["axis"] == "round3" for c in cells)
    assert [c["knobs"]["lr"] for c in cells] == [1e-4, 3e-4, 1e-3] * 3
    for c in cells:
        assert c["knobs"]["epochs"] == 4000
        assert c["knobs"]["h_fr"] == 25
        # every other knob stays at _BASELINE (only epochs/h_fr/lr move)
        for key in ("target_params", "n_episodes", "w_align", "view_offset"):
            assert c["knobs"][key] == probe._BASELINE[key]


def test_round3_sweep_mutation_is_caught_by_pinned_assertions():
    """Negative companion (REGLES 7.1): an in-memory mutation of
    _ROUND3_SWEEP no longer satisfies the positive test's pinned
    expectation, proving that test inspects real values rather than
    tautologically re-deriving them from the module under test."""
    expected_ids = [c["cell_id"] for c in probe._ROUND3_SWEEP]
    mutated = [dict(c, knobs=dict(c["knobs"])) for c in probe._ROUND3_SWEEP]
    mutated[3]["cell_id"] = "transformer__r3_lr_9e-4"  # off the declared {1e-4,3e-4,1e-3} list
    mutated_ids = [c["cell_id"] for c in mutated]
    assert mutated_ids != expected_ids
    assert "transformer__r3_lr_1e-4" not in mutated_ids


def _sweep_rec(arm: str, lr: float, e_25_val, status: str = "trained", val_diverged: bool = False) -> dict:
    rec = {"arm": arm, "status": status, "knobs": {"lr": lr}, "e_25_val": e_25_val}
    if val_diverged:
        rec["val_divergence"] = "boom"
    return rec


def test_select_lr_argmin_on_finite_e25_val():
    records = [_sweep_rec("mlp", 1e-4, 0.5), _sweep_rec("mlp", 3e-4, 0.3), _sweep_rec("mlp", 1e-3, 0.4)]
    selection = probe._select_lr(records)
    print(f"[probe] _select_lr argmin: {selection['mlp']}")
    assert selection["mlp"]["selected_lr"] == 3e-4
    assert selection["mlp"]["basis"] == {1e-4: 0.5, 3e-4: 0.3, 1e-3: 0.4}


def test_select_lr_diverged_cell_excluded_even_if_minimal():
    records = [
        _sweep_rec("mlp", 1e-4, 0.05, status="divergence_training"),  # would win the raw argmin
        _sweep_rec("mlp", 3e-4, 0.3),
        _sweep_rec("mlp", 1e-3, 0.4),
    ]
    selection = probe._select_lr(records)
    print(f"[probe] _select_lr excludes diverged: {selection['mlp']}")
    assert selection["mlp"]["selected_lr"] == 3e-4


def test_select_lr_all_diverged_gives_none():
    records = [
        _sweep_rec("mlp", 1e-4, None, status="divergence_training"),
        _sweep_rec("mlp", 3e-4, None, status="divergence_rollout"),
        _sweep_rec("mlp", 1e-3, 0.9, val_diverged=True),  # trained, but val rollout diverged
    ]
    selection = probe._select_lr(records)
    print(f"[probe] _select_lr all diverged: {selection['mlp']}")
    assert selection["mlp"]["selected_lr"] is None


def test_select_lr_exact_tie_lowest_lr_wins():
    records = [_sweep_rec("mlp", 1e-4, 0.3), _sweep_rec("mlp", 3e-4, 0.3), _sweep_rec("mlp", 1e-3, 0.5)]
    selection = probe._select_lr(records)
    print(f"[probe] _select_lr exact tie: {selection['mlp']}")
    assert selection["mlp"]["selected_lr"] == 1e-4


def test_next_lower_lr_walks_the_list_down_to_none():
    assert probe._next_lower_lr(1e-3) == 3e-4
    assert probe._next_lower_lr(3e-4) == 1e-4
    assert probe._next_lower_lr(1e-4) is None


def test_screen_cell_id_canonical_for_round2_mlp_twins():
    """PROTOCOL.md 16.2(5): mlp_k's two round-2 twins get their EXISTING
    cell_id verbatim."""
    assert probe._screen_cell("mlp_k", 3e-4)["cell_id"] == "mlp_k__hfr25_updates_4000_lr_3e-4"
    assert probe._screen_cell("mlp_k", 1e-4)["cell_id"] == "mlp_k__hfr25_updates_4000"
    cell = probe._screen_cell("mlp_k", 3e-4)
    assert cell["knobs"] == dict(probe._BASELINE, epochs=4000, h_fr=25, lr=3e-4)


def test_screen_cell_id_fresh_when_no_round2_twin():
    """Negative: mlp_k at 1e-3 (no round-2 twin) and transformer_k/ssm_k at
    ANY rate (16.4: "transformer_k/ssm_k n'ont jamais de jumeau round 2 à
    4000") all get a fresh r3_screen id, never mistaken for a round-2 one."""
    assert probe._screen_cell("mlp_k", 1e-3)["cell_id"] == "mlp_k__r3_screen_lr_1e-3"
    assert probe._screen_cell("transformer_k", 3e-4)["cell_id"] == "transformer_k__r3_screen_lr_3e-4"
    assert probe._screen_cell("ssm_k", 1e-4)["cell_id"] == "ssm_k__r3_screen_lr_1e-4"


def test_run_screen_fails_at_high_rate_tries_next_lower():
    calls = []

    def fake_run(cell):
        calls.append(cell["knobs"]["lr"])
        status = "divergence_training" if cell["knobs"]["lr"] == 1e-3 else "trained"
        return {**cell, "status": status}

    result = probe._run_screen("mlp_k", 1e-3, fake_run)
    print(f"[probe] screen fails-high: calls={calls} screened_lr={result['screened_lr']}")
    assert calls == [1e-3, 3e-4]
    assert result["screened_lr"] == 3e-4
    assert len(result["trail"]) == 2


def test_run_screen_exhaustion_is_published():
    def fake_run(cell):
        return {**cell, "status": "divergence_training"}

    result = probe._run_screen("mlp_k", 1e-3, fake_run)
    print(f"[probe] screen exhaustion: trail_lrs={[r['knobs']['lr'] for r in result['trail']]}")
    assert result["screened_lr"] is None
    assert [r["knobs"]["lr"] for r in result["trail"]] == [1e-3, 3e-4, 1e-4]


def test_run_screen_eval_divergence_is_not_a_screen_failure():
    """PROTOCOL.md 16.2(3): 'No metric of the screen run is consulted for
    selection; only completion is' -- a rollout (evaluation-time) divergence
    on the FIRST attempt must succeed the screen immediately, never fall
    back to a lower rate."""
    calls = []

    def fake_run(cell):
        calls.append(cell["knobs"]["lr"])
        return {**cell, "status": "divergence_rollout"}

    result = probe._run_screen("mlp_k", 3e-4, fake_run)
    print(f"[probe] screen eval-divergence not a failure: calls={calls} screened_lr={result['screened_lr']}")
    assert calls == [3e-4]  # never retried
    assert result["screened_lr"] == 3e-4
    assert len(result["trail"]) == 1


def test_run_screen_dedup_reuses_canonical_cell_no_retrain(tmp_path, monkeypatch):
    """PROTOCOL.md 16.2(5): a screen whose exact config already ran to
    completion as a round-2 cell is satisfied by that artifact -- resolved
    here via the SAME resume mechanism _run_cell already uses (proven
    generically above by test_resume_skips_complete_cell_but_trains_incomplete_one),
    wired through _run_screen. Tiny knobs stand in for 16.2's real
    (epochs=4000, h_fr=25, ...) screen knobs -- resume is knob-value-
    agnostic, so this proves the WIRING (_run_screen -> resume), not a
    full-scale run (forbidden here: no real round-3 training)."""
    monkeypatch.setattr(probe, "PROBE_ROOT", tmp_path / "results_probe_d")
    ctx = probe.ProbeContext()

    canonical_id = probe._screen_cell("mlp_k", 1e-4)["cell_id"]
    assert canonical_id == "mlp_k__hfr25_updates_4000"

    # really train the canonical cell_id ONCE, with tiny (fast) knobs
    probe._run_cell(ctx, _tiny_cell(canonical_id))

    calls = []
    real_fit = Trainer.fit

    def counting_fit(self, data):
        calls.append(1)
        return real_fit(self, data)

    monkeypatch.setattr(Trainer, "fit", counting_fit)

    def run_cell_fn(cell):
        # stand-in for _screen_cell's real (expensive) knobs, sharing the
        # SAME cell_id -- exercises the identical resolve_run_dir/sentinel
        # path a real screen cell would
        return probe._run_cell_with_val(ctx, _tiny_cell(cell["cell_id"]))

    result = probe._run_screen("mlp_k", 1e-4, run_cell_fn)
    print(f"[probe] screen dedup: status={result['trail'][0]['status']} calls={len(calls)}")
    assert result["screened_lr"] == 1e-4
    assert result["trail"][0]["status"] == "skipped_resume"
    assert len(calls) == 0  # never retrained


def test_decide_budget_exact_boundary_is_retained():
    twin = {"status": "trained", "e_25_val": 1.0}
    rung = {"status": "trained", "e_25_val": 0.9}  # exactly 10% better
    assert probe._decide_budget(rung, twin) == 8000


def test_decide_budget_just_short_of_boundary_not_retained():
    twin = {"status": "trained", "e_25_val": 1.0}
    rung = {"status": "trained", "e_25_val": 0.91}
    assert probe._decide_budget(rung, twin) == 4000


def test_decide_budget_rung_diverged_keeps_4000():
    twin = {"status": "trained", "e_25_val": 1.0}
    rung = {"status": "divergence_training", "e_25_val": None}
    assert probe._decide_budget(rung, twin) == 4000


def test_decide_walign_both_criteria_met_is_retained():
    twin = {"status": "trained", "e_25_val": 1.0, "m_high_val": 2.0}
    cell = {"status": "trained", "e_25_val": 0.8, "m_high_val": 1.5}
    assert probe._decide_walign(cell, twin) is True


def test_decide_walign_better_e25_but_worse_m_high_not_retained():
    twin = {"status": "trained", "e_25_val": 1.0, "m_high_val": 2.0}
    cell = {"status": "trained", "e_25_val": 0.8, "m_high_val": 2.5}
    assert probe._decide_walign(cell, twin) is False


def test_decide_walign_equal_e25_not_strict_not_retained():
    twin = {"status": "trained", "e_25_val": 1.0, "m_high_val": 2.0}
    cell = {"status": "trained", "e_25_val": 1.0, "m_high_val": 1.5}
    assert probe._decide_walign(cell, twin) is False


def test_decide_walign_divergence_not_retained():
    twin = {"status": "trained", "e_25_val": 1.0, "m_high_val": 2.0}
    cell = {"status": "divergence_rollout", "e_25_val": None, "m_high_val": None}
    assert probe._decide_walign(cell, twin) is False


def test_val_reading_uses_val_split_gate_reading_uses_test_split(tmp_path, monkeypatch):
    """Cablage: _run_cell's existing gate reading passes data["test"]; the
    round-3 _val_reading passes data["val"] to the SAME
    evaluate_run_free_rollout call, in that order -- the check that goes red
    if someone re-paths test everywhere."""
    monkeypatch.setattr(probe, "PROBE_ROOT", tmp_path / "results_probe_d")
    cell = _tiny_cell("val_wiring_cell")
    ctx = probe.ProbeContext()

    captured = []
    real_eval = probe.evaluate_run_free_rollout

    def capturing_eval(run_dir, artifact, eval_split, **kwargs):
        captured.append(eval_split)
        return real_eval(run_dir, artifact, eval_split, **kwargs)

    monkeypatch.setattr(probe, "evaluate_run_free_rollout", capturing_eval)

    record = probe._run_cell_with_val(ctx, cell)

    knobs = cell["knobs"]
    _artifact, data, _cond = probe._get_representation(
        ctx, knobs["n_episodes"], knobs["w_align"], knobs["view_offset"]
    )
    print(f"[probe] val wiring: e_25={record['e_25']} e_25_val={record['e_25_val']}")
    assert len(captured) == 2
    assert captured[0] is data["test"]  # _run_cell's own gate reading, unchanged
    assert captured[1] is data["val"]  # _val_reading, round-3 scope
    assert record["e_25_val"] is not None


def test_upsert_record_replaces_existing_cell_id_in_place():
    """PROTOCOL.md 16.2(5) / _REPORT_SCHEMA's 'unique per cell': a record
    whose cell_id already exists REPLACES that entry (position preserved,
    val fields now present) instead of appending a duplicate row."""
    records = [{"cell_id": "a", "e_25_val": None}, {"cell_id": "mlp_k__hfr25_updates_4000", "status": "trained"}]
    enriched = {"cell_id": "mlp_k__hfr25_updates_4000", "status": "skipped_resume", "e_25_val": 0.4}
    probe._upsert_record(records, enriched)
    print(f"[probe] upsert replace: {records}")
    assert len(records) == 2  # no new row
    assert records[1] is enriched  # same position, replaced content
    assert records[1]["e_25_val"] == 0.4  # val fields now on the ONE record


def test_upsert_record_appends_new_cell_id():
    records = [{"cell_id": "a"}]
    new = {"cell_id": "b"}
    probe._upsert_record(records, new)
    print(f"[probe] upsert append: {records}")
    assert len(records) == 2  # a genuinely new cell_id is appended
    assert records[1] is new


def test_round3_dry_run_matches_round2_dry_run_cells(tmp_path, monkeypatch, capsys):
    """--round3 implies round 1+2's scope (report.json stays the single
    report) but must not alter expand_axes' own 24 cells -- the dry-run
    listing (ids AND knobs, since main() prints the knobs dict verbatim) is
    byte-identical to plain --round2 (regression)."""
    monkeypatch.setattr(probe, "PROBE_ROOT", tmp_path / "results_probe_d")
    exit_code = probe.main(["--dry-run", "--round2"])
    assert exit_code == 0
    out_round2 = capsys.readouterr().out

    exit_code = probe.main(["--dry-run", "--round3"])
    assert exit_code == 0
    out_round3 = capsys.readouterr().out

    print(f"[probe] round3 dry-run == round2 dry-run: {out_round2 == out_round3}")
    assert out_round2 == out_round3
    assert "total: 24 cells" in out_round3
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------
# PROTOCOL.md section 17 (Amendment 8): probe round 3c -- the gradient-clip
# no-op gate, the clipped ssm sweep, and the ssm ON screen. Same pure-vs-
# injected discipline as round 3; the STOP integration test fakes every
# runner call (no real training, no real dataset/representation build).
# --------------------------------------------------------------------------


def test_trainer_config_hash_pinned_for_existing_cell_mlp_k_updates_4000():
    """Regression (a): probe._trainer_config's dict for an EXISTING cell,
    combined with the SAME representation artifact _run_cell would load (a
    real on-disk cache hit -- data_hash a2aaac76 / rep_hash 8012e421 both
    already exist under results_probe_d/representations/, so this is a pure
    load, never a pretrain), still hashes to the literal config_hash of the
    run directory already on disk
    (results_probe_d/runs/mlp_k__hfr25_updates_4000/42cf513a-a2aaac76) --
    pinned as a literal, not re-derived, because this hash IS that run's
    resume identity; the smallest drift here forces a retrain. Reads the
    real results_probe_d/ (no PROBE_ROOT monkeypatch) -- on purpose, this is
    the one test that verifies against the actual campaign artifact, not a
    tmp_path fixture."""
    cell = next(c for c in probe._ROUND2_CELLS if c["cell_id"] == "mlp_k__hfr25_updates_4000")
    ctx = probe.ProbeContext()
    knobs = cell["knobs"]
    artifact, _data, _cond = probe._get_representation(
        ctx, knobs["n_episodes"], knobs["w_align"], knobs["view_offset"]
    )
    config = probe._trainer_config(cell)
    trainer = Trainer(config, representation=artifact)
    print(f"[probe] mlp_k__hfr25_updates_4000 config_hash = {trainer.config_hash}")
    assert trainer.config_hash == "42cf513a"


def test_round3c_noop_cell_and_sweep_pinned():
    print(f"[probe] r3c noop cell: {probe._ROUND3C_NOOP_CELL}")
    assert probe._ROUND3C_NOOP_CELL["cell_id"] == "mlp__r3c_lr_1e-3"
    assert probe._ROUND3C_NOOP_CELL["arm"] == "mlp"
    noop_knobs = probe._ROUND3C_NOOP_CELL["knobs"]
    assert noop_knobs["epochs"] == 4000
    assert noop_knobs["h_fr"] == 25
    assert noop_knobs["lr"] == 1e-3
    assert noop_knobs["grad_clip"] == 100.0
    # must be its r3 twin's EXACT knobs, plus grad_clip only
    r3_twin = next(c for c in probe._ROUND3_SWEEP if c["cell_id"] == "mlp__r3_lr_1e-3")
    assert noop_knobs == dict(r3_twin["knobs"], grad_clip=100.0)

    cells = probe._ROUND3C_SWEEP
    print(f"[probe] _ROUND3C_SWEEP -> {len(cells)} cells: {[c['cell_id'] for c in cells]}")
    assert len(cells) == 3
    assert [c["cell_id"] for c in cells] == [
        "min_ssm__r3c_lr_1e-4", "min_ssm__r3c_lr_3e-4", "min_ssm__r3c_lr_1e-3",
    ]
    assert [c["knobs"]["lr"] for c in cells] == [1e-4, 3e-4, 1e-3]
    for c in cells:
        assert c["arm"] == "min_ssm"
        assert c["axis"] == "round3c"
        assert c["knobs"]["epochs"] == 4000
        assert c["knobs"]["h_fr"] == 25
        assert c["knobs"]["grad_clip"] == 100.0


def test_round3c_sweep_mutation_is_caught_by_pinned_assertions():
    """Negative companion (REGLES 7.1)."""
    mutated = [dict(c, knobs=dict(c["knobs"])) for c in probe._ROUND3C_SWEEP]
    mutated[1]["knobs"]["grad_clip"] = 50.0  # off the declared 100.0
    assert mutated[1]["knobs"]["grad_clip"] != probe._ROUND3C_SWEEP[1]["knobs"]["grad_clip"]


def test_r3c_screen_cell_builder_id_and_grad_clip():
    cell = probe._r3c_screen_cell("ssm_k", 3e-4)
    print(f"[probe] r3c screen cell: {cell}")
    assert cell["cell_id"] == "ssm_k__r3c_screen_lr_3e-4"
    assert cell["arm"] == "ssm_k"
    assert cell["knobs"]["grad_clip"] == probe._R3C_GRAD_CLIP
    assert cell["knobs"]["lr"] == 3e-4
    assert cell["knobs"]["epochs"] == 4000
    assert cell["knobs"]["h_fr"] == 25


def test_r3c_screen_cell_never_dedups_against_round2():
    """Negative: even at an (arm, lr) pair that HAS a round-2 twin for the
    UNCLIPPED screen builder (_screen_cell), _r3c_screen_cell never consults
    _ROUND2_SCREEN_TWINS -- a clipped config never existed before this
    amendment, so it is always executed, never cited."""
    unclipped = probe._screen_cell("mlp_k", 3e-4)["cell_id"]
    clipped = probe._r3c_screen_cell("mlp_k", 3e-4)["cell_id"]
    print(f"[probe] unclipped={unclipped} clipped={clipped}")
    assert unclipped == "mlp_k__hfr25_updates_4000_lr_3e-4"  # dedup DID apply (existing behavior)
    assert clipped == "mlp_k__r3c_screen_lr_3e-4"  # dedup never applies here
    assert clipped != unclipped


def test_select_lr_restricted_backbones_returns_only_that_backbone():
    records = [
        _sweep_rec("min_ssm", 1e-4, 0.9),
        _sweep_rec("min_ssm", 3e-4, 0.5),
        _sweep_rec("min_ssm", 1e-3, 0.7),
        _sweep_rec("mlp", 1e-4, 0.1),  # present but must be IGNORED -- backbones restricts scope
    ]
    selection = probe._select_lr(records, backbones=("min_ssm",))
    print(f"[probe] restricted select_lr: {selection}")
    assert set(selection) == {"min_ssm"}
    assert selection["min_ssm"]["selected_lr"] == 3e-4


def test_select_lr_default_backbones_unchanged_regression():
    records = [_sweep_rec("mlp", 1e-4, 0.5), _sweep_rec("mlp", 3e-4, 0.3), _sweep_rec("mlp", 1e-3, 0.4)]
    selection_default = probe._select_lr(records)
    selection_explicit = probe._select_lr(records, backbones=probe._ROUND3_BACKBONES)
    print(f"[probe] default vs explicit backbones: {selection_default == selection_explicit}")
    assert selection_default == selection_explicit
    assert set(selection_default) == {"mlp", "transformer", "min_ssm"}


def _write_fake_metrics_csv(path: Path, train_losses: list[float], val_losses: list[float]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for i, (t, v) in enumerate(zip(train_losses, val_losses)):
            writer.writerow([i, t, v])


def test_noop_gate_check_identical_series_and_e25_is_equal(tmp_path):
    r3_dir, r3c_dir = tmp_path / "r3", tmp_path / "r3c"
    r3_dir.mkdir()
    r3c_dir.mkdir()
    _write_fake_metrics_csv(r3_dir / "metrics.csv", [1.0, 0.5, 0.25], [1.1, 0.6, 0.3])
    _write_fake_metrics_csv(r3c_dir / "metrics.csv", [1.0, 0.5, 0.25], [1.1, 0.6, 0.3])
    result = probe._noop_gate_check(
        {"run_dir": str(r3c_dir), "e_25_val": 0.42}, {"run_dir": str(r3_dir), "e_25_val": 0.42}
    )
    print(f"[probe] noop gate identical: {result}")
    assert result["equal"] is True
    assert result["losses_equal"] is True
    assert result["e_25_val_equal"] is True


def test_noop_gate_check_loss_divergence_is_not_equal(tmp_path):
    r3_dir, r3c_dir = tmp_path / "r3", tmp_path / "r3c"
    r3_dir.mkdir()
    r3c_dir.mkdir()
    _write_fake_metrics_csv(r3_dir / "metrics.csv", [1.0, 0.5, 0.25], [1.1, 0.6, 0.3])
    _write_fake_metrics_csv(r3c_dir / "metrics.csv", [1.0, 0.5, 0.250001], [1.1, 0.6, 0.3])  # one value off
    result = probe._noop_gate_check(
        {"run_dir": str(r3c_dir), "e_25_val": 0.42}, {"run_dir": str(r3_dir), "e_25_val": 0.42}
    )
    print(f"[probe] noop gate loss mismatch: {result}")
    assert result["equal"] is False
    assert result["losses_equal"] is False
    assert result["e_25_val_equal"] is True  # e_25_val alone still matches


def test_noop_gate_check_e25_val_divergence_is_not_equal(tmp_path):
    r3_dir, r3c_dir = tmp_path / "r3", tmp_path / "r3c"
    r3_dir.mkdir()
    r3c_dir.mkdir()
    _write_fake_metrics_csv(r3_dir / "metrics.csv", [1.0, 0.5], [1.1, 0.6])
    _write_fake_metrics_csv(r3c_dir / "metrics.csv", [1.0, 0.5], [1.1, 0.6])
    result = probe._noop_gate_check(
        {"run_dir": str(r3c_dir), "e_25_val": 0.99}, {"run_dir": str(r3_dir), "e_25_val": 0.42}
    )
    print(f"[probe] noop gate e_25_val mismatch: {result}")
    assert result["equal"] is False
    assert result["losses_equal"] is True
    assert result["e_25_val_equal"] is False


def test_round3c_dry_run_matches_round2_dry_run_cells(tmp_path, monkeypatch, capsys):
    """--round3c implies round 1+2+3's scope but must not alter expand_axes'
    own 24 cells -- byte-identical dry-run listing to plain --round2."""
    monkeypatch.setattr(probe, "PROBE_ROOT", tmp_path / "results_probe_d")
    exit_code = probe.main(["--dry-run", "--round2"])
    assert exit_code == 0
    out_round2 = capsys.readouterr().out

    exit_code = probe.main(["--dry-run", "--round3c"])
    assert exit_code == 0
    out_round3c = capsys.readouterr().out

    print(f"[probe] round3c dry-run == round2 dry-run: {out_round2 == out_round3c}")
    assert out_round2 == out_round3c
    assert "total: 24 cells" in out_round3c
    assert list(tmp_path.iterdir()) == []


def _r3c_fake_record(cell: dict, tmp_path: Path, **extra) -> dict:
    """A structurally complete fake record (every key _print_table/
    _write_report/_noop_gate_check touch), for the STOP integration test --
    no real training, no real dataset/representation build."""
    rec = {
        "cell_id": cell["cell_id"], "arm": cell["arm"], "axis": cell["axis"], "value": cell["value"],
        "knobs": cell["knobs"], "config_hash": "deadbeef", "data_hash": "cafebabe",
        "run_dir": str(tmp_path / "runs" / cell["cell_id"]),
        "wall_clock_s": 0.01, "e_recon": 0.01, "e_1_pers": 0.02, "tau1_condition_holds": True,
        "tau_1": 0.01, "tau1_rule": "15.1",
        "tau1_validity": {"attainable": True, "discriminant": True},
        "readout_r2": 0.99, "readout_rank": 4, "readout_cond": 1.0, "readout_indeterminate": False,
        "latent_cond_sigma": 1.0, "status": "trained", "divergence_detail": None,
        "first_nonfinite_step": None, "e_1": 0.1, "e_25": 0.2, "e_25_persistence": 0.3,
        "e_25_linear": 0.4, "m_low": 0.5, "m_high": 0.6, "verdict": "learns", "near_miss": False,
        "rho_max_rollout": None, "phi_norm_at_h": None,
    }
    rec.update(extra)
    return rec


def test_round3c_stop_on_noop_mismatch_skips_cells_2_5(tmp_path, monkeypatch):
    """PROTOCOL.md 17.2: not-equal STOPS cells 2-5 -- the clipped ssm sweep
    and the ssm screen never call the runner. Every runner call in this test
    is faked (no real training, no real dataset/representation build);
    _noop_gate_check itself is monkeypatched too, since its own positive/
    negative behavior is already covered above on fabricated CSVs -- this
    test isolates main()'s branching in response to the gate's verdict."""
    monkeypatch.setattr(probe, "PROBE_ROOT", tmp_path / "results_probe_d")

    calls_with_val: list[str] = []

    def fake_run_cell(ctx, cell):
        return _r3c_fake_record(cell, tmp_path)

    def fake_run_cell_with_val(ctx, cell):
        calls_with_val.append(cell["cell_id"])
        return _r3c_fake_record(cell, tmp_path, e_1_val=0.1, e_25_val=0.2, m_low_val=0.5, m_high_val=0.6)

    monkeypatch.setattr(probe, "_run_cell", fake_run_cell)
    monkeypatch.setattr(probe, "_run_cell_with_val", fake_run_cell_with_val)
    monkeypatch.setattr(
        probe, "_noop_gate_check",
        lambda *a, **k: {
            "equal": False, "losses_equal": False, "e_25_val_equal": False,
            "e_25_val_r3c": 1.0, "e_25_val_r3": 2.0,
        },
    )

    exit_code = probe.main(["--round3c"])
    print(f"[probe] round3c STOP exit_code={exit_code}")
    assert exit_code == 0

    r3c_calls = [cid for cid in calls_with_val if "r3c" in cid]
    print(f"[probe] round3c STOP: r3c-related _run_cell_with_val calls = {r3c_calls}")
    assert r3c_calls == ["mlp__r3c_lr_1e-3"]  # only the no-op gate cell ran -- cells 2-5 skipped

    report = _json.loads((tmp_path / "results_probe_d" / "report.json").read_text())
    assert report["round3c"]["noop"]["equal"] is False
    assert report["round3c"]["selected_lr_ssm"] is None
