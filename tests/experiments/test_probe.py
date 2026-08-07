"""Tests for sld.experiments.probe -- the calibration probe harness
(PROTOCOL.md sections 14.3, 14.11).

Lean by design (module docstring in probe.py): this is exploratory-code
infrastructure, not the gate itself. Tests cover exactly what the wave
brief asks for -- the axis expansion (one axis at a time, no
cross-product), --dry-run writing nothing, resume skipping a genuinely
complete cell but training an incomplete one, the namespace guard actually
refusing an outside path, and e_25 being gate.py's own returned value
rather than a second computation.
"""

from __future__ import annotations

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
