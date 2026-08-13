"""The section 23.3 selection column: where it lives, and what it refuses.

The defect this pins was real and silent-until-late: `tools/select_rates.py`
and `sld.viz.read_sweep` both declare `e_25_val` as the criterion, and both
looked for it in `metrics.csv` -- which carries the TRAINING log
(`epoch,train_loss,val_loss,lr,pred,consistency,var,cov,jac_pen,grad_norm`)
and has never carried a free-rollout quantity. `e_25` on val is produced by
`evaluate_run_free_rollout` and persisted by `sld-gate --split val` into
`gate_val.json`.

REGLES 7.1's two checks, on the reading path:

- positive -- a hand-built `gate_val.json` whose `e_h_gate` is a number chosen
  here is read back as exactly that number;
- negative -- FIVE ways of being wrong are each refused by name rather than
  silently answered: no artifact at all, an artifact that is really the test
  reading, a horizon that is not the one the column name claims, a diverged
  cell with no measures, and a non-finite value. The one that matters is the
  first: falling back to `metrics.csv` would silently minimise `val_loss`
  instead of `e_25`, changing 23.3's criterion with nothing on screen.
"""

from __future__ import annotations

import json

import pytest

from sld.experiments.gate import gate_filename
from sld.viz.read_sweep import _VAL_COLUMNS, _final_metric


def _artifact(**over) -> dict:
    payload = {
        "schema": "gate/1",
        "split": "val",
        "protocol": {"h_gate": 25},
        "measures": {"e_1": 0.05, "e_h_gate": 0.7842, "m_low": 0.156, "m_high": 2.74},
        "status": "does_not_learn",
    }
    payload.update(over)
    return payload


def _write(run_dir, payload, name="gate_val.json"):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / name).write_text(json.dumps(payload))
    return run_dir


def test_filename_carries_the_split_so_a_selection_never_overwrites_a_verdict():
    """15.5: the gate scores on test, anything that SELECTS reads val. The two
    readings are different numbers about the same run, so they must not share
    a filename -- `sld.viz.ingest.load_gate` globs `gate.json` and would
    otherwise ingest a selection reading as a verdict."""
    assert gate_filename("test") == "gate.json"
    assert gate_filename("val") == "gate_val.json"


@pytest.mark.parametrize(
    ("column", "expected"),
    [("e_1_val", 0.05), ("e_25_val", 0.7842), ("m_low_val", 0.156), ("m_high_val", 2.74)],
)
def test_positive_each_val_column_reads_its_declared_field(tmp_path, column, expected):
    run_dir = _write(tmp_path / "run", _artifact())
    assert _final_metric(run_dir, column) == (expected, "ok")


def test_positive_the_four_names_are_probe_val_readings_names():
    """These are `probe.py::_val_reading`'s keys verbatim -- the criterion 23.3
    minimises has to be the same OBJECT round 3 minimised, not a look-alike."""
    assert set(_VAL_COLUMNS) == {"e_1_val", "e_25_val", "m_low_val", "m_high_val"}


def test_negative_a_missing_artifact_never_falls_back_to_metrics_csv(tmp_path):
    """The whole point. metrics.csv IS present and DOES carry `val_loss`; a
    fallback would return it and quietly change 23.3's criterion."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.csv").write_text("epoch,train_loss,val_loss,lr\n0,7.7,6.7,0.001\n")

    value, why = _final_metric(run_dir, "e_25_val")

    assert value is None
    assert "gate_val.json absent" in why
    assert "--split val" in why, "the refusal must name the command that repairs it"
    # and the training log is still readable under its own name
    assert _final_metric(run_dir, "val_loss") == (6.7, "ok")


def test_negative_a_test_split_artifact_is_refused_not_read(tmp_path):
    """A `gate.json` copied to `gate_val.json` still says split=test. Reading
    it would score the selection on the split 16.4 forbids for selection."""
    run_dir = _write(tmp_path / "run", _artifact(split="test"))
    value, why = _final_metric(run_dir, "e_25_val")
    assert value is None and "split='test'" in why


def test_negative_a_horizon_the_column_name_does_not_claim_is_refused(tmp_path):
    """`e_25_val` names h = 25. An artifact scored at h_gate = 100 holds a
    perfectly finite number that is not the one asked for."""
    run_dir = _write(tmp_path / "run", _artifact(protocol={"h_gate": 100}))
    value, why = _final_metric(run_dir, "e_25_val")
    assert value is None and "h_gate=100" in why


def test_negative_a_diverged_cell_yields_no_value_and_says_so(tmp_path):
    """15.3: a divergence is published with its status, never as a number that
    could win an argmin. 16.2(4): a diverged run cannot win."""
    run_dir = _write(tmp_path / "run", _artifact(measures=None, status="divergence"))
    value, why = _final_metric(run_dir, "e_25_val")
    assert value is None and "divergence" in why


def test_negative_a_non_finite_criterion_is_excluded(tmp_path):
    run_dir = _write(tmp_path / "run", _artifact(measures={"e_h_gate": float("inf")}))
    value, why = _final_metric(run_dir, "e_25_val")
    assert value is None and "non finie" in why
