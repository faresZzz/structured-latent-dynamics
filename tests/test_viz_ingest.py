"""The figure pipeline's contract: it reads artifacts, and what it reads
reproduces the tables the protocol already published.

The second class of test here is the important one. PROTOCOL.md section 21.0
publishes per-cell medians over five seeds; every figure built on `gate.json`
inherits those numbers. If a refactor of the loader ever changed them, the
figures would silently disagree with the protocol they cite. The rule declared
for this phase is that such a gap IS the result and no figure ships -- so the
gap is caught here, mechanically, rather than noticed later by a reader.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from sld.viz.ingest import D3_GOLD_SCHEMA, GATE_SCHEMA, arm_label, load_d3_gold, load_gate

GATE_ROOT = Path("results_gate_d")
D3_ROOT = Path("results_d3_oscillator")

# PROTOCOL.md section 21.0 -- medians over the five seeds, and the per-seed
# counts that pass each bound individually (section 12.4: no reducer).
PUBLISHED_D4: dict[tuple[str, str], tuple[float, float, float, int, int]] = {
    ("OU", "mlp"): (0.0106, 0.505, 0.540, 5, 5),
    ("OU", "transformer"): (0.0229, 0.723, 1.172, 5, 1),
    ("OU", "mlp_k"): (0.0520, 1.148, 2.656, 1, 0),
    ("OU", "transformer_k"): (0.0492, 1.112, 2.516, 1, 0),
    ("AC", "mlp"): (0.0119, 0.966, 0.777, 3, 5),
    ("AC", "transformer"): (0.0297, 0.854, 1.934, 5, 0),
    ("AC", "mlp_k"): (0.0438, 1.690, 2.857, 0, 0),
    ("AC", "transformer_k"): (0.0436, 1.502, 2.841, 0, 0),
}

# PROTOCOL.md section 20.0 -- the D3 verdict.
PUBLISHED_D3 = {"n_runs": 10, "gain_passes": 10, "state_passes": 0, "spectral_passes": 0, "action_passes": 0}

requires_gate = pytest.mark.skipif(not GATE_ROOT.exists(), reason="results_gate_d/ not present")
requires_d3 = pytest.mark.skipif(not D3_ROOT.exists(), reason="results_d3_oscillator/ not present")


@requires_gate
def test_gate_frames_have_the_declared_shape() -> None:
    scalars, curves = load_gate(GATE_ROOT)
    assert len(scalars) == 40, "phase D declares 4 arms x 2 strata x 5 seeds"
    # one curve row per (run, h); h_gate is uniform across the root
    h_gate = int(scalars["h_gate"].unique()[0])
    assert len(curves) == 40 * h_gate
    assert set(curves["h"]) == set(range(1, h_gate + 1))
    assert not curves[["e_h", "persistence_e_h", "linear_e_h"]].isna().any().any()


@requires_gate
def test_recompute_reproduces_protocol_21_0() -> None:
    """The binding check: medians and per-seed counts, cell by cell."""
    scalars, _ = load_gate(GATE_ROOT)
    scalars["arm"] = [arm_label(b, s) for b, s in zip(scalars.backbone, scalars.structure, strict=True)]
    grouped = scalars.groupby(["stratum", "arm"])
    assert len(grouped) == len(PUBLISHED_D4)
    for (stratum, arm), sub in grouped:
        e_1, m_low, m_high, n_lower, n_upper = PUBLISHED_D4[(stratum, arm)]
        assert sub["e_1"].median() == pytest.approx(e_1, abs=5e-5), f"{stratum}/{arm} e_1"
        assert sub["m_low"].median() == pytest.approx(m_low, abs=5e-4), f"{stratum}/{arm} m_low"
        assert sub["m_high"].median() == pytest.approx(m_high, abs=5e-4), f"{stratum}/{arm} m_high"
        assert int(sub["lower_bound"].sum()) == n_lower, f"{stratum}/{arm} lower-bound count"
        assert int(sub["upper_bound"].sum()) == n_upper, f"{stratum}/{arm} upper-bound count"


@requires_gate
def test_only_one_cell_of_eight_passes() -> None:
    """Section 21.0's headline sentence, as an assertion."""
    scalars, _ = load_gate(GATE_ROOT)
    scalars["arm"] = [arm_label(b, s) for b, s in zip(scalars.backbone, scalars.structure, strict=True)]
    passing = {
        key: bool(sub["lower_bound"].all() and sub["upper_bound"].all())
        for key, sub in scalars.groupby(["stratum", "arm"])
    }
    assert sum(passing.values()) == 1
    assert passing[("OU", "mlp")]


@requires_d3
def test_recompute_reproduces_protocol_20_0() -> None:
    df = load_d3_gold(D3_ROOT)
    assert len(df) == PUBLISHED_D3["n_runs"]
    assert int(df["gain_passes"].sum()) == PUBLISHED_D3["gain_passes"]
    for col, key in (("state_h100", "state_passes"), ("spectral", "spectral_passes"), ("action", "action_passes")):
        n_pass = int((df[col] < df[f"threshold_{col.split('_')[0]}"]).sum())
        assert n_pass == PUBLISHED_D3[key], f"{col}: {n_pass} pass, protocol says {PUBLISHED_D3[key]}"
    # section 20.0's medians, to one significant figure of their published form
    assert df["m_low"].median() == pytest.approx(0.156, abs=5e-3)
    assert df["state_h100"].median() == pytest.approx(0.189, abs=5e-3)


def test_unknown_schema_is_refused(tmp_path: Path) -> None:
    """A mis-parsed artifact must raise, never be guessed at."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "gate.json").write_text(json.dumps({"schema": "gate/999"}))
    with pytest.raises(ValueError, match=GATE_SCHEMA):
        load_gate(tmp_path)
    (run / "d3_gold.json").write_text(json.dumps({"schema": "nope/0"}))
    with pytest.raises(ValueError, match=D3_GOLD_SCHEMA):
        load_d3_gold(tmp_path)


def test_missing_root_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_gate(tmp_path / "absent")
    with pytest.raises(FileNotFoundError):
        load_gate(tmp_path)  # exists but holds no gate.json


def test_arm_label_matches_the_published_names() -> None:
    assert arm_label("mlp", "off") == "mlp"
    assert arm_label("mlp", "on") == "mlp_k"
    assert arm_label("transformer", "on") == "transformer_k"
