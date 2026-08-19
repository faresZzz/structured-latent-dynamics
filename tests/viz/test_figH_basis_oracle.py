"""Tests for ``sld.viz.figH_basis_oracle`` -- PROTOCOL.md section 28's
basis-oracle root.

The one piece of logic worth locking down here is one ``gate.json``'s own
identity cannot do: separating ``on-learned`` from ``on-oracle``. Both are
``(backbone=mlp, structure=on, family=mlp_k)`` in every ``gate.json``; the
two differ only in the run's ``config.yaml`` (``lift: frozen`` for
``on-oracle``, section 28.2's table). Every fixture below is synthetic,
built entirely under ``tmp_path`` -- no real ``results_basis/`` bytes are
read except in the ``requires_basis``-guarded recompute check at the bottom,
mirroring ``tests/test_viz_ingest.py``'s own real-root pattern.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from sld.viz.figH_basis_oracle import ARMS, PREDICTIONS, _contrast, _load_arms, _verdict, make_figH

_SEEDS = (0, 1, 2, 3, 4)
_KINDS = ("off", "on-learned", "on-oracle")

# off = 0.0 every seed; on-learned = 1..5 -- the repo's own sqrt(0.5)/mean=3
# numeric lock (tests/viz/test_viz.py:521-534, reused here for an
# independent cross-check); on-oracle = 0.0 every seed, IDENTICAL to off.
# So P1 and P2 both replicate that lock (mean +3/-3, se sqrt(0.5)) and P3 is
# the trivial all-zero contrast (mean 0, se 0) -- supported/supported/
# indeterminate, the same qualitative pattern as the real root (28.7).
_E_H_GATE: dict[str, dict[int, float]] = {
    "off": {s: 0.0 for s in _SEEDS},
    "on-learned": dict(zip(_SEEDS, [1.0, 2.0, 3.0, 4.0, 5.0], strict=True)),
    "on-oracle": {s: 0.0 for s in _SEEDS},
}
_STATUS = {"off": "learns", "on-learned": "does_not_learn", "on-oracle": "learns"}


def _gate_payload(*, backbone: str, structure: str, family: str, seed: int, run_dir: str,
                   status: str, e_h_gate: float, cell_id: int) -> dict:
    return {
        "schema": "gate/1",
        "identity": {
            "cell_id": cell_id,
            "arm": {"backbone": backbone, "structure": structure, "family": family},
            "seed": seed,
            "stratum_system": "CoupledOscillators",
            "stratum_regime": "OURegime",
            "run_dir": run_dir,
        },
        "split": "test",
        "protocol": {"h_gate": 1, "gamma": 5.0, "tau_1": 0.1, "e_1_pers": 0.2, "e_recon": 1e-8, "gamma_1": 2.0},
        "measures": {
            "e_h": [e_h_gate], "persistence_e_h": [0.2], "linear_e_h": [0.3],
            "e_1": e_h_gate, "e_h_gate": e_h_gate, "m_low": 0.1, "m_high": 0.1,
            "lower_bound": True, "upper_bound": True, "near_miss": False,
        },
        "status": status,
    }


def _build_fixture(root: Path) -> None:
    """The 15-cell root: 5 ``off`` + 5 ``on-learned`` + 5 ``on-oracle``.
    ``off`` writes no ``config.yaml`` at all (``_arm_label`` never reads one
    for ``structure == "off"``, exactly like the real root's ``off`` cells).
    ``on-learned`` writes one with no ``lift`` key; ``on-oracle`` writes one
    with ``lift: frozen`` -- the only signal that tells the two apart."""
    cell_id = 0
    for kind in _KINDS:
        structure = "off" if kind == "off" else "on"
        family = "mlp" if kind == "off" else "mlp_k"
        for seed in _SEEDS:
            run_dir = f"{kind}-{seed}"
            payload = _gate_payload(
                backbone="mlp", structure=structure, family=family, seed=seed, run_dir=run_dir,
                status=_STATUS[kind], e_h_gate=_E_H_GATE[kind][seed], cell_id=cell_id,
            )
            run_path = root / run_dir
            run_path.mkdir(parents=True)
            (run_path / "gate.json").write_text(json.dumps(payload))
            if kind == "on-oracle":
                (run_path / "config.yaml").write_text(yaml.safe_dump({"lift": "frozen"}))
            elif kind == "on-learned":
                (run_path / "config.yaml").write_text(yaml.safe_dump({"lr": 1e-3}))
            cell_id += 1


# -- (a) absent / empty root -------------------------------------------------


def test_missing_root_raises_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        make_figH(root=tmp_path / "absent", out=str(tmp_path / "figH.png"))


def test_root_with_no_gate_json_raises_a_clear_error(tmp_path: Path) -> None:
    (tmp_path / "empty_root").mkdir()
    with pytest.raises(FileNotFoundError):
        make_figH(root=tmp_path / "empty_root", out=str(tmp_path / "figH.png"))


# -- negative: an 'on' run with no config.yaml cannot be classified ---------


def test_an_on_run_with_no_config_yaml_raises_a_clear_error(tmp_path: Path) -> None:
    """NEGATIVE: ``gate.json``'s own identity cannot tell ``on-learned``
    from ``on-oracle`` apart -- the module falls back to the run's
    ``config.yaml``, and a ``structure='on'`` run that carries none must
    raise, never silently default to one of the two arms."""
    root = tmp_path / "results_basis"
    run_dir = "on-nolift-0"
    payload = _gate_payload(
        backbone="mlp", structure="on", family="mlp_k", seed=0, run_dir=run_dir,
        status="learns", e_h_gate=0.1, cell_id=0,
    )
    run_path = root / run_dir
    run_path.mkdir(parents=True)
    (run_path / "gate.json").write_text(json.dumps(payload))
    # no config.yaml written

    with pytest.raises(FileNotFoundError, match="config.yaml"):
        make_figH(root=root, out=str(tmp_path / "figH.png"))


# -- (b) the figure builds and the sidecar CSV carries 15 rows --------------


def test_make_figH_builds_and_the_sidecar_csv_has_15_rows_and_the_right_columns(tmp_path: Path) -> None:
    root = tmp_path / "results_basis"
    root.mkdir()
    _build_fixture(root)

    out_path = tmp_path / "assets" / "figH_basis_oracle.png"
    returned = make_figH(root=root, out=str(out_path))

    assert returned == out_path
    assert out_path.exists() and out_path.stat().st_size > 0

    csv_path = out_path.with_suffix(".csv")
    sha_path = csv_path.with_suffix(csv_path.suffix + ".sha256")
    assert csv_path.exists()
    assert sha_path.exists()
    import hashlib
    assert sha_path.read_text().strip() == hashlib.sha256(csv_path.read_bytes()).hexdigest()

    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 15
    assert set(rows[0].keys()) == {"arm", "seed", "e_h_gate", "verdict"}
    assert {r["arm"] for r in rows} == set(ARMS)
    for kind in _KINDS:
        assert sorted(int(r["seed"]) for r in rows if r["arm"] == kind) == list(_SEEDS)

    # the arm split is exactly what config.yaml's lift field determined --
    # this is the regression lock on _arm_label, the fragile piece
    by_key = {(r["arm"], int(r["seed"])): float(r["e_h_gate"]) for r in rows}
    verdict_by_key = {(r["arm"], int(r["seed"])): r["verdict"] for r in rows}
    for kind in _KINDS:
        for seed in _SEEDS:
            assert by_key[(kind, seed)] == pytest.approx(_E_H_GATE[kind][seed])
            assert verdict_by_key[(kind, seed)] == _STATUS[kind]


def test_make_figH_contrasts_match_an_independent_numpy_computation(tmp_path: Path) -> None:
    """REGLES section 7.1 (two checks): the module computes P1/P2/P3 via a
    pandas pivot inside ``_contrast``; this recomputes the same three paired
    differences by hand from the sidecar CSV with plain numpy, independent
    of that code path, and checks both land on the same, independently
    checkable-by-inspection numbers."""
    root = tmp_path / "results_basis"
    root.mkdir()
    _build_fixture(root)
    out_path = tmp_path / "figH.png"
    make_figH(root=root, out=str(out_path))

    rows = list(csv.DictReader(out_path.with_suffix(".csv").open()))
    by_arm_seed = {(r["arm"], int(r["seed"])): float(r["e_h_gate"]) for r in rows}

    def series(arm: str) -> np.ndarray:
        return np.array([by_arm_seed[(arm, s)] for s in _SEEDS])

    p1, p2, p3 = (series("on-learned") - series("off"),
                  series("on-oracle") - series("on-learned"),
                  series("on-oracle") - series("off"))

    assert np.allclose(p1, [1.0, 2.0, 3.0, 4.0, 5.0])
    assert p1.mean() == pytest.approx(3.0)
    assert p1.std(ddof=1) / np.sqrt(5) == pytest.approx(np.sqrt(0.5))

    assert np.allclose(p2, [-1.0, -2.0, -3.0, -4.0, -5.0])
    assert p2.mean() == pytest.approx(-3.0)
    assert p2.std(ddof=1) / np.sqrt(5) == pytest.approx(np.sqrt(0.5))

    assert np.allclose(p3, 0.0)
    assert p3.mean() == pytest.approx(0.0)
    assert p3.std(ddof=1) == pytest.approx(0.0)

    assert _verdict(float(p1.mean()), float(p1.std(ddof=1) / np.sqrt(5)), ">0") == "supported"
    assert _verdict(float(p2.mean()), float(p2.std(ddof=1) / np.sqrt(5)), "<0") == "supported"
    assert _verdict(float(p3.mean()), float(p3.std(ddof=1) / np.sqrt(5)), "<0") == "indeterminate"


def test_verdict_covers_all_three_branches() -> None:
    """PROTOCOL.md section 28.3: ``supported`` / ``contradicted`` /
    ``indeterminate``, no p-value ever. The fixture above only exercises
    ``supported`` and ``indeterminate`` (28.7's own real verdicts) --
    ``contradicted`` is checked here directly since no root in this file
    needs to produce one."""
    assert _verdict(1.0, 0.1, ">0") == "supported"
    assert _verdict(1.0, 0.1, "<0") == "contradicted"
    assert _verdict(-1.0, 0.1, "<0") == "supported"
    assert _verdict(-1.0, 0.1, ">0") == "contradicted"
    assert _verdict(0.05, 0.1, ">0") == "indeterminate"
    assert _verdict(0.05, 0.1, "<0") == "indeterminate"


# -- the real root, when present ---------------------------------------------

BASIS_ROOT = Path("results_basis")
requires_basis = pytest.mark.skipif(not BASIS_ROOT.exists(), reason="results_basis/ not present")


@requires_basis
def test_recompute_reproduces_protocol_28_7() -> None:
    """The binding check, same pattern as
    ``tests/test_viz_ingest.py::test_recompute_reproduces_protocol_21_0``:
    PROTOCOL.md section 28.7 publishes arm means, positive-control counts,
    and the three contrasts' mean/SE/sign-counts/verdicts. This recomputes
    them from the real ``results_basis/`` gate.json + config.yaml bytes and
    checks they land on the published numbers to the printed precision."""
    df = _load_arms(BASIS_ROOT)
    assert len(df) == 15

    means = df.groupby("arm")["e_h_gate"].mean()
    assert means["off"] == pytest.approx(0.0702, abs=5e-5)
    assert means["on-learned"] == pytest.approx(0.2153, abs=5e-5)
    assert means["on-oracle"] == pytest.approx(0.0851, abs=5e-5)

    learns = df.groupby("arm")["status"].apply(lambda s: int((s == "learns").sum()))
    assert learns["off"] == 5
    assert learns["on-learned"] == 0
    assert learns["on-oracle"] == 5

    # (mean, SE, n_pos, n_neg, verdict), PROTOCOL.md section 28.7's table
    published = {
        "P1": (0.1451, 0.0351, 5, 0, "supported"),
        "P2": (-0.1302, 0.0340, 0, 5, "supported"),
        "P3": (0.0149, 0.0189, 3, 2, "indeterminate"),
    }
    for name, a, b, sign in PREDICTIONS:
        diffs, mean, se = _contrast(df, a, b)
        exp_mean, exp_se, exp_pos, exp_neg, exp_verdict = published[name]
        assert mean == pytest.approx(exp_mean, abs=5e-5), name
        assert se == pytest.approx(exp_se, abs=5e-5), name
        assert int((diffs > 0).sum()) == exp_pos, name
        assert int((diffs < 0).sum()) == exp_neg, name
        assert _verdict(mean, se, sign) == exp_verdict, name


def test_contrast_pairs_by_seed_never_by_row_order():
    """MUTANT/NEGATIVE, aimed at the exact bug PROTOCOL 28.3 names: "pairing
    by file order or group mean is forbidden". Rows are inserted with seeds in
    scrambled, MISALIGNED order between the two arms; a positional (row-order)
    pairing would compute a different, wrong mean. ``_contrast`` must still
    land on the seed-paired answer. Then one seed's value is perturbed by a
    known delta and the mean must move by EXACTLY ``delta / n`` -- the
    linear-in-one-seed identity a correctly paired mean satisfies, and one a
    broken pairing (delta smeared onto the wrong seed) fails.
    """
    learned = {0: 0.30, 1: 0.25, 2: 0.35, 3: 0.20, 4: 0.40}
    oracle = {0: 0.15, 1: 0.13, 2: 0.17, 3: 0.14, 4: 0.16}
    scrambled = [3, 0, 4, 1, 2]
    rows = [{"seed": s, "arm": "on-learned", "e_h_gate": learned[s]} for s in scrambled]
    rows += [{"seed": s, "arm": "on-oracle", "e_h_gate": oracle[s]} for s in reversed(scrambled)]
    df = pd.DataFrame(rows)

    diffs, mean, _ = _contrast(df, "on-oracle", "on-learned")
    assert mean == pytest.approx(-0.15, abs=1e-12)
    assert (int((diffs > 0).sum()), int((diffs < 0).sum())) == (0, 5)

    mutated = df.copy()
    mutated.loc[(mutated["arm"] == "on-oracle") & (mutated["seed"] == 2), "e_h_gate"] = 0.50
    new_diffs, new_mean, _ = _contrast(mutated, "on-oracle", "on-learned")
    assert new_mean == pytest.approx(mean + 0.33 / 5, abs=1e-9)
    assert (int((new_diffs > 0).sum()), int((new_diffs < 0).sum())) == (1, 4), (
        "only seed 2's sign should have flipped"
    )
