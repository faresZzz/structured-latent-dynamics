"""Tests for ``sld.viz.results`` -- PROTOCOL.md section 27's reading module.

Every fixture is synthetic, built entirely under ``tmp_path``: no real
``results_primary_factorial/`` bytes are read (that campaign is live). The
expected-grid yaml is ALSO synthetic (``results._CONFIGS_DIR`` is
monkeypatched to a tmp directory) so these tests never depend on the
committed ``primary_factorial.yaml`` either.

REGLES section 7.1 (the rule of two checks): every object below carries a
positive test AND a negative test, with an independent reference where one
exists (``sld.viz.fig1_rollout._aggregate_curves``'s own numeric lock,
``tests/viz/test_viz.py:511-524``).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from sld.viz import results

# The six arms PROTOCOL.md section 27.1 declares, in yaml order.
_ARMS = [
    ("mlp", "off"), ("mlp", "on"),
    ("transformer", "off"), ("transformer", "on"),
    ("ssm", "off"), ("ssm", "on"),
]
_SEEDS = (0, 1, 2, 3, 4)
_STRATUM_SYSTEM = "SynthOscillators"
_STRATUM_REGIME = "SynthOURegime"


def _write_exp_yaml(configs_dir: Path, exp_name: str, arms=_ARMS, seeds=_SEEDS) -> None:
    configs_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "datasets": {
            "s0": {
                "system": {"name": _STRATUM_SYSTEM},
                "regime": {"name": _STRATUM_REGIME},
            }
        },
        "arms": [{"backbone": b, "structure": s} for b, s in arms],
        "seeds": list(seeds),
    }
    (configs_dir / f"{exp_name}.yaml").write_text(yaml.safe_dump(doc))


def _gate_payload(
    *, cell_id, backbone, structure, family, seed, status="learns",
    e_h=(0.1, 0.2, 0.3), e_h_gate=0.3, m_low=0.05,
    vpt_median=1.0, vpt_censored=0, per_episode_vpt=(1.0,),
    config_hash="c" * 40, data_hash="d" * 8, representation_digest="r" * 8,
    run_dir="",
) -> dict:
    return {
        "schema": "gate/1",
        "identity": {
            "cell_id": cell_id,
            "arm": {"backbone": backbone, "structure": structure, "family": family},
            "seed": seed,
            "stratum_system": _STRATUM_SYSTEM,
            "stratum_regime": _STRATUM_REGIME,
            "run_dir": run_dir,
        },
        "digests": {
            "config_hash": config_hash, "data_hash": data_hash,
            "representation_digest": representation_digest,
        },
        "split": "test",
        "protocol": {
            "h_gate": len(e_h), "gamma": 5, "tau_1": 0.1, "tau_vpt": 0.5,
            "horizon": len(e_h), "gamma_1": 2, "e_1_pers": 0.2, "e_recon": 1e-8,
            "attainable": True,
        },
        "measures": {
            "e_h": list(e_h), "persistence_e_h": list(e_h), "linear_e_h": list(e_h),
            "e_1": e_h[0], "e_h_gate": e_h_gate, "m_low": m_low, "m_high": m_low,
            "lower_bound": True, "upper_bound": False, "near_miss": False,
            "vpt_median": vpt_median, "vpt_censored": vpt_censored,
            "per_episode_vpt": list(per_episode_vpt) if per_episode_vpt is not None else None,
        },
        "status": status,
        "first_step": None,
    }


def _write_gate(exp_root: Path, run_dir_name: str, payload: dict) -> None:
    run_dir = exp_root / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    payload["identity"]["run_dir"] = run_dir_name
    (run_dir / "gate.json").write_text(json.dumps(payload))


def _write_failure(exp_root: Path, run_dir_name: str, *, family: str, seed: int) -> None:
    run_dir = exp_root / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "failure.json").write_text(json.dumps({
        "config_hash": "c" * 40, "data_hash": "d" * 8, "epoch": 3,
        "reason": "non-finite loss", "nonfinite_fields": ["train_loss"],
        "train_loss": "nan", "val_loss": "nan", "exception_class": "RuntimeError",
    }))
    (run_dir / "config.yaml").write_text(yaml.safe_dump({
        "family": family, "seed": seed,
        "stratum_system": _STRATUM_SYSTEM, "stratum_regime": _STRATUM_REGIME,
    }))


def _build_full_grid(
    tmp_path: Path,
    exp_name: str = "exp",
    *,
    e_h_gate_by_family: dict[str, float] | None = None,
    status_overrides: dict[tuple[str, int], str] | None = None,
    data_hash_overrides: dict[tuple[str, int], str] | None = None,
) -> tuple[Path, str]:
    """A complete 6-family x 5-seed grid, all ``status='learns'`` unless
    ``status_overrides[(family, seed)]`` says otherwise. Every row shares one
    ``data_hash``/``representation_digest`` (one stratum, one dataset,
    section 27.1) unless ``data_hash_overrides`` breaks that for a specific
    ``(family, seed)`` -- used by the broken-pairing negative test."""
    e_h_gate_by_family = e_h_gate_by_family or {}
    status_overrides = status_overrides or {}
    data_hash_overrides = data_hash_overrides or {}
    results_root = tmp_path / "results"
    exp_root = results_root / exp_name
    idx = 0
    for backbone, structure in _ARMS:
        family = results._FAMILY_BY_PAIR[(backbone, structure)]
        for seed in _SEEDS:
            status_key = status_overrides.get((family, seed), "learns")
            data_hash = data_hash_overrides.get((family, seed), "d" * 8)
            run_dir_name = f"{family}-{seed}"
            if status_key == "divergence_training":
                _write_failure(exp_root, run_dir_name, family=family, seed=seed)
            else:
                e_h_gate = e_h_gate_by_family.get(family, 0.3)
                payload = _gate_payload(
                    cell_id=idx, backbone=backbone, structure=structure, family=family,
                    seed=seed, status=status_key, e_h_gate=e_h_gate, data_hash=data_hash,
                )
                _write_gate(exp_root, run_dir_name, payload)
            idx += 1
    _write_exp_yaml(tmp_path / "configs", exp_name)
    return results_root, exp_name


@pytest.fixture(autouse=True)
def _isolate_configs_dir(tmp_path, monkeypatch):
    """Every test gets its own synthetic ``configs/`` -- never the real
    ``src/sld/experiments/configs/``."""
    monkeypatch.setattr(results, "_CONFIGS_DIR", tmp_path / "configs")


# ---------------------------------------------------------------------------
# build_primary_results_table
# ---------------------------------------------------------------------------

def test_build_table_never_drops_a_row_even_with_nothing_on_disk(tmp_path):
    _write_exp_yaml(tmp_path / "configs", "exp")
    rows = results.build_primary_results_table(tmp_path / "results", "exp")
    assert len(rows) == len(_ARMS) * len(_SEEDS)
    assert all(r["status"] == "missing" for r in rows)
    assert all(r["e_h"] is None and r["e_h_gate"] is None for r in rows)
    # cell_id is still a well-defined, distinct number for a missing row
    assert sorted(r["cell_id"] for r in rows) == list(range(len(rows)))


def test_build_table_classifies_ok_does_not_learn_divergence_missing(tmp_path):
    results_root, exp_name = _build_full_grid(
        tmp_path,
        status_overrides={
            ("mlp", 0): "does_not_learn",
            ("mlp", 1): "divergence_training",
        },
    )
    rows = results.build_primary_results_table(results_root, exp_name)
    by_key = {(r["family"], r["seed"]): r for r in rows}
    assert by_key[("mlp", 0)]["status"] == "does_not_learn"
    assert by_key[("mlp", 1)]["status"] == "divergence"
    assert by_key[("mlp", 1)]["e_h_gate"] is None
    assert by_key[("mlp", 2)]["status"] == "ok"
    # a family untouched by overrides is fully ok
    assert all(by_key[("ssm_k", s)]["status"] == "ok" for s in _SEEDS)


def test_build_table_recovers_divergence_identity_from_config_yaml(tmp_path):
    results_root, exp_name = _build_full_grid(
        tmp_path, status_overrides={("transformer_k", 3): "divergence_training"}
    )
    rows = results.build_primary_results_table(results_root, exp_name)
    row = next(r for r in rows if r["family"] == "transformer_k" and r["seed"] == 3)
    assert row["status"] == "divergence"
    assert row["backbone"] == "transformer" and row["structure"] == "on"
    assert row["run_dir"] is not None


# ---------------------------------------------------------------------------
# aggregate_rollout_curves -- numeric lock, independent reference:
# sld.viz.fig1_rollout._aggregate_curves + tests/viz/test_viz.py:511-524
# ---------------------------------------------------------------------------

def test_aggregate_curves_reuses_fig1_rollout_lock():
    """results.py imports fig1_rollout._aggregate_curves -- same object,
    same numeric lock as the round-2 reviewer reproduction: seed curves
    [1, 1, 1, 1, 100] give mean 20.8 (not 1, which is what a median would
    say), SE = std(ddof=1)/sqrt(5)."""
    curves = np.array([[1.0], [1.0], [1.0], [1.0], [100.0]]) * np.ones((5, 3))
    mean_curve, se = results._aggregate_curves(curves)
    assert np.allclose(mean_curve, 20.8)
    expected_se = np.std([1.0, 1.0, 1.0, 1.0, 100.0], ddof=1) / np.sqrt(5)
    assert np.allclose(se, expected_se)
    assert abs(float(expected_se) - 19.8) < 1e-6


def test_aggregate_rollout_curves_five_constant_curves_analytic(tmp_path):
    """Five seeds with constant-but-different e_h -- [1, 2, 3, 4, 5] -- give
    the textbook mean 3 and SE sqrt(2.5/5) = sqrt(0.5), to 1e-12."""
    results_root = tmp_path / "results"
    exp_root = results_root / "exp"
    for i, (backbone, structure) in enumerate(_ARMS):
        family = results._FAMILY_BY_PAIR[(backbone, structure)]
        for seed in _SEEDS:
            value = float(seed + 1) if family == "mlp" else 0.3
            payload = _gate_payload(
                cell_id=i * 5 + seed, backbone=backbone, structure=structure,
                family=family, seed=seed, e_h=(value, value, value), e_h_gate=value,
            )
            _write_gate(exp_root, f"{family}-{seed}", payload)
    _write_exp_yaml(tmp_path / "configs", "exp")

    rows = results.build_primary_results_table(results_root, "exp")
    agg = results.aggregate_rollout_curves(rows)
    assert agg["mlp"]["eligible"] is True
    mean_curve = np.array(agg["mlp"]["mean_e_h"])
    se_curve = np.array(agg["mlp"]["se_e_h"])
    assert np.allclose(mean_curve, 3.0, atol=1e-12)
    assert np.allclose(se_curve, math.sqrt(0.5), atol=1e-12)


def test_aggregate_rollout_curves_ineligible_cell_has_no_mean(tmp_path):
    results_root, exp_name = _build_full_grid(
        tmp_path, status_overrides={("mlp", 4): "does_not_learn"}
    )
    rows = results.build_primary_results_table(results_root, exp_name)
    agg = results.aggregate_rollout_curves(rows)
    assert agg["mlp"]["eligible"] is False
    assert "mean_e_h" not in agg["mlp"]
    assert agg["mlp"]["statuses"][4] == "does_not_learn"
    # untouched cell stays eligible
    assert agg["ssm_k"]["eligible"] is True


# ---------------------------------------------------------------------------
# summarize_vpt
# ---------------------------------------------------------------------------

def test_summarize_vpt_reads_stored_value_not_recomputed_from_e_h(tmp_path):
    """NEGATIVE test 4: e_h crosses 0.5 at h=1 (a naive from-e_h VPT would
    say 1), but the stored vpt_median is 7.0 -- summarize_vpt must report
    7.0 verbatim (section 27.3's 'one endpoint, one set of bytes' -- VPT is
    read off gate.json, never rederived from the e_h curve)."""
    results_root = tmp_path / "results"
    exp_root = results_root / "exp"
    for i, (backbone, structure) in enumerate(_ARMS):
        family = results._FAMILY_BY_PAIR[(backbone, structure)]
        for seed in _SEEDS:
            vpt = 7.0 if (family, seed) == ("mlp", 0) else 1.0
            payload = _gate_payload(
                cell_id=i * 5 + seed, backbone=backbone, structure=structure,
                family=family, seed=seed, e_h=(0.9, 0.9, 0.9), vpt_median=vpt,
            )
            _write_gate(exp_root, f"{family}-{seed}", payload)
    _write_exp_yaml(tmp_path / "configs", "exp")

    rows = results.build_primary_results_table(results_root, "exp")
    vpt = results.summarize_vpt(rows)
    assert vpt["mlp"]["per_seed"][0]["vpt_median"] == 7.0
    assert vpt["mlp"]["per_seed"][0]["status"] == "crossed"


def test_summarize_vpt_censors_infinite_and_missing(tmp_path):
    results_root, exp_name = _build_full_grid(
        tmp_path, status_overrides={("mlp", 2): "divergence_training"}
    )
    exp_root = results_root / exp_name
    # seed 3's mlp gate.json carries an explicit Infinity vpt_median (JSON
    # accepts this literal; a censored run per PROTOCOL.md section 14.5)
    payload = _gate_payload(
        cell_id=999, backbone="mlp", structure="off", family="mlp", seed=3,
        vpt_median=float("inf"),
    )
    (exp_root / "mlp-3" / "gate.json").write_text(json.dumps(payload))

    rows = results.build_primary_results_table(results_root, exp_name)
    vpt = results.summarize_vpt(rows)
    assert vpt["mlp"]["per_seed"][3]["status"] == "censored"
    assert vpt["mlp"]["per_seed"][3]["vpt_median"] == float("inf")
    # seed 2 has no gate.json at all (training divergence) -- also censored
    assert vpt["mlp"]["per_seed"][2]["status"] == "censored"
    assert vpt["mlp"]["per_seed"][2]["vpt_median"] is None
    assert vpt["mlp"]["n_censored"] == 2


# ---------------------------------------------------------------------------
# paired_structure_contrasts
# ---------------------------------------------------------------------------

def test_paired_structure_contrasts_numeric_lock(tmp_path):
    """on - off contrasts [1, 2, 3, 4, 5] -> mean 3, SE sqrt(0.5), 1e-12."""
    results_root = tmp_path / "results"
    exp_root = results_root / "exp"
    idx = 0
    for backbone, structure in [("mlp", "off"), ("mlp", "on")]:
        family = results._FAMILY_BY_PAIR[(backbone, structure)]
        for seed in _SEEDS:
            e_h_gate = 0.0 if structure == "off" else float(seed + 1)
            payload = _gate_payload(
                cell_id=idx, backbone=backbone, structure=structure, family=family,
                seed=seed, e_h_gate=e_h_gate,
            )
            _write_gate(exp_root, f"{family}-{seed}", payload)
            idx += 1
    _write_exp_yaml(tmp_path / "configs", "exp", arms=[("mlp", "off"), ("mlp", "on")])

    rows = results.build_primary_results_table(results_root, "exp")
    contrasts = results.paired_structure_contrasts(rows)
    assert np.allclose(contrasts["mlp"]["contrasts"], [1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(contrasts["mlp"]["mean"] - 3.0) < 1e-12
    assert abs(contrasts["mlp"]["se"] - math.sqrt(0.5)) < 1e-12


def test_paired_structure_contrasts_skips_unreadable_seed_never_fakes(tmp_path):
    """NEGATIVE test 2 (partial): a seed with a diverged ON arm contributes
    NO contrast for that seed -- never a zero, never dropped from the
    surrounding table, just absent from this specific list."""
    results_root, exp_name = _build_full_grid(
        tmp_path, status_overrides={("mlp_k", 2): "divergence_training"}
    )
    rows = results.build_primary_results_table(results_root, exp_name)
    contrasts = results.paired_structure_contrasts(rows)
    assert len(contrasts["mlp"]["contrasts"]) == 4


def test_paired_structure_contrasts_permuted_seed_raises(tmp_path):
    """NEGATIVE test 1: an ON row's data_hash disagrees with its OFF
    counterpart at the same seed (as a file-order-based pairing bug would
    produce by silently pairing across seeds) -- must raise, never return a
    silent contrast."""
    results_root, exp_name = _build_full_grid(
        tmp_path, data_hash_overrides={("mlp_k", 1): "e" * 8}
    )
    rows = results.build_primary_results_table(results_root, exp_name)
    with pytest.raises(ValueError, match="pairing broken"):
        results.paired_structure_contrasts(rows)


# ---------------------------------------------------------------------------
# evaluate_preregistered_orders
# ---------------------------------------------------------------------------

_TRUE_ORDER_E_H_GATE = {
    "ssm_k": 0.1, "transformer_k": 0.2, "mlp_k": 0.3,
    "min_ssm": 0.4, "transformer": 0.5, "mlp": 0.6,
}


def test_all_nine_relations_supported_on_a_grid_built_to_satisfy_them(tmp_path):
    results_root, exp_name = _build_full_grid(tmp_path, e_h_gate_by_family=_TRUE_ORDER_E_H_GATE)
    rows = results.build_primary_results_table(results_root, exp_name)
    verdicts = results.evaluate_preregistered_orders(rows)
    predicted = [
        "ssm_k>transformer_k", "transformer_k>mlp_k", "ssm_k>mlp_k",
        "min_ssm>transformer", "transformer>mlp", "min_ssm>mlp",
        "mlp_k>mlp", "transformer_k>transformer", "ssm_k>min_ssm",
    ]
    for relation in predicted:
        assert verdicts[relation]["verdict"] == "supported", relation
    assert verdicts["min_ssm>mlp_k"]["verdict"] == "not_tested"
    assert len(verdicts) == 10


def test_ineligible_cell_is_indeterminate_never_a_win(tmp_path):
    """POSITIVE lock for the eligibility rule: a numerically favourable but
    INELIGIBLE cell (one seed does_not_learn) must never win a relation."""
    results_root, exp_name = _build_full_grid(
        tmp_path,
        e_h_gate_by_family=_TRUE_ORDER_E_H_GATE,
        status_overrides={("mlp", 0): "does_not_learn"},
    )
    rows = results.build_primary_results_table(results_root, exp_name)
    verdicts = results.evaluate_preregistered_orders(rows)
    for relation in ("mlp_k>mlp", "transformer>mlp", "min_ssm>mlp"):
        assert verdicts[relation]["verdict"] == "indeterminate"


def test_inverted_true_order_yields_contradicted(tmp_path):
    """NEGATIVE test 3: sign check. mlp_k measurably WORSE than mlp (higher
    e_h_gate, section 27.3's already-known-to-fail leg) must verdict
    'contradicted', proving the 'lower e_h_gate is better' sign (section
    27.7) is wired the right way round -- an inverted implementation would
    say 'supported' here."""
    e_h_gate = dict(_TRUE_ORDER_E_H_GATE)
    e_h_gate["mlp_k"] = 0.9  # now worse (higher) than mlp's 0.6
    results_root, exp_name = _build_full_grid(tmp_path, e_h_gate_by_family=e_h_gate)
    rows = results.build_primary_results_table(results_root, exp_name)
    verdicts = results.evaluate_preregistered_orders(rows)
    assert verdicts["mlp_k>mlp"]["verdict"] == "contradicted"
    assert verdicts["mlp_k>mlp"]["mean"] > 0


def test_min_ssm_vs_mlp_k_never_gets_a_verdict_even_when_favourable(tmp_path):
    """NEGATIVE test 5: min_ssm the best cell, mlp_k the worst -- numerically
    this pair would 'obviously' support min_ssm > mlp_k, but the relation is
    'not_tested' by CONSTRUCTION (PROTOCOL.md section 27.3(4)): it is never
    run through the seed-paired comparison at all."""
    e_h_gate = dict(_TRUE_ORDER_E_H_GATE)
    e_h_gate["min_ssm"] = 0.01
    e_h_gate["mlp_k"] = 0.99
    results_root, exp_name = _build_full_grid(tmp_path, e_h_gate_by_family=e_h_gate)
    rows = results.build_primary_results_table(results_root, exp_name)
    verdicts = results.evaluate_preregistered_orders(rows)
    assert verdicts["min_ssm>mlp_k"] == {"verdict": "not_tested", "diffs": None, "mean": None, "se": None}
    assert not any(k not in ("min_ssm>mlp_k",) and "min_ssm" in k and "mlp_k" in k for k in verdicts)


def test_divergent_cell_never_enters_any_mean(tmp_path):
    """NEGATIVE test 2: a divergent seed makes its whole cell ineligible, so
    it contributes to NEITHER aggregate_rollout_curves' mean NOR a
    preregistered-order verdict -- confirmed on a grid otherwise built to
    satisfy every relation."""
    results_root, exp_name = _build_full_grid(
        tmp_path,
        e_h_gate_by_family=_TRUE_ORDER_E_H_GATE,
        status_overrides={("ssm_k", 1): "divergence_training"},
    )
    rows = results.build_primary_results_table(results_root, exp_name)
    agg = results.aggregate_rollout_curves(rows)
    assert agg["ssm_k"]["eligible"] is False
    verdicts = results.evaluate_preregistered_orders(rows)
    for relation in ("ssm_k>transformer_k", "ssm_k>mlp_k", "ssm_k>min_ssm"):
        assert verdicts[relation]["verdict"] == "indeterminate"
