"""Read ``gate.json`` artifacts off ``results_primary_factorial/`` into a long
results table, paired contrasts, and verdicts on PROTOCOL.md section 27's
nine pre-registered relations.

Torch-free, like every module under ``sld.viz`` (REGLES section 2): only
``numpy``, ``yaml`` and stdlib. This module does not import ``sld.training``
or ``sld.experiments.factorial`` -- both pull in ``torch`` through
``Trainer`` -- so the small pieces of those modules this reading needs
(the ``(backbone, structure) -> family`` map of section 27.1, the expected
grid's ordering) are re-declared here rather than imported. See
``_FAMILY_BY_PAIR``'s docstring for why that duplication is deliberate.

One artifact schema is read here, ``gate/1`` -- the same one
``sld.viz.ingest.load_gate`` reads. That loader is reused for its file
discovery (``ingest._read_json_tree``) and its schema constant
(``ingest.GATE_SCHEMA``); it is not called directly because it does not
surface ``measures.vpt_median``/``vpt_censored``/``per_episode_vpt``, which
this module's readers (``summarize_vpt``) need and ``load_gate`` does not
expose.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml

from sld.viz import ingest
from sld.viz.fig1_rollout import _aggregate_curves

_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "experiments" / "configs"

# PROTOCOL.md section 27.1: "The six arms resolve to exactly {mlp, mlp_k,
# transformer, transformer_k, min_ssm, ssm_k}." This is a verbatim copy of
# ``sld.experiments.factorial._FAMILY_BY_PAIR`` -- not imported because
# ``factorial.py`` imports ``sld.training.trainer`` (torch), and ``viz/``
# stays torch-free. If this map and factorial's ever disagree, that is a
# real defect; there is no third place either could drift from unnoticed
# since both are two lines and both cite this same protocol section.
_FAMILY_BY_PAIR: dict[tuple[str, str], str] = {
    ("mlp", "off"): "mlp",
    ("mlp", "on"): "mlp_k",
    ("transformer", "off"): "transformer",
    ("transformer", "on"): "transformer_k",
    ("ssm", "off"): "min_ssm",
    ("ssm", "on"): "ssm_k",
}
_PAIR_BY_FAMILY: dict[str, tuple[str, str]] = {fam: pair for pair, fam in _FAMILY_BY_PAIR.items()}

# Pre-registered seed law (PROTOCOL.md section 27.1): seeds 0-4, five seeds.
_REQUIRED_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

_ROW_KEYS = (
    "cell_id", "backbone", "structure", "family", "seed",
    "stratum_system", "stratum_regime", "run_dir",
    "config_hash", "data_hash", "representation_digest",
    "status", "e_h", "e_h_gate", "m_low",
    "vpt_median", "vpt_censored", "per_episode_vpt",
)

# A row with no gate.json and no failure.json under it: every measured field
# is None, cell_id is left absent here so build_primary_results_table's merge
# falls back to the EXPECTED cell_id computed from the yaml's own ordering.
_MISSING_INFO: dict = {
    "run_dir": None, "config_hash": None, "data_hash": None,
    "representation_digest": None, "status": "missing",
    "e_h": None, "e_h_gate": None, "m_low": None,
    "vpt_median": None, "vpt_censored": None, "per_episode_vpt": None,
}


def _expected_grid(exp_name: str) -> list[dict]:
    """The full expected ``(dataset, arm, seed)`` grid straight out of
    ``src/sld/experiments/configs/<exp_name>.yaml``, in the same order
    ``factorial.expand()`` assigns ``cell_id`` (datasets in yaml order, then
    arms in yaml order, then seeds in yaml order) -- so the ``cell_id``
    computed here agrees with a real ``gate.json``'s ``identity.cell_id``
    for any cell that has actually run, and is still a well-defined number
    for a cell that has not.
    """
    exp_yaml = _CONFIGS_DIR / f"{exp_name}.yaml"
    config = yaml.safe_load(exp_yaml.read_text())
    grid: list[dict] = []
    idx = 0
    for ds_cfg in config["datasets"].values():
        stratum_system = ds_cfg["system"]["name"]
        stratum_regime = ds_cfg["regime"]["name"]
        for arm in config["arms"]:
            backbone, structure = arm["backbone"], arm["structure"]
            pair = (backbone, structure)
            if pair not in _FAMILY_BY_PAIR:
                raise ValueError(f"{exp_yaml}: no known family for (backbone, structure) = {pair!r}")
            family = _FAMILY_BY_PAIR[pair]
            for seed in config["seeds"]:
                grid.append(
                    {
                        "cell_id": idx,
                        "backbone": backbone,
                        "structure": structure,
                        "family": family,
                        "seed": seed,
                        "stratum_system": stratum_system,
                        "stratum_regime": stratum_regime,
                    }
                )
                idx += 1
    return grid


def _scan_gate_jsons(exp_root: Path) -> dict[tuple, dict]:
    """Every ``gate.json`` under ``exp_root``, keyed by
    ``(stratum_system, stratum_regime, backbone, structure, seed)``.

    NOTE on a gap between this packet and the actual ``gate/1`` writer
    (``sld/experiments/gate.py``): the packet describes ``status`` as closed
    on ``{'learns', 'does_not_learn'}``, but ``gate.py``'s own schema (its
    ``_STATUSES`` constant and ``_evaluate_gate_cell``) allows a THIRD value,
    ``'divergence'``, written when the free ROLLOUT itself diverges at eval
    time (``measures`` is then ``null``, ``first_step`` records where) --
    distinct from a TRAINING-time divergence, which never produces a
    ``gate.json`` at all and is read from ``failure.json`` instead (see
    ``_scan_failures``). This is flagged in the final report rather than
    silently reconciled: this reader maps that third value to this module's
    own ``'divergence'`` status (all measures ``None``), which keeps every
    row honest -- a rollout-divergent cell is never averaged and never wins
    a comparison -- without inventing a threshold or silently dropping the
    row.
    """
    try:
        found = ingest._read_json_tree(exp_root, "gate.json")
    except FileNotFoundError:
        return {}
    out: dict[tuple, dict] = {}
    for path, d in found:
        if d.get("schema") != ingest.GATE_SCHEMA:
            raise ValueError(f"{path}: expected schema {ingest.GATE_SCHEMA!r}, got {d.get('schema')!r}")
        ident, arm, digests = d["identity"], d["identity"]["arm"], d["digests"]
        key = (
            ident["stratum_system"], ident["stratum_regime"],
            arm["backbone"], arm["structure"], ident["seed"],
        )
        raw_status = d["status"]
        if raw_status == "learns":
            status = "ok"
        elif raw_status == "does_not_learn":
            status = "does_not_learn"
        elif raw_status == "divergence":
            status = "divergence"
        else:
            raise ValueError(f"{path}: unknown gate status {raw_status!r}")

        if status == "divergence":
            e_h = e_h_gate = m_low = vpt_median = vpt_censored = per_episode_vpt = None
        else:
            meas = d["measures"]
            e_h = list(meas["e_h"])
            e_h_gate = meas["e_h_gate"]
            m_low = meas["m_low"]
            vpt_median = meas.get("vpt_median")
            vpt_censored = meas.get("vpt_censored")
            episodes = meas.get("per_episode_vpt")
            per_episode_vpt = list(episodes) if episodes is not None else None

        out[key] = {
            "cell_id": ident["cell_id"],
            "run_dir": ident["run_dir"],
            "config_hash": digests["config_hash"],
            "data_hash": digests["data_hash"],
            "representation_digest": digests["representation_digest"],
            "status": status,
            "e_h": e_h,
            "e_h_gate": e_h_gate,
            "m_low": m_low,
            "vpt_median": vpt_median,
            "vpt_censored": vpt_censored,
            "per_episode_vpt": per_episode_vpt,
        }
    return out


def _scan_failures(exp_root: Path, results_root: Path) -> dict[tuple, dict]:
    """Every TRAINING-time divergence under ``exp_root``: a run directory
    with a ``failure.json`` (``Trainer.fit``'s terminal sentinel, PROTOCOL.md
    section 11.7) and therefore no ``gate.json``, ever, since gating requires
    a completion sentinel that a diverged run never wrote.

    ``failure.json`` itself carries no cell identity (``config_hash``,
    ``data_hash``, ``epoch``, the offending losses -- no backbone/structure/
    seed). Identity is recovered from the run directory's OWN
    ``config.yaml`` (the same file ``sld.viz.read_sweep.run_identity``
    reads), which the trainer writes before training starts and therefore
    before any divergence can occur.
    """
    out: dict[tuple, dict] = {}
    for failure_path in sorted(exp_root.glob("*/failure.json")):
        run_dir = failure_path.parent
        cfg_path = run_dir / "config.yaml"
        if not cfg_path.exists():
            raise ValueError(f"{failure_path}: no sibling config.yaml -- cannot recover cell identity")
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
        family = cfg.get("family")
        seed = cfg.get("seed")
        stratum_system = cfg.get("stratum_system")
        stratum_regime = cfg.get("stratum_regime")
        if family is None or seed is None or stratum_system is None or stratum_regime is None:
            raise ValueError(f"{cfg_path}: missing family/seed/stratum_system/stratum_regime")
        if family not in _PAIR_BY_FAMILY:
            raise ValueError(f"{cfg_path}: unknown family {family!r}")
        backbone, structure = _PAIR_BY_FAMILY[family]
        key = (stratum_system, stratum_regime, backbone, structure, seed)
        config_hash, _, data_hash = run_dir.name.partition("-")
        out[key] = {
            "run_dir": str(run_dir.relative_to(results_root)),
            "config_hash": config_hash or None,
            "data_hash": data_hash or None,
            "representation_digest": None,  # not recoverable without re-running (never re-run here)
            "status": "divergence",
            "e_h": None, "e_h_gate": None, "m_low": None,
            "vpt_median": None, "vpt_censored": None, "per_episode_vpt": None,
        }
    return out


def build_primary_results_table(results_root: str | Path, exp_name: str) -> list[dict]:
    """One row per EXPECTED identity of ``exp_name``'s grid, never fewer.

    Columns: ``cell_id, backbone, structure, family, seed, stratum_system,
    stratum_regime, run_dir, config_hash, data_hash, representation_digest,
    status, e_h, e_h_gate, m_low, vpt_median, vpt_censored,
    per_episode_vpt``. ``status`` is one of ``{'ok', 'does_not_learn',
    'divergence', 'missing'}`` (PROTOCOL.md section 27.3's eligibility rule).
    No row is ever dropped: a cell with neither ``gate.json`` nor
    ``failure.json`` gets ``status='missing'`` and every measured field
    ``None``.
    """
    grid = _expected_grid(exp_name)
    results_root_path = Path(results_root)
    exp_root = results_root_path / exp_name

    gate_found = _scan_gate_jsons(exp_root)
    failure_found = _scan_failures(exp_root, results_root_path)

    rows = []
    for cell in grid:
        key = (cell["stratum_system"], cell["stratum_regime"], cell["backbone"], cell["structure"], cell["seed"])
        found = gate_found.get(key) or failure_found.get(key) or _MISSING_INFO
        merged = {**cell, **found}
        rows.append({k: merged[k] for k in _ROW_KEYS})
    return rows


def _group_by_family(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in rows:
        out.setdefault(row["family"], []).append(row)
    return out


def _cell_eligible(family_rows: list[dict]) -> bool:
    """PROTOCOL.md section 27.3: a cell is eligible only with 5/5 seeds
    terminal and ``'ok'``. Any ``does_not_learn``, ``divergence`` or
    ``missing`` among the five makes the whole cell ineligible."""
    return len(family_rows) == len(_REQUIRED_SEEDS) and all(r["status"] == "ok" for r in family_rows)


def aggregate_rollout_curves(rows: list[dict]) -> dict:
    """Per cell (``family`` = backbone x structure), mean +/-1 SE of ``e_h``
    over the five seeds -- ``_aggregate_curves`` (``fig1_rollout.py``)
    reused unchanged, its seed-mean/SE arithmetic is exactly what section
    27.4's Fig. 1 needs. Only ELIGIBLE cells (27.3) are aggregated; a
    non-eligible cell is returned with its five per-seed statuses and no
    mean/SE.
    """
    out: dict = {}
    for family, family_rows in _group_by_family(rows).items():
        if _cell_eligible(family_rows):
            ordered = sorted(family_rows, key=lambda r: r["seed"])
            curves = np.array([r["e_h"] for r in ordered], dtype=float)
            mean_curve, se = _aggregate_curves(curves)
            out[family] = {
                "eligible": True,
                "mean_e_h": mean_curve.tolist(),
                "se_e_h": se.tolist(),
            }
        else:
            out[family] = {
                "eligible": False,
                "statuses": {r["seed"]: r["status"] for r in family_rows},
            }
    return out


def _vpt_status(value: float | None) -> str:
    """PROTOCOL.md section 27.3: a censored value is an infinite VPT OR
    an absent one (no run to read it from) -- never replaced, never
    dropped."""
    return "censored" if value is None or not math.isfinite(value) else "crossed"


def summarize_vpt(rows: list[dict]) -> dict:
    """Per cell, the five raw ``vpt_median`` values by seed with their
    censoring status, and the censored count. Read verbatim off
    ``gate.json``'s ``measures.vpt_median`` (never recomputed from ``e_h``:
    section 27.3's "one endpoint, one set of bytes")."""
    out: dict = {}
    for family, family_rows in _group_by_family(rows).items():
        per_seed = {}
        n_censored = 0
        for row in sorted(family_rows, key=lambda r: r["seed"]):
            value = row["vpt_median"]
            status = _vpt_status(value)
            if status == "censored":
                n_censored += 1
            per_seed[row["seed"]] = {"vpt_median": value, "status": status}
        out[family] = {"per_seed": per_seed, "n_censored": n_censored}
    return out


def _check_pairing(a: dict, b: dict) -> None:
    """PROTOCOL.md section 27.3's pairing rule: same seed, same
    ``data_hash``, same ``representation_digest``. Never by file order."""
    if a["seed"] != b["seed"]:
        raise ValueError(f"pairing broken: seed {a['seed']!r} != {b['seed']!r}")
    if a["data_hash"] != b["data_hash"]:
        raise ValueError(f"pairing broken: data_hash {a['data_hash']!r} != {b['data_hash']!r}")
    if a["representation_digest"] != b["representation_digest"]:
        raise ValueError(
            f"pairing broken: representation_digest {a['representation_digest']!r} "
            f"!= {b['representation_digest']!r}"
        )


def _mean_se(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return mean, se


def paired_structure_contrasts(rows: list[dict]) -> dict:
    """Per backbone, the ``on - off`` contrast of ``e_h_gate`` (lower is
    better, section 27.7), paired by seed (section 27.3's pairing rule).
    Only seeds where both arms have a readable ``e_h_gate`` (status ``'ok'``
    or ``'does_not_learn'``) contribute a contrast; a seed missing on either
    side is skipped, never faked."""
    by_key = {(r["backbone"], r["structure"], r["seed"]): r for r in rows}
    out: dict = {}
    for backbone in sorted({r["backbone"] for r in rows}):
        diffs = []
        for seed in _REQUIRED_SEEDS:
            off = by_key.get((backbone, "off", seed))
            on = by_key.get((backbone, "on", seed))
            if off is None or on is None:
                continue
            if off["status"] not in ("ok", "does_not_learn") or on["status"] not in ("ok", "does_not_learn"):
                continue
            _check_pairing(off, on)
            diffs.append(on["e_h_gate"] - off["e_h_gate"])
        mean, se = _mean_se(diffs)
        out[backbone] = {"contrasts": diffs, "mean": mean, "se": se}
    return out


# The nine pre-registered relations of PROTOCOL.md section 27.3, "X > Y"
# meaning X is better than Y (lower e_h_gate). Groups 1/2 compare cells
# across DIFFERENT backbones, which share no pairing key of their own beyond
# seed -- 27.2/27.3 name only ON-OFF pairing explicitly, so this module
# extends the same seed/data_hash/representation_digest pairing rule to the
# cross-backbone relations too, rather than inventing a second, looser rule:
# on this single-stratum grid every cell trains on the SAME dataset build,
# so data_hash/representation_digest agreeing across backbones is exactly
# what the shared stratum promises, and disagreement here is a real defect
# worth raising loudly rather than silently averaging past it.
_RELATIONS_STRUCTURED = (("ssm_k", "transformer_k"), ("transformer_k", "mlp_k"), ("ssm_k", "mlp_k"))
_RELATIONS_UNSTRUCTURED = (("min_ssm", "transformer"), ("transformer", "mlp"), ("min_ssm", "mlp"))
_RELATIONS_STRUCTURE_EFFECT = (("mlp_k", "mlp"), ("transformer_k", "transformer"), ("ssm_k", "min_ssm"))
_NOT_TESTED = ("min_ssm", "mlp_k")


def _evaluate_relation(x: str, y: str, by_key: dict, eligible: dict) -> dict:
    if not eligible.get(x, False) or not eligible.get(y, False):
        return {"verdict": "indeterminate", "diffs": None, "mean": None, "se": None}
    x_backbone, x_structure = _PAIR_BY_FAMILY[x]
    y_backbone, y_structure = _PAIR_BY_FAMILY[y]
    diffs = []
    for seed in _REQUIRED_SEEDS:
        rx = by_key[(x_backbone, x_structure, seed)]
        ry = by_key[(y_backbone, y_structure, seed)]
        _check_pairing(rx, ry)
        diffs.append(rx["e_h_gate"] - ry["e_h_gate"])  # X - Y; predicted X>Y means this is < 0
    mean, se = _mean_se(diffs)
    lo, hi = mean - se, mean + se
    # e_h_gate is lower-is-better (section 27.7): predicted "X > Y" tests
    # e_h_gate(X) < e_h_gate(Y), i.e. mean(X - Y) < 0.
    if mean < 0 and hi < 0:
        verdict = "supported"
    elif mean > 0 and lo > 0:
        verdict = "contradicted"
    else:
        verdict = "indeterminate"
    return {"verdict": verdict, "diffs": diffs, "mean": mean, "se": se}


def evaluate_preregistered_orders(rows: list[dict]) -> dict:
    """The nine relations of PROTOCOL.md section 27.3, plus the untested
    pair. Verdicts are ``{'supported', 'contradicted', 'indeterminate',
    'not_tested'}``; a relation naming an ineligible cell is always
    ``'indeterminate'``, never a win."""
    by_family = _group_by_family(rows)
    eligible = {family: _cell_eligible(family_rows) for family, family_rows in by_family.items()}
    by_key = {(r["backbone"], r["structure"], r["seed"]): r for r in rows}

    out: dict = {}
    for x, y in (*_RELATIONS_STRUCTURED, *_RELATIONS_UNSTRUCTURED, *_RELATIONS_STRUCTURE_EFFECT):
        out[f"{x}>{y}"] = _evaluate_relation(x, y, by_key, eligible)
    out[f"{_NOT_TESTED[0]}>{_NOT_TESTED[1]}"] = {"verdict": "not_tested", "diffs": None, "mean": None, "se": None}
    return out
