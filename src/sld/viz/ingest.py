"""Read verdict artifacts off disk into tidy frames -- the single entry point
every figure module uses.

Why this module exists, and why it imports no torch: REGLES section 2 requires
that figures regenerate from ``results*/`` and never from live training
objects. A figure module that could import a predictor could, one refactor
later, retrain one. This module reads JSON only, so that failure mode is
structurally impossible rather than merely discouraged.

Two artifact schemas are read, both closed and both versioned:

- ``gate/1``   -- one per phase-D run, written by ``sld-gate`` (PROTOCOL.md
  section 19.4). Carries the per-horizon curves ``e_h``/``persistence_e_h``/
  ``linear_e_h`` and the two gate margins ``m_low``/``m_high``.
- ``d3_gold/1`` -- one per D3 control run, written by ``sld-d3-gate``. Carries
  the three absolute exactness bounds plus the relative gain.

Both loaders REFUSE an unknown ``schema`` value rather than guessing at the
fields: a silently mis-parsed artifact is exactly the class of error that
section 19.8's wiring defect belonged to.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

GATE_SCHEMA = "gate/1"
D3_GOLD_SCHEMA = "d3_gold/1"


def _read_json_tree(root: str | Path, filename: str) -> list[tuple[Path, dict]]:
    """Every ``filename`` under ``root``, sorted by path for a stable frame order."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"result root does not exist: {root}")
    found = sorted(root.rglob(filename))
    if not found:
        raise FileNotFoundError(f"no {filename} under {root}")
    return [(p, json.loads(p.read_text())) for p in found]


def load_gate(root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load every ``gate.json`` under ``root``.

    Returns:
        ``(scalars, curves)``. ``scalars`` has one row per run:
        ``cell_id, backbone, structure, family, seed, stratum, split, e_1,
        e_h_gate, m_low, m_high, lower_bound, upper_bound, near_miss, status,
        tau_1, e_1_pers, e_recon, gamma, gamma_1, h_gate, run_dir``.
        ``curves`` is long, one row per ``(run, h)``:
        ``cell_id, backbone, structure, family, seed, stratum, h, e_h,
        persistence_e_h, linear_e_h``.

    Raises:
        ValueError: any artifact whose ``schema`` is not ``gate/1``.
    """
    scalars: list[dict] = []
    curves: list[dict] = []
    for path, d in _read_json_tree(root, "gate.json"):
        if d.get("schema") != GATE_SCHEMA:
            raise ValueError(f"{path}: expected schema {GATE_SCHEMA!r}, got {d.get('schema')!r}")
        ident, arm = d["identity"], d["identity"]["arm"]
        proto, meas = d["protocol"], d["measures"]
        key = {
            "cell_id": ident["cell_id"],
            "backbone": arm["backbone"],
            "structure": arm["structure"],
            "family": arm["family"],
            "seed": ident["seed"],
            "stratum": _stratum_label(ident["stratum_regime"]),
        }
        scalars.append(
            {
                **key,
                "split": d["split"],
                "e_1": meas["e_1"],
                "e_h_gate": meas["e_h_gate"],
                "m_low": meas["m_low"],
                "m_high": meas["m_high"],
                "lower_bound": meas["lower_bound"],
                "upper_bound": meas["upper_bound"],
                "near_miss": meas["near_miss"],
                "status": d["status"],
                "tau_1": proto["tau_1"],
                "e_1_pers": proto["e_1_pers"],
                "e_recon": proto["e_recon"],
                "gamma": proto["gamma"],
                "gamma_1": proto["gamma_1"],
                "h_gate": proto["h_gate"],
                "run_dir": ident["run_dir"],
            }
        )
        for i, (e, pers, lin) in enumerate(
            zip(meas["e_h"], meas["persistence_e_h"], meas["linear_e_h"], strict=True), start=1
        ):
            curves.append({**key, "h": i, "e_h": e, "persistence_e_h": pers, "linear_e_h": lin})
    return pd.DataFrame(scalars), pd.DataFrame(curves)


def load_d3_gold(root: str | Path) -> pd.DataFrame:
    """Load every ``d3_gold.json`` under ``root``, one row per run.

    Columns: ``cell_id, backbone, structure, family, seed, state_h100,
    spectral, action, m_low, gain_passes, threshold_state, threshold_spectral,
    threshold_action, status, run_dir`` plus the per-probe dispersion lists
    ``spectral_values``/``action_values`` (kept as objects: the panels show the
    spread, and flattening it here would hide how few probes there are).

    Raises:
        ValueError: any artifact whose ``schema`` is not ``d3_gold/1``.
    """
    rows: list[dict] = []
    for path, d in _read_json_tree(root, "d3_gold.json"):
        if d.get("schema") != D3_GOLD_SCHEMA:
            raise ValueError(f"{path}: expected schema {D3_GOLD_SCHEMA!r}, got {d.get('schema')!r}")
        ident, arm, b = d["identity"], d["identity"]["arm"], d["bounds"]
        rows.append(
            {
                "cell_id": ident["cell_id"],
                "backbone": arm["backbone"],
                "structure": arm["structure"],
                "family": arm["family"],
                "seed": ident["seed"],
                "state_h100": b["state_h100"]["value"],
                "spectral": b["spectral"]["value"],
                "action": b["action"]["value"],
                "m_low": b["gain"]["m_low"],
                "gain_passes": b["gain"]["passes"],
                "threshold_state": b["state_h100"]["threshold"],
                "threshold_spectral": b["spectral"]["threshold"],
                "threshold_action": b["action"]["threshold"],
                "spectral_values": d["dispersion"]["spectral_values"],
                "action_values": d["dispersion"]["action_values"],
                "n_probes": d["dispersion"]["n_probes"],
                "status": d["status"],
                "run_dir": ident["run_dir"],
            }
        )
    return pd.DataFrame(rows)


def _stratum_label(regime: str) -> str:
    """``OURegime`` -> ``OU``; ``ActionConditionedRegime`` -> ``AC``.

    The published tables (PROTOCOL.md section 21.0) use the short labels, and a
    figure that disagreed with its own protocol table on a row name would be
    unreadable next to it.
    """
    return {"OURegime": "OU", "ActionConditionedRegime": "AC"}.get(regime, regime)


def arm_label(backbone: str, structure: str) -> str:
    """``(mlp, off) -> 'mlp'``, ``(mlp, on) -> 'mlp_k'`` -- the published names."""
    return backbone if structure == "off" else f"{backbone}_k"
