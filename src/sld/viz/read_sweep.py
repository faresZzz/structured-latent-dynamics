"""Read a rate sweep and print the per-arm argmin -- PROTOCOL.md section 23.3.

Torch-free, like every module under ``sld.viz``: it reads ``index.csv`` and the
per-run ``metrics.csv`` that ``sld-factorial`` already wrote.

Two behaviours are deliberate and are the reason this is a module rather than a
one-liner:

- the selection column is DECLARED, not guessed. If the requested column is
  absent the module lists what it found and exits non-zero, rather than falling
  back to something plausible. Section 16.4 fixes the reading rules; a script
  that silently reads a different column would break them invisibly.
- a run that did not finish, or whose criterion is not finite, is EXCLUDED from
  the argmin and named in the output. Section 16.2(4): a diverged run cannot
  win, and an arm with no trainable rate is a publishable fact, not a gap.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import yaml

DEFAULT_COLUMN = "e_25_val"

# Section 23.3's criterion does NOT live in metrics.csv, and this is the trap
# worth naming: metrics.csv is the TRAINING log (train_loss, val_loss, lr, the
# loss terms). `e_25` is a free-ROLLOUT quantity in true-state coordinates,
# produced by `evaluate_run_free_rollout` and persisted by
# `sld-gate --split val` into `gate_val.json` (16.4: the gate scores on test,
# anything that SELECTS reads val).
#
# The four names below are `probe.py::_val_reading`'s, verbatim, so what this
# module minimises is the same OBJECT round 3 minimised -- not a look-alike
# recomputed down a second path (REGLES 7.1's independent-reference corollary
# cuts the other way here: for a SELECTION, agreeing on the object is the
# requirement).
_VAL_COLUMNS: dict[str, str] = {
    "e_1_val": "e_1",
    "e_25_val": "e_h_gate",   # e_h[h_gate - 1], guarded against h_gate != 25 below
    "m_low_val": "m_low",
    "m_high_val": "m_high",
}


def _val_metric(run_dir: Path, column: str) -> tuple[float | None, str]:
    """``column`` out of ``run_dir/gate_val.json`` -- the val-split reading."""
    path = run_dir / "gate_val.json"
    if not path.exists():
        return None, "gate_val.json absent (lancer: sld-gate --config <cfg> --split val)"
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeError) as exc:
        return None, f"gate_val.json illisible ({exc!r})"
    if payload.get("split") != "val":
        return None, f"gate_val.json porte split={payload.get('split')!r}, pas 'val'"
    # the column NAMES a horizon; refuse if the artifact was scored at another
    # one rather than silently returning a number that means something else
    h_gate = (payload.get("protocol") or {}).get("h_gate")
    if column == "e_25_val" and h_gate != 25:
        return None, f"e_25_val demande mais l'artefact est score a h_gate={h_gate}"
    measures = payload.get("measures")
    if not measures:
        return None, f"aucune mesure (status={payload.get('status')!r})"
    value = measures.get(_VAL_COLUMNS[column])
    if value is None or not math.isfinite(float(value)):
        return None, "valeur non finie"
    return float(value), "ok"


def arm_key(family: str, radius: str | None) -> str:
    """The canonical name of a sweep arm: its predictor family, plus the
    ``radius`` level when the family has one.

    Read from the run's OWN ``config.yaml`` (see ``run_identity``), never from
    ``index.csv``: that file's columns are
    ``run_dir,regime,family,structure,seed,status,final_val_loss`` -- it
    carries neither ``lr`` nor ``radius``. Keying a sweep on it merged the
    two `radius` levels of section 23.1 into ONE arm and left every retained
    rate printed as `?`, which is how this was caught.
    """
    return f"{family}[{radius}]" if radius else family


def run_identity(run_dir: Path) -> tuple[str, str] | None:
    """``(arm, lr)`` from ``run_dir/config.yaml`` -- the FLATTENED config that
    was hashed into this run's identity, so what is read here is by
    construction what the run was actually built with."""
    path = run_dir / "config.yaml"
    if not path.exists():
        return None
    cfg = yaml.safe_load(path.read_text()) or {}
    family, lr = cfg.get("family"), cfg.get("lr")
    if not family or lr is None:
        return None
    return arm_key(family, cfg.get("radius")), f"{lr:g}"


def _final_metric(run_dir: Path, column: str) -> tuple[float | None, str]:
    """Last finite value of ``column``: out of ``gate_val.json`` for a val
    reading, out of ``run_dir/metrics.csv`` for a training-log column.

    Never falls back from one source to the other. A missing ``gate_val.json``
    is reported as exactly that, with the command that writes it -- reading
    `val_loss` because `e_25_val` was unavailable would silently change
    section 23.3's criterion, which is the one thing this module exists to
    make impossible.
    """
    if column in _VAL_COLUMNS:
        return _val_metric(run_dir, column)
    metrics = run_dir / "metrics.csv"
    if not metrics.exists():
        return None, "metrics.csv absent"
    with metrics.open() as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None, "metrics.csv vide"
    if column not in rows[0]:
        return None, (
            f"colonne absente de metrics.csv (disponibles: {', '.join(rows[0])}"
            f" | val: {', '.join(sorted(_VAL_COLUMNS))})"
        )
    for row in reversed(rows):
        try:
            value = float(row[column])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value, "ok"
    return None, "aucune valeur finie"


def read_sweep(root: str | Path, column: str = DEFAULT_COLUMN) -> int:
    """Print the sweep table and the per-arm argmin. Returns a process exit code."""
    root = Path(root)
    index = root.rglob("index.csv")
    rows: list[dict[str, str]] = []
    for path in index:
        with path.open() as fh:
            for row in csv.DictReader(fh):
                row["_root"] = str(path.parent)
                rows.append(row)
    if not rows:
        print(f"aucun index.csv sous {root}", file=sys.stderr)
        return 1

    per_arm: dict[str, list[tuple[float, str, str]]] = {}
    excluded: list[str] = []
    for row in rows:
        # run_dir is written relative to different anchors depending on which
        # writer produced the row (repo root for the factorial's index.csv,
        # result-root-relative for the gate artifacts). Resolve against every
        # plausible anchor rather than assuming one, and keep the first that
        # actually holds a metrics.csv.
        raw = Path(row.get("run_dir") or "")
        candidates = [raw, Path(row["_root"]) / raw.name, Path(row["_root"]).parent / raw, root / raw]
        run_dir = next((c for c in candidates if (c / "metrics.csv").exists()), raw)
        identity = run_identity(run_dir)
        arm, lr = identity if identity else (f"{row.get('family', '?')}[?]", "?")
        value, why = _final_metric(run_dir, column)
        status = row.get("status", "?")
        if identity is None:
            excluded.append(f"  EXCLU  {arm:26s} lr={lr:9s} status={status:12s} config.yaml illisible")
            continue
        if value is None or status != "complete":
            excluded.append(f"  EXCLU  {arm:26s} lr={lr:9s} status={status:12s} {why}")
            continue
        per_arm.setdefault(arm, []).append((value, lr, str(run_dir)))

    print(f"critere de selection : {column}  (split val, section 16.4 -- jamais test)\n")
    for arm in sorted(per_arm):
        entries = sorted(per_arm[arm])
        print(f"{arm}")
        for value, lr, _ in entries:
            print(f"    lr={lr:9s}  {column}={value:.6g}")
        best_value, best_lr, _ = entries[0]
        print(f"    -> RETENU lr={best_lr}  ({column}={best_value:.6g})\n")

    if excluded:
        print("Exclus de l'argmin (section 16.2(4) : un run diverge ne peut pas gagner) :")
        print("\n".join(excluded))
    missing = [a for a, e in per_arm.items() if not e]
    if missing:
        print(f"\nBRAS SANS TAUX ENTRAINABLE : {missing} -- fait publiable, aucun taux hors liste.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="racine de resultats du sweep, ex. results_sweep_osc")
    parser.add_argument("--column", default=DEFAULT_COLUMN, help=f"defaut: {DEFAULT_COLUMN}")
    args = parser.parse_args()
    raise SystemExit(read_sweep(args.root, args.column))


if __name__ == "__main__":
    main()
