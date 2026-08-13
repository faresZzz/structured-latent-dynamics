"""Read the §23.3 sweep, pick each arm's own rate, and write it into the two
campaign configs -- textually, so every comment in those files survives.

Why a script and not a hand edit: §23.3 declares the criterion (argmin of that
arm's own e_25 on the VAL split) and the exclusions (a run that did not finish,
or whose criterion is not finite, cannot win). A hand edit can silently pick the
wrong column or forget an exclusion; this prints its decision, refuses when the
decision is not well posed, and leaves the configs untouched in that case.

Refuses, with exit code 1 and nothing written, when:
  - the declared column is absent from metrics.csv (it lists what it found);
  - an arm has no admissible run -> that arm has no trainable rate at this
    regime, which §16.2(4) says is published, never patched around;
  - an arm's argmin is tied at the criterion's printed precision.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

# ONE implementation of the criterion lookup, in the module that documents it.
# A second copy here is a second place for the column to drift.
from sld.viz.read_sweep import _final_metric as _criterion
from sld.experiments.factorial import _FAMILY_BY_PAIR
from sld.viz.read_sweep import arm_key, run_identity

COLUMN = "e_25_val"
CONFIGS = ("osc_sample_size.yaml", "osc_misspecification.yaml")


def select(root: Path, column: str) -> dict[str, str]:
    candidates: dict[str, list[tuple[float, str]]] = {}
    seen: set[str] = set()
    for index in root.rglob("index.csv"):
        for row in csv.DictReader(index.open()):
            raw = Path(row.get("run_dir") or "")
            anchors = [raw, index.parent / raw.name, index.parent.parent / raw, root / raw]
            run_dir = next((a for a in anchors if (a / "config.yaml").exists()), raw)
            identity = run_identity(run_dir)
            if identity is None:
                print(f"  exclu   {row.get('run_dir','?')} -- config.yaml illisible")
                continue
            key, lr = identity
            seen.add(key)
            value, why = _criterion(run_dir, column)
            if value is None or row.get("status") != "complete" or not lr:
                print(f"  exclu   {key:28s} lr={lr or '?':9s} status={row.get('status','?')} -- {why}")
                continue
            candidates.setdefault(key, []).append((value, lr))

    chosen: dict[str, str] = {}
    for key in sorted(seen):
        entries = sorted(candidates.get(key, []))
        if not entries:
            print(f"\nREFUS: le bras {key} n'a AUCUN taux entrainable a ce regime.")
            print("       §16.2(4): le fait est publie, aucun taux hors liste n'est essaye.")
            raise SystemExit(1)
        print(f"\n{key}")
        for value, lr in entries:
            print(f"    lr={lr:9s}  {column}={value:.6g}")
        if len(entries) > 1 and abs(entries[0][0] - entries[1][0]) < 1e-9:
            print(f"REFUS: argmin ex aequo pour {key} ({entries[0][0]:.6g}). Decision humaine requise.")
            raise SystemExit(1)
        chosen[key] = entries[0][1]
        print(f"    -> RETENU lr={entries[0][1]}")
    return chosen


def patch(config: Path, chosen: dict[str, str], dry_run: bool) -> int:
    text = config.read_text()
    head, sep, arms_block = text.partition("\narms:\n")
    if not sep:
        print(f"{config}: pas de bloc 'arms:' -- non modifie", file=sys.stderr)
        return 1
    tail_split = re.split(r"\n(?=seeds:)", arms_block, maxsplit=1)
    arms_text, tail = tail_split[0], ("\n" + tail_split[1] if len(tail_split) > 1 else "")
    chunks = re.split(r"\n(?=  - backbone:)", arms_text)
    out, changed = [], 0
    for chunk in chunks:
        m_b = re.search(r"backbone:\s*(\S+)", chunk)
        if not m_b:
            out.append(chunk)
            continue
        backbone = m_b.group(1)
        structure = (re.search(r"structure:\s*'?(\w+)'?", chunk) or [None, "?"])[1]
        family = _FAMILY_BY_PAIR.get((backbone, structure))
        if family is None:
            print(f"  {config.name}: paire ({backbone}, {structure}) inconnue -- laisse tel quel")
            out.append(chunk)
            continue
        m_r = re.search(r"radius:\s*(\w+)", chunk)
        key = arm_key(family, m_r.group(1) if m_r else None)
        if key in chosen:
            new_chunk, n = re.subn(r"lr:\s*[0-9.eE+-]+", f"lr: {chosen[key]}", chunk)
            if n:
                changed += 1
                print(f"  {config.name}: {key:28s} -> lr: {chosen[key]}")
            out.append(new_chunk)
        else:
            print(f"  {config.name}: {key:28s} ABSENT du sweep -- laisse tel quel")
            out.append(chunk)
    new_text = head + sep + "\n".join(out) + tail
    if dry_run:
        print(f"  (dry-run) {config.name}: {changed} bras auraient ete modifies")
        return 0
    config.write_text(new_text)
    print(f"  {config.name}: {changed} bras ecrits")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", nargs="?", default="results_sweep_osc")
    ap.add_argument("--column", default=COLUMN)
    ap.add_argument("--configs-dir", default="src/sld/experiments/configs")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    if not root.exists():
        print(f"racine de sweep absente: {root}", file=sys.stderr)
        raise SystemExit(1)
    print(f"critere: {a.column} (split val, §16.4 -- jamais test)\n")
    chosen = select(root, a.column)
    print("\n--- report dans les configs de campagne ---")
    for name in CONFIGS:
        patch(Path(a.configs_dir) / name, chosen, a.dry_run)
    print("\nSelection terminee. §23.3: UNE selection, aucune seconde passe.")


if __name__ == "__main__":
    main()
