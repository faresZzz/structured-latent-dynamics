"""Score a results root's action-Jacobian conditioning into a per-run artifact.

The torch half of figure F. `sld.viz` must stay torch-free (REGLES section 2),
so the conditioning is computed HERE, once, against each run's committed
checkpoint, and persisted as ``action_jacobian.json`` beside the run's other
artifacts. The figure then reads JSON like every other figure module does.

Every number this writes has already been cross-checked against centred finite
differences to better than 1e-8 relative -- `conditioning` runs that check
itself and raises rather than return an unchecked Jacobian
(`sld.audit.action_jacobian`).

    uv run python -m sld.audit.scan_action_jacobian results_osc_n
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sld.audit.action_jacobian import conditioning
from sld.experiments.loading import load_run

SCHEMA = "action_jacobian/1"
FILENAME = "action_jacobian.json"
HORIZON = 25


def score_run(run_dir: Path, horizon: int = HORIZON, seed: int = 0) -> dict:
    """Conditioning of ``d x_H / d u`` for one run, in float64 on CPU.

    The probe state and action sequence are drawn from a FIXED seed, identical
    for every arm: the quantity compared across arms must differ by the model
    and by nothing else.
    """
    run = load_run(run_dir)
    predictor = run["predictor"].double().cpu().eval()
    cfg = run["config"]
    d_h = 2 * cfg["r"] if "r" in cfg else cfg["d_z"]
    g = torch.Generator().manual_seed(seed)
    h0 = torch.randn(d_h, generator=g, dtype=torch.float64)
    u = torch.randn(horizon, cfg["d_u"], generator=g, dtype=torch.float64)

    result = conditioning(predictor, h0, u)
    return {
        "schema": SCHEMA,
        "horizon": horizon,
        "probe_seed": seed,
        "cond": result.cond,
        "singular_values": result.singular_values.tolist(),
        "fd_max_rel_error": result.fd_max_rel_error,
        "config_hash": cfg.get("config_hash", run_dir.name.split("-")[0]),
    }


def scan(root: str | Path, horizon: int = HORIZON) -> int:
    root = Path(root)
    done = skipped = 0
    for cfg_path in sorted(root.rglob("config.yaml")):
        run_dir = cfg_path.parent
        if not (run_dir / "run_complete.json").exists():
            skipped += 1
            continue
        payload = score_run(run_dir, horizon)
        tmp = run_dir / (FILENAME + ".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(run_dir / FILENAME)
        done += 1
        print(f"{run_dir.name}  cond={payload['cond']:.4g}  fd_err={payload['fd_max_rel_error']:.2e}")
    print(f"\n{done} runs scored, {skipped} skipped (no completion sentinel -- never fabricated)")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root")
    ap.add_argument("--horizon", type=int, default=HORIZON)
    raise SystemExit(scan(**vars(ap.parse_args())))


if __name__ == "__main__":
    main()
