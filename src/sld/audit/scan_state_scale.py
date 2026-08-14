"""Persist ``sigma_train`` as a torch-free JSON artifact.

Why this module exists. ``PROTOCOL.md`` section 27.4 requires every primary
figure's caption to name ``sigma_train``, the per-coordinate train-split scale
that normalizes ``e_h``. That vector is measured once per stratum by
``sld.encoders.pretrain`` and stored inside the frozen representation's ``.pt``
artifact -- which only ``torch`` can open. ``REGLES.md`` section 2 keeps
``sld/viz/`` torch-free, so a figure cannot read it.

This is the same shape as ``scan_action_jacobian`` / ``figF``: the torch-ful
pass persists a JSON artifact, the torch-free figure reads it. No quantity is
recomputed here and none is invented -- the vector is copied out of the
artifact that the runs themselves were normalized by, so a caption that cites
it cites the number the measurement actually used.

``e_recon`` and ``e_1_pers`` are copied alongside for the same reason: they are
already in every ``gate.json``, and writing them here too lets a reader check
the two sources against each other rather than trust one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCHEMA = "state_scale/1"
FILENAME = "state_scale.json"


def scan_state_scale(representations_dir: Path) -> dict:
    """Read every frozen-representation artifact in ``representations_dir`` and
    return the JSON-ready payload.

    Raises ``FileNotFoundError`` when the directory holds no ``.pt`` artifact:
    an empty result would otherwise reach a caption as a silent blank.
    """
    import numpy as np
    import torch

    artifacts = sorted(Path(representations_dir).glob("*.pt"))
    if not artifacts:
        raise FileNotFoundError(
            f"no frozen-representation artifact (*.pt) under {representations_dir} -- "
            "run the grid first; this scan reads what the runs were normalized by, "
            "it does not recompute it"
        )

    strata: dict[str, dict] = {}
    for path in artifacts:
        # weights_only=False is required: the artifact carries a numpy array
        # and a manifest dict, not just tensors. Same convention as
        # experiments/loading.py, and the file is one this repository's own
        # pretraining step wrote into its own results root -- never an input.
        blob = torch.load(path, weights_only=False, map_location="cpu")
        sigma = np.asarray(blob["sigma_train"], dtype=float)
        if sigma.ndim != 1 or not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
            raise ValueError(
                f"{path}: sigma_train is not a finite strictly-positive 1-D vector "
                f"(shape {sigma.shape}) -- a caption must not quote it"
            )
        manifest = blob["manifest"]
        strata[path.stem] = {
            "system": manifest["system"],
            "regime": manifest["regime"],
            "digest": manifest["digest"],
            "sigma_train": [float(v) for v in sigma],
            "e_recon": float(blob["e_recon"]),
            "e_1_pers": float(manifest["e_1_pers"]),
        }

    return {"schema": SCHEMA, "strata": strata}


def load_state_scale(path: Path) -> dict:
    """Torch-free reader, for ``sld/viz/`` (REGLES section 2)."""
    payload = json.loads(Path(path).read_text())
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"{path}: expected schema {SCHEMA!r}, got {payload.get('schema')!r}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representations", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        payload = scan_state_scale(args.representations)
    except (FileNotFoundError, ValueError) as exc:
        print(f"scan_state_scale: {exc}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(args.out)
    for name, block in payload["strata"].items():
        sigma = ", ".join(f"{v:.4f}" for v in block["sigma_train"])
        print(f"{name}: sigma_train = [{sigma}]  e_recon = {block['e_recon']:.3e}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
