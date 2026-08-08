"""Reconstruct trained models and datasets from ``results/`` artifacts.

This is the only place ``viz/`` is allowed to touch predictor/encoder
objects (REGLES section 2: ``viz/`` reads ``results/``, never live training
objects) -- everything here rebuilds architectures FROM a run's ``config``
and loads its saved weights, it never receives a model handed to it live.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import torch
import yaml

from sld.data.generate import build_dataset
from sld.data.regimes import ActionConditionedRegime, OURegime
from sld.encoders.frame_mlp import FrameMLPEncoder
from sld.predictors.registry import get_predictor
from sld.systems.coupled_oscillators import CoupledOscillators
from sld.systems.damped_pendulum import DampedPendulum

# single source of truth for which config keys are forwarded to which
# predictor family's constructor (mirrors Trainer.fit's construction block,
# trainer.py) -- reused rather than re-declared so the two never drift apart
from sld.training.trainer import _FAMILY_CONFIG_KEYS

# local minimal registry for build_dataset_from_expconfig: v0 pendulum scope
# only (matches src/sld/experiments/configs/pendulum_core.yaml's two
# datasets), grows alongside the configs that need it, not ahead of them.
# CoupledOscillators (PROTOCOL.md section 19, D3) added Wave D3 so viz can
# rebuild a D3 run's dataset the same way sld-factorial does.
_SYSTEM_REGISTRY: dict[str, type] = {
    "DampedPendulum": DampedPendulum,
    "CoupledOscillators": CoupledOscillators,
}
_REGIME_REGISTRY: dict[str, type] = {
    "OURegime": OURegime,
    "ActionConditionedRegime": ActionConditionedRegime,
}

# "structure" is the canonical name (phase B1: backbone x structure replaces
# family x constraint) -- but index.csv files written before this rename
# (results/, results_amended/, reports/historical/mu_j/traces/) carry
# "constraint" instead, and those traces are frozen, never rewritten.
_INDEX_COLUMNS = ("run_dir", "regime", "family", "structure", "seed", "status", "final_val_loss")
# ponytail: legacy column name tolerated on read only; drop once no pre-B4
# index.csv (frozen trace) is read anymore.
_LEGACY_STRUCTURE_COLUMN = "constraint"


def load_run(run_dir: str | Path) -> dict:
    """Reconstruct ``{"encoder", "predictor", "config"}`` from a run directory.

    Reads ``ckpt.pt`` (state dicts + the resolved config dict ``Trainer.fit``
    wrote), rebuilds the encoder/predictor architectures FROM that config,
    loads the weights, and sets both modules to eval mode.

    Construction happens under a temporarily forced ``torch.float32``
    default dtype, restored in a ``finally`` -- the exact policy
    ``Trainer.fit`` uses (trainer.py's construction block) -- so init-time
    shapes/dtypes match the checkpoint regardless of the caller's ambient
    default dtype.

    Raises:
        FileNotFoundError: no ``ckpt.pt`` in ``run_dir``.
    """
    run_dir = Path(run_dir)
    ckpt_path = run_dir / "ckpt.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"no ckpt.pt in run dir {run_dir}")
    # weights_only=False: the checkpoint carries a plain config dict alongside
    # the state dicts (trainer.py's declared ckpt.pt contract, REGLES section 6)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    ambient = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        encoder = FrameMLPEncoder(cfg["d_x"], cfg["d_z"], tuple(cfg["encoder_hidden"]))
        passthrough = {k: cfg[k] for k in _FAMILY_CONFIG_KEYS[cfg["family"]] if k in cfg}
        predictor = get_predictor(
            cfg["family"],
            cfg["d_z"],
            cfg["d_u"],
            target_params=cfg["target_params"],
            **passthrough,
        )
    finally:
        torch.set_default_dtype(ambient)

    encoder.load_state_dict(ckpt["encoder"])
    predictor.load_state_dict(ckpt["predictor"])
    encoder.eval()
    predictor.eval()
    return {"encoder": encoder, "predictor": predictor, "config": cfg}


def data_hash_of(meta: dict) -> str:
    """Fingerprint a dataset's ``meta`` block EXACTLY as
    ``Trainer.resolve_run_dir`` does (trainer.py): drop the ``paths`` key
    (infra, not part of run identity), ``json.dumps(sort_keys=True,
    default=str)``, sha1, first 8 hex chars. Reused here (not
    reimplemented) so the run-dir naming recipe never drifts between the
    ``Trainer`` that wrote a run and the ``viz`` that reads it back.
    """
    meta = {k: v for k, v in meta.items() if k != "paths"}
    return hashlib.sha1(json.dumps(meta, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:8]


def check_run_matches_dataset(run_dir: str | Path, dataset_meta: dict) -> None:
    """Verify ``run_dir``'s name still matches the CURRENTLY rebuilt dataset.

    Run directories are named ``<config_hash>-<data_hash>``
    (``Trainer.resolve_run_dir``, trainer.py). This recomputes
    ``data_hash`` from the dataset ``build_dataset_from_expconfig`` just
    reconstructed and checks it against the run directory's trailing
    8-hex-char suffix -- catching the case where the experiment YAML
    (seed, system/regime params, ``T``, ``dt``, ...) changed AFTER the
    grid was trained: without this, old checkpoints get silently evaluated
    on new trajectories (Wave 5 review finding F6).

    Raises:
        ValueError: ``run_dir``'s name has no ``<...>-<8 hex chars>``
            suffix, or that suffix does not match ``dataset_meta``'s
            current fingerprint.
    """
    run_dir = Path(run_dir)
    prefix, sep, actual = run_dir.name.rpartition("-")
    if not sep or len(actual) != 8:
        raise ValueError(
            f"run dir name {run_dir.name!r} does not carry the expected "
            "<config_hash>-<data_hash> (8 hex char) suffix; cannot verify "
            "it against the current dataset spec"
        )
    expected = data_hash_of(dataset_meta)
    if actual != expected:
        raise ValueError(
            f"run {run_dir} was trained on a dataset with data_hash "
            f"{actual!r}, but the CURRENT experiment YAML rebuilds a "
            f"dataset with data_hash {expected!r} -- the YAML (seed, "
            "system/regime params, T, dt, ...) has changed since this run "
            "was trained; regenerate the grid, or point exp_yaml back at "
            "the config that produced it"
        )


def read_index(results_root: str | Path, exp_name: str) -> list[dict]:
    """Read ``<results_root>/<exp_name>/index.csv``.

    Required, never a silent scan fallback over ``results/`` -- a missing
    index is a clear "run the sweep first" error, not a raw traceback or a
    reconstructed-from-directory-listing guess.

    ``structure`` is the canonical column name (phase B1); an index.csv
    written under the pre-B4 name ``constraint`` is tolerated as a legacy
    alias (frozen traces under ``results/``, ``results_amended/``, and
    ``reports/historical/mu_j/traces/`` are never rewritten) -- every
    returned row carries a ``structure`` key regardless of which name the
    file on disk used.

    Raises:
        FileNotFoundError: ``index.csv`` does not exist.
        ValueError: ``index.csv`` is missing one of the required columns
            (``run_dir, regime, family, structure, seed, status,
            final_val_loss`` -- the ``sld-factorial`` contract -- accepting
            the legacy ``constraint`` name in place of ``structure``).
    """
    index_path = Path(results_root) / exp_name / "index.csv"
    if not index_path.exists():
        raise FileNotFoundError(
            f"{index_path} does not exist -- run the experiment sweep first "
            "(see the Makefile / RESUME_SESSION.md for the current wave); "
            "viz never falls back to scanning results/ directly"
        )
    with open(index_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        legacy_structure = "structure" not in fieldnames and _LEGACY_STRUCTURE_COLUMN in fieldnames
        required = [c for c in _INDEX_COLUMNS if c != "structure" or not legacy_structure]
        missing = [c for c in required if c not in fieldnames]
        if missing:
            raise ValueError(f"{index_path} is missing required columns {missing}")
        rows = []
        for row in reader:
            row = dict(row)
            if legacy_structure:
                row["structure"] = row[_LEGACY_STRUCTURE_COLUMN]
            row["seed"] = int(row["seed"])
            try:
                row["final_val_loss"] = float(row["final_val_loss"])
            except (TypeError, ValueError):
                row["final_val_loss"] = None
            rows.append(row)
    return rows


def build_dataset_from_expconfig(exp_yaml_path: str | Path, regime_name: str) -> dict:
    """Deterministically rebuild the ``regime_name`` dataset from an
    experiment YAML.

    Schema (matches ``src/sld/experiments/configs/pendulum_core.yaml``):
    ``datasets.<regime_name>.{system: {name, params}, regime: {name,
    params}, obs, n_episodes, T, dt, seed}``.

    Minimal local registry (v0 pendulum scope): ``DampedPendulum``,
    ``OURegime``, ``ActionConditionedRegime``.

    Raises:
        FileNotFoundError: ``exp_yaml_path`` does not exist.
        KeyError: no ``datasets.<regime_name>`` block in the YAML.
        ValueError: an unknown system/regime class name.
        NotImplementedError: a non-null ``obs`` -- the v0 registry above
            carries no observation-mixing classes (matches the current
            pendulum_core.yaml, where both regimes declare ``obs: null``).
    """
    exp_yaml_path = Path(exp_yaml_path)
    if not exp_yaml_path.exists():
        raise FileNotFoundError(f"experiment config not found: {exp_yaml_path}")
    with open(exp_yaml_path) as f:
        exp_config = yaml.safe_load(f)

    try:
        spec = exp_config["datasets"][regime_name]
    except KeyError as e:
        raise KeyError(f"{exp_yaml_path} has no datasets.{regime_name!r} block") from e

    if spec.get("obs") is not None:
        raise NotImplementedError(
            f"datasets.{regime_name}.obs is non-null; build_dataset_from_expconfig's "
            "v0 registry (DampedPendulum, OURegime, ActionConditionedRegime) carries "
            "no observation-mixing classes"
        )

    def _build(kind: str, registry: dict[str, type]):
        block = spec[kind]
        name = block["name"]
        if name not in registry:
            raise ValueError(f"unknown {kind} {name!r}; known: {sorted(registry)}")
        return registry[name](**block.get("params", {}))

    system = _build("system", _SYSTEM_REGISTRY)
    regime = _build("regime", _REGIME_REGISTRY)

    return build_dataset(
        system,
        regime,
        None,
        spec["n_episodes"],
        spec["T"],
        spec["dt"],
        spec["seed"],
    )
