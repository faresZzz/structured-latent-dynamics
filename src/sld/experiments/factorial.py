"""Factorial runner: dataset x arm x seed grid -> ``Trainer`` runs (REGLES section 6).

A YAML config declares a dataset axis (regime), an arm axis (backbone x
structure, cells named explicitly -- no cross-product magic), and a seed
list. ``structure`` is an ARCHITECTURAL property (the Koopman-structured
variant of a backbone), not a numeric penalty -- it selects a predictor
family, it does not tune ``mu_j``. ``expand`` turns that into a
deterministic list of run cells (datasets in YAML order, then arms, then
seeds); ``main`` builds each dataset once, resolves every cell's run
identity up front (preflight: collision check, F8), runs (or
resumes/skips) every cell through ``Trainer``, and regenerates
``results/<exp_name>/index.csv`` at the end -- covering the FULL expected
grid even if the run stopped early (F17).

v0 scope: full-state observation only (``obs: null`` in every dataset
entry); non-null ``obs`` is rejected -- the windowed-observation path
arrives with Wave 6 (see ``Trainer.fit`` docstring).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import yaml

from sld.data.generate import build_dataset
from sld.data.regimes import (
    ActionConditionedRegime,
    AnisotropicRegime,
    NarrowPolicyRegime,
    OURegime,
)
from sld.systems.damped_pendulum import DampedPendulum
from sld.training import trainer as trainer_mod
from sld.training.trainer import FAILURE_FILENAME, RUN_COMPLETE_FILENAME, Trainer

# explicit name -> class registries (v0: DampedPendulum + the four regimes,
# per the factorial's design brief -- an unknown name lists the known ones
# rather than raising an opaque KeyError)
_SYSTEMS: dict[str, type] = {"DampedPendulum": DampedPendulum}
_REGIMES: dict[str, type] = {
    "OURegime": OURegime,
    "ActionConditionedRegime": ActionConditionedRegime,
    "NarrowPolicyRegime": NarrowPolicyRegime,
    "AnisotropicRegime": AnisotropicRegime,
}

_TOP_KEYS = frozenset({"exp_name", "datasets", "base", "arms", "seeds"})
_REQUIRED_TOP_KEYS = frozenset({"exp_name", "datasets", "arms", "seeds"})
_DATASET_KEYS = frozenset({"system", "regime", "obs", "n_episodes", "T", "dt", "seed"})
_NAME_PARAMS_KEYS = frozenset({"name", "params"})
_ARM_KEYS = frozenset({"backbone", "structure", "overrides"})

_BACKBONES = ("mlp", "transformer", "ssm")
_STRUCTURES = ("off", "on")

# backbone x structure -> predictor family (see registry.FAMILIES). Only 4
# pairs exist today; the 2 missing "on" backbones (mlp_k, transformer_k)
# arrive in phase C -- looked up in _validate_arm, never silently defaulted.
_FAMILY_BY_PAIR: dict[tuple[str, str], str] = {
    ("mlp", "off"): "mlp",
    ("transformer", "off"): "transformer",
    ("ssm", "off"): "min_ssm",
    ("ssm", "on"): "koopman_ssm",
}

# keys the RUNNER injects into every cell's config (exp_name from the top
# level, seed from the seed axis, family derived from backbone/structure) --
# never settable by the user in base/overrides, the runner owns them (F7)
_RUNNER_OWNED_KEYS = frozenset({"exp_name", "seed", "family"})


def _resolve(name: str, registry: dict[str, type], kind: str) -> None:
    if name not in registry:
        raise ValueError(f"unknown {kind}: {name!r}, expected one of {tuple(registry)}")


def _validate_top(config: dict) -> None:
    unknown = sorted(set(config) - _TOP_KEYS)
    if unknown:
        raise ValueError(f"unknown top-level config keys: {unknown}")
    missing = sorted(_REQUIRED_TOP_KEYS - set(config))
    if missing:
        raise ValueError(f"config missing required top-level keys: {missing}")
    # M11 (adversarial review round 4): an empty axis is a valid schema but a
    # meaningless experiment -- 0 runs, exit 0, an empty index.csv, no
    # warning. Named per-axis so the error points at exactly which one.
    for axis in ("datasets", "arms", "seeds"):
        if not config[axis]:
            raise ValueError(f"config {axis!r} must be non-empty -- an empty axis expands to 0 runs")


def _validate_dataset_entry(name: str, ds_cfg: dict) -> None:
    unknown = sorted(set(ds_cfg) - _DATASET_KEYS)
    if unknown:
        raise ValueError(f"dataset {name!r}: unknown keys {unknown}")
    missing = sorted(_DATASET_KEYS - set(ds_cfg))
    if missing:
        raise ValueError(f"dataset {name!r}: missing keys {missing}")
    for role, registry in (("system", _SYSTEMS), ("regime", _REGIMES)):
        block = ds_cfg[role]
        block_unknown = sorted(set(block) - _NAME_PARAMS_KEYS)
        if block_unknown:
            raise ValueError(f"dataset {name!r} {role}: unknown keys {block_unknown}")
        if "name" not in block:
            raise ValueError(f"dataset {name!r} {role}: missing 'name'")
        _resolve(block["name"], registry, role)
    if ds_cfg["obs"] is not None:
        raise ValueError(
            f"dataset {name!r}: obs must be null at v0 -- non-null obs "
            "(windowed observation) arrives with Wave 6"
        )


def _validate_arm(idx: int, arm: dict) -> str:
    """Validate arm ``idx`` and return its derived predictor family.

    ``backbone`` and ``structure`` are the axis (F8): ``structure`` is a
    treatment LABEL, not free text -- an unrecognized value would sail
    through undetected. The pair is then looked up in ``_FAMILY_BY_PAIR``;
    a pair not in that table is a backbone whose structured ("on") variant
    does not exist YET -- refused by name, naming the phase that builds it,
    never silently ignored or misrouted to an unrelated family.
    """
    unknown = sorted(set(arm) - _ARM_KEYS)
    if unknown:
        raise ValueError(f"arms[{idx}]: unknown keys {unknown}")
    missing = sorted({"backbone", "structure"} - set(arm))
    if missing:
        raise ValueError(f"arms[{idx}]: missing keys {missing}")
    if arm["backbone"] not in _BACKBONES:
        raise ValueError(f"arms[{idx}]: unknown backbone {arm['backbone']!r}, expected one of {_BACKBONES}")
    if arm["structure"] not in _STRUCTURES:
        raise ValueError(
            f"arms[{idx}]: structure must be 'off' or 'on', got {arm['structure']!r}"
        )
    pair = (arm["backbone"], arm["structure"])
    if pair not in _FAMILY_BY_PAIR:
        raise ValueError(
            f"arms[{idx}]: backbone={arm['backbone']!r} structure='on' has no predictor "
            "family yet -- the structured (Koopman) variant of this backbone is built in "
            "phase C, it does not exist today"
        )
    return _FAMILY_BY_PAIR[pair]


def _trainer_allowed_keys(families: set[str]) -> frozenset[str]:
    """Every key ``base``/``overrides`` may legally name, for ``families``.

    The Trainer is the reference of truth (F7): ``_DEFAULTS`` (tunable,
    has a default) plus the non-runner-owned entries of ``_REQUIRED_KEYS``
    (``d_x``/``d_z``/``d_u`` -- ``exp_name``/``seed``/``family`` are
    injected by the runner) plus each family's ``_FAMILY_CONFIG_KEYS``.
    """
    family_keys: frozenset[str] = frozenset().union(
        *(trainer_mod._FAMILY_CONFIG_KEYS[f] for f in families)
    )
    return (
        frozenset(trainer_mod._DEFAULTS)
        | (frozenset(trainer_mod._REQUIRED_KEYS) - _RUNNER_OWNED_KEYS)
        | family_keys
    )


def _validate_trainer_keys(location: str, keys: set[str], allowed: frozenset[str]) -> None:
    """Reject any key not in the Trainer's own accepted schema (F7).

    A typo like ``lrr`` or ``mu_jj`` must never silently enter the config
    hash and do nothing -- it did with ``factorial.py``'s OLD schema, which
    only validated container keys (``base``/``arms``) and never looked
    inside them. Budget-owned and derived/policy-owned keys get the
    Trainer's own rejection message (they ARE known names, just never
    settable from a run config); anything else is simply unrecognized.
    """
    runner_owned = sorted(keys & _RUNNER_OWNED_KEYS)
    if runner_owned:
        raise ValueError(
            f"{location}: keys {runner_owned} are injected by the runner "
            "(exp_name/seed/family), not settable in base/overrides"
        )
    budget_owned = sorted(keys & trainer_mod._BUDGET_OWNED_KEYS)
    if budget_owned:
        raise ValueError(
            f"{location}: config keys {budget_owned} are capacity knobs owned by "
            "match_budget -- set target_params instead (params +/-5% is the "
            "fairness contract)"
        )
    derived = sorted(keys & trainer_mod._DERIVED_OR_POLICY_KEYS)
    if derived:
        raise ValueError(
            f"{location}: config keys {derived} are derived or policy-owned (d_h "
            "follows from d_z/n_pairs; precision is the Trainer's declared "
            "CPU/float32 policy)"
        )
    unknown = sorted(keys - allowed)
    if unknown:
        raise ValueError(f"{location}: unknown Trainer config keys {unknown}")


def expand(config: dict, dry_run: bool = False) -> list[dict]:
    """Validate ``config`` and expand it into the deterministic list of run cells.

    Order: datasets in YAML insertion order, then arms in list order, then
    seeds in list order. Each cell carries ``dataset``/``backbone``/
    ``structure``/``family`` (derived)/``seed``/``mu_j`` (display
    convenience) and the resolved ``Trainer`` config (``base + {exp_name,
    seed, family} + arm overrides``).

    Validation runs BEFORE any dataset is built or any model is
    constructed. A closed schema rejects an unknown key at any level,
    including inside ``base`` and every arm's ``overrides`` (F7) -- the
    Trainer's own accepted key sets are the reference of truth, not a
    duplicated list. ``structure`` is architectural (F8): it selects a
    predictor family via ``_FAMILY_BY_PAIR``, never a numeric penalty.
    ``mu_j`` is no longer an axis -- any cell resolving ``mu_j > 0`` is
    refused (see below). A ``null`` arm override is a ``ValueError``
    UNLESS ``dry_run`` is set, in which case the cell is still listed with
    the null displayed.
    """
    _validate_top(config)
    exp_name = config["exp_name"]
    datasets_cfg = config["datasets"]
    base = config.get("base", {})
    arms = config["arms"]
    seeds = config["seeds"]

    for ds_name, ds_cfg in datasets_cfg.items():
        _validate_dataset_entry(ds_name, ds_cfg)
    families = [_validate_arm(i, arm) for i, arm in enumerate(arms)]

    # F7: closed schema over base AND every arm's overrides
    _validate_trainer_keys("base", set(base), _trainer_allowed_keys(set(families)))
    for i, (arm, family) in enumerate(zip(arms, families)):
        arm_allowed = _trainer_allowed_keys({family})
        _validate_trainer_keys(f"arms[{i}].overrides", set(arm.get("overrides", {})), arm_allowed)

    cells = []
    idx = 0
    for ds_name in datasets_cfg:
        for arm, family in zip(arms, families):
            overrides = arm.get("overrides", {})
            null_keys = sorted(k for k, v in overrides.items() if v is None)
            if null_keys and not dry_run:
                raise ValueError(
                    f"arm backbone={arm['backbone']!r} structure={arm['structure']!r}: "
                    f"overrides {null_keys} are null -- value to be filled by "
                    "calibration (PROTOCOL.md section 5)"
                )
            # mu_j is no longer an experimental axis (structure is
            # architectural now) -- but the Trainer still accepts a mu_j
            # knob, and training on a resolved mu_j > 0 would require the
            # frozen gauge (PROTOCOL.md section 10.3) that this runner never
            # threads into Trainer (``Trainer(cell["config"])`` in
            # ``_preflight_resolve_and_check_collisions`` always passes
            # ``gauge=None``). Refused here, before any dataset is built,
            # same convention as the null-override refusal above:
            # ``--dry-run`` still lists the cell (the null is displayed
            # there too), only the run path refuses.
            resolved_mu_j = (
                overrides["mu_j"] if "mu_j" in overrides
                else base.get("mu_j", trainer_mod._DEFAULTS["mu_j"])
            )
            if (
                not dry_run
                and isinstance(resolved_mu_j, (int, float))
                and resolved_mu_j > 0
            ):
                raise ValueError(
                    f"arm backbone={arm['backbone']!r} structure={arm['structure']!r}: "
                    f"mu_j={resolved_mu_j!r} > 0 -- mu_j is not an experimental axis, "
                    "structure is architectural, and this runner cannot supply the "
                    "frozen gauge PROTOCOL.md section 10.3 requires to train on a "
                    "positive mu_j"
                )
            for seed in seeds:
                run_config = {
                    **base,
                    "exp_name": exp_name,
                    "seed": seed,
                    "family": family,
                    **overrides,
                }
                cells.append(
                    {
                        "index": idx,
                        "dataset": ds_name,
                        "backbone": arm["backbone"],
                        "structure": arm["structure"],
                        "family": family,
                        "seed": seed,
                        "mu_j": overrides.get("mu_j"),
                        "config": run_config,
                    }
                )
                idx += 1
    return cells


def _build_dataset_for(ds_cfg: dict) -> dict:
    system = _SYSTEMS[ds_cfg["system"]["name"]](**ds_cfg["system"].get("params", {}))
    regime = _REGIMES[ds_cfg["regime"]["name"]](**ds_cfg["regime"].get("params", {}))
    return build_dataset(
        system, regime, None, ds_cfg["n_episodes"], ds_cfg["T"], ds_cfg["dt"], ds_cfg["seed"]
    )


def _preflight_resolve_and_check_collisions(cells: list[dict], datasets: dict) -> None:
    """Resolve every cell's ``Trainer`` + run identity BEFORE any training.

    Two purposes (adversarial review F8/F17): (1) every cell gets a
    concrete ``run_dir`` up front, so the index always covers the FULL
    expected grid even if training stops early (F17a); (2) two distinct
    cells resolving to the SAME run_dir (config_hash + data_hash) is a
    scientific bug -- two different treatments would silently share one
    artifact directory, the second run resuming/overwriting the first's.
    Raises ``ValueError`` listing every collision, before touching disk.
    """
    by_run_dir: dict[Path, list[dict]] = {}
    for cell in cells:
        trainer = Trainer(cell["config"])
        run_dir = trainer.resolve_run_dir(datasets[cell["dataset"]])
        cell["trainer"] = trainer
        cell["run_dir"] = run_dir
        by_run_dir.setdefault(run_dir, []).append(cell)

    collisions = {rd: cs for rd, cs in by_run_dir.items() if len(cs) > 1}
    if collisions:
        lines = [
            f"{run_dir}: "
            + ", ".join(
                f"[{c['index']}] {c['dataset']} {c['backbone']} {c['structure']} {c['family']} seed={c['seed']}"
                for c in cs
            )
            for run_dir, cs in sorted(collisions.items())
        ]
        raise ValueError(
            "run identity collision(s) -- distinct cells share a run_dir:\n" + "\n".join(lines)
        )


def _final_val_loss(run_dir: Path) -> str:
    """Last ``val_loss`` field in ``run_dir/metrics.csv``, raw string.

    Returns ``""`` if the file is absent, empty, headerless, missing the
    column, undecodable (M4, adversarial review round 3 -- a corrupted-byte
    metrics.csv is the same "unreadable" case, not a crash), or otherwise
    unreadable (F17b) -- a truncated write from an interrupted run must
    never raise out of index generation. Does NOT judge whether the value
    is finite; that is the caller's job (F17c).
    """
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return ""
    try:
        with open(metrics_path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return ""
        val = rows[-1]["val_loss"]
    except (csv.Error, KeyError, IndexError, OSError, UnicodeError):
        return ""
    # a torn row (fewer fields than the header) gives DictReader's restval
    # (None) for the missing column, not a KeyError -- same "unreadable" case
    return val if val is not None else ""


# the 7 keys the sentinel must carry -- exactly (M9, adversarial review round
# 2; ckpt_sha1/metrics_rows added M3, round 3): a missing key or an empty {}
# are both valid JSON but neither is a trustworthy "complete" (json.loads
# also accepts the bare NaN/Infinity literals Python's json module writes, so
# schema shape alone is not enough -- see the finite check below)
_SENTINEL_KEYS = frozenset(
    {
        "config_hash",
        "data_hash",
        "epochs_completed",
        "final_train_loss",
        "final_val_loss",
        "ckpt_sha1",
        "metrics_rows",
    }
)
# every artifact the public contract (REGLES section 6) promises for a
# genuinely complete run
_REQUIRED_ARTIFACTS = ("ckpt.pt", "metrics.csv", "config.yaml", "audit.json")
# ``figs/`` is part of that same public contract, but unlike the artifacts
# above it must be a directory rather than a regular non-empty file.
_REQUIRED_DIRECTORIES = ("figs",)


def _artifact_ok(path: Path) -> bool:
    """``path`` is a regular, non-empty, readable file (M3, adversarial
    review round 3): a DIRECTORY named ``ckpt.pt``, or a zero-byte file left
    by a truncated write, must never be accepted as a real artifact just
    because ``exists()`` is true. Reads one byte rather than checking
    permission bits (``os.access`` can lie under ACLs) -- any failure to
    read is treated as "not a usable artifact", never propagated."""
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        with open(path, "rb") as f:
            f.read(1)
        return True
    except OSError:
        return False


def _validate_completion_sentinel(run_dir: Path) -> dict | None:
    """The trustworthy completion sentinel in ``run_dir``, or ``None``.

    ONE validator, shared by BOTH the resume check in ``main`` and
    ``_cell_status`` (index generation) -- M9, adversarial review round 2 --
    so the two can never disagree on what "complete" means. Before M9,
    resume skipped on the sentinel's mere EXISTENCE and the index accepted
    any ``final_val_loss`` that merely parsed: a truncated write landing
    ``{"final_val_loss": NaN}`` (which ``json.loads`` happily accepts --
    Python's json module parses the bare ``NaN``/``Infinity`` literals it
    also writes) was skipped by resume yet would have been rejected as
    ``nonfinite`` if read like a normal metrics.csv row, and an empty ``{}``
    sentinel was skipped by resume too.

    M3 (adversarial review round 3) closed a further gap: the checks above
    only ever compared STRINGS (the sentinel's own claimed hashes against
    the run dir's folder name) and only checked artifact *existence* --
    never that the sentinel is actually bound to what is on disk. Three
    concrete repros this closes: (a) a genuine sentinel from one run,
    presented next to the OTHER run's ``config.yaml`` (mismatched names but
    matching claimed hashes) used to validate; (b) a zero-byte file or a
    directory named ``ckpt.pt`` used to satisfy the artifact check; (c) a
    substituted checkpoint of identical shape from a different run would be
    silently accepted and evaluated under the wrong cell identity.

    A sentinel is trusted only when ALL of:

    - it parses as a JSON object carrying every one of the 7 declared keys
      (``_SENTINEL_KEYS``) -- a missing key or an empty ``{}`` both fail
      this despite being valid JSON;
    - its ``config_hash``/``data_hash`` match the identity encoded in
      ``run_dir``'s OWN name -- every run dir is named
      ``f"{config_hash}-{data_hash}"`` (``Trainer.resolve_run_dir``, both 8
      hex chars, never containing ``-``);
    - every required file artifact (``ckpt.pt``, ``metrics.csv``,
      ``config.yaml``, ``audit.json``) is a regular, non-empty, readable FILE
      (``_artifact_ok``), and ``figs/`` exists as a DIRECTORY -- not merely
      "exists";
    - the config hash is RECOMPUTED from ``config.yaml``'s own bytes and
      matches the folder name -- ``Trainer`` writes ``config.yaml`` as
      EXACTLY ``self._canonical_yaml`` (the sorted, ``results_root``-excluded
      dump that was hashed at construction -- see ``Trainer.__init__``), so
      hashing the file's raw text reproduces ``config_hash`` bit for bit. A
      ``config.yaml`` swapped in from a DIFFERENT run's identity (repro (a))
      recomputes to a DIFFERENT hash and is rejected here, never trusted by
      comparing names alone;
    - ``epochs_completed`` equals the ``epochs`` this run's OWN
      ``config.yaml`` declares;
    - both losses are FINITE -- checked with ``math.isfinite``, never "did
      it parse" (REGLES section 5/6, finite-metrics law: NaN is a bug,
      never a result);
    - ``ckpt_sha1`` matches a digest RECOMPUTED by streaming ``ckpt.pt``
      (``Trainer._sha1_file`` -- never ``torch.load`` here: a corrupted or
      substituted checkpoint of the same shape would load fine but must
      still be rejected, repro (c));
    - ``metrics.csv`` itself is bound, not just its row count (M3, round 4 --
      before this, only the sentinel's OWN claimed losses were checked for
      finiteness, and metrics.csv was checked for row count/last epoch only,
      so corruption confined to metrics.csv alone -- e.g. NaN written into
      the last row, or a finite but different value -- validated clean):
      the header is EXACTLY ``trainer_mod.METRICS_HEADER``; ``metrics_rows``
      and ``epochs_completed`` both equal the row count actually read back;
      the epoch column is the exact sequence ``0..N-1``; every numeric value
      in every row is finite; and the LAST row's ``train_loss``/``val_loss``
      are EXACTLY the sentinel's ``final_train_loss``/``final_val_loss``.

    Every file read here (sentinel, config.yaml, metrics.csv) can fail with
    a decode error on corrupted bytes, not just a parse error -- caught
    alongside the format errors (M4, adversarial review round 3). An
    oversized JSON integer (hundreds of digits) in a loss field parses fine
    via ``json.loads`` but raises ``OverflowError`` when converted to
    ``float`` -- caught the same way (M4, round 4). A malformed or
    adversarial run must invalidate the sentinel, never crash the caller.

    Returns the parsed sentinel dict when every check passes, else
    ``None`` -- callers treat ``None`` as "this run is NOT complete, retrain
    it", never as an error to propagate.
    """
    sentinel_path = run_dir / RUN_COMPLETE_FILENAME
    if not sentinel_path.exists():
        return None
    try:
        sentinel = json.loads(sentinel_path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeError):
        return None
    if not isinstance(sentinel, dict) or set(sentinel) != _SENTINEL_KEYS:
        return None

    name_parts = run_dir.name.split("-")
    if (
        len(name_parts) != 2
        or sentinel["config_hash"] != name_parts[0]
        or sentinel["data_hash"] != name_parts[1]
    ):
        return None

    if any(not _artifact_ok(run_dir / artifact) for artifact in _REQUIRED_ARTIFACTS):
        return None
    if any(not (run_dir / directory).is_dir() for directory in _REQUIRED_DIRECTORIES):
        return None

    # M3: recompute, don't just compare strings -- config.yaml's raw bytes
    # ARE the canonical yaml Trainer hashed (see docstring above)
    try:
        config_text = (run_dir / "config.yaml").read_text()
    except (OSError, UnicodeError):
        return None
    recomputed_config_hash = hashlib.sha1(config_text.encode("utf-8")).hexdigest()[:8]
    if recomputed_config_hash != name_parts[0]:
        return None

    try:
        run_config = yaml.safe_load(config_text)
    except yaml.YAMLError:
        return None
    if not isinstance(run_config, dict) or sentinel["epochs_completed"] != run_config.get("epochs"):
        return None

    # M4 (round 4): an oversized JSON integer (hundreds of digits) parses
    # fine as a Python int, but converting IT to float overflows -- caught
    # here alongside the format errors, never left to propagate out of the
    # validator and crash both resume and index generation
    try:
        train_loss = float(sentinel["final_train_loss"])
        val_loss = float(sentinel["final_val_loss"])
    except (TypeError, ValueError, OverflowError):
        return None
    if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
        return None

    # M3: checkpoint digest, recomputed by STREAMING the file -- never
    # torch.load in the validator (repro (c): a substituted checkpoint of
    # the same shape would load fine but must still be rejected)
    try:
        if trainer_mod._sha1_file(run_dir / "ckpt.pt") != sentinel["ckpt_sha1"]:
            return None
    except OSError:
        return None

    # M3 round 4: metrics.csv coherence goes further than row count and the
    # last epoch alone -- BOTH used to be the only things checked, so a
    # corruption confined entirely to metrics.csv (sentinel JSON untouched,
    # row count and final epoch preserved) validated clean. Now checked, all
    # read back from disk rather than trusted:
    #   - schema: the header is EXACTLY trainer_mod.METRICS_HEADER;
    #   - every numeric value in every row is finite (the finite-metrics law
    #     applies to what actually landed on disk, not just the sentinel's
    #     own claimed final values);
    #   - the epoch column is the exact sequence 0..N-1, not merely "the
    #     last one matches";
    #   - the LAST row's train_loss/val_loss are EXACTLY the sentinel's
    #     final_train_loss/final_val_loss -- the sentinel is a claim ABOUT
    #     this file, not an independent source of truth.
    try:
        with open(run_dir / "metrics.csv", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != list(trainer_mod.METRICS_HEADER):
                return None
            metrics_rows = list(reader)
    except (csv.Error, OSError, UnicodeError):
        return None
    if not metrics_rows or len(metrics_rows) != sentinel["metrics_rows"]:
        return None
    # DictReader preserves cells beyond the exact header under a ``None``
    # key. They are still values in metrics.csv, so accept neither a hidden
    # nonfinite value nor any unregistered trailing field.
    if any(None in row for row in metrics_rows):
        return None
    # epochs_completed must ALSO be grounded in the row count actually read
    # back (not merely internally consistent with metrics_rows inside the
    # sentinel's own JSON) -- ties the two sentinel fields together via disk
    if sentinel["epochs_completed"] != len(metrics_rows):
        return None
    try:
        epochs = [int(row["epoch"]) for row in metrics_rows]
        numeric_values = [
            float(row[col])
            for row in metrics_rows
            for col in trainer_mod.METRICS_HEADER
            if col != "epoch"
        ]
        last_train_loss = float(metrics_rows[-1]["train_loss"])
        last_val_loss = float(metrics_rows[-1]["val_loss"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if epochs != list(range(len(metrics_rows))):
        return None
    if not all(math.isfinite(v) for v in numeric_values):
        return None
    if last_train_loss != train_loss or last_val_loss != val_loss:
        return None

    return sentinel


def _cell_status(run_dir: Path) -> tuple[str, str]:
    """``(status, final_val_loss)`` for one cell's ``run_dir``.

    ``run_complete.json`` (F9), VALIDATED by ``_validate_completion_sentinel``
    (M9), is the ONLY trustworthy completion signal -- a ``ckpt.pt`` without
    a VALID sentinel means an interrupted (or corrupted) run, re-executed on
    the next invocation, never reported as ``complete``. A sentinel that
    exists but fails validation gets its own explicit status
    (``invalid_sentinel``), distinct from both ``complete`` and a run that
    never got as far as writing one at all. A run dir carrying
    ``failure.json`` (``FAILURE_FILENAME``, ``Trainer.fit``) is reported as
    ``divergence`` (PROTOCOL.md section 11.7) -- a TERMINAL result about
    that (family, regime, seed, LR), distinct from ``missing`` (a cell never
    attempted at all): before this status existed, a diverged cell and a
    never-run cell were both indistinguishably ``missing``. Otherwise
    metrics.csv's last row (if any) is read defensively; a value that fails
    to parse or is not finite (nan/inf) is reported as status ``nonfinite``
    -- explicit, never silently accepted as a valid number (REGLES section
    5, finite-metrics law; F17c).
    """
    sentinel = _validate_completion_sentinel(run_dir)
    if sentinel is not None:
        return "complete", str(sentinel["final_val_loss"])
    if (run_dir / RUN_COMPLETE_FILENAME).exists():
        # present but untrustworthy (M9): never "complete", but distinct
        # from a run that never wrote a sentinel at all
        return "invalid_sentinel", _final_val_loss(run_dir)
    if (run_dir / FAILURE_FILENAME).exists():
        # PROTOCOL.md section 11.7: a divergence is a terminal, durable
        # result -- reported by name, never folded into "missing" (which
        # means "this cell was never attempted"). The reported value is
        # metrics.csv's last GOOD row (the epoch before the abort), the
        # same value this run dir already reported before failure.json
        # existed -- only the status label changes.
        return "divergence", _final_val_loss(run_dir)

    raw = _final_val_loss(run_dir)
    if raw == "":
        return "missing", ""
    try:
        # M4: an oversized numeric literal (hundreds of digits) parses as an
        # int via float()'s own text parsing, then overflows converting to
        # float -- caught alongside the plain parse failure, never left to
        # crash index generation
        finite = math.isfinite(float(raw))
    except (ValueError, OverflowError):
        finite = False
    return ("missing" if finite else "nonfinite"), raw


def _write_index(results_root: Path, exp_name: str, cells: list[dict]) -> None:
    """Regenerate ``results_root/exp_name/index.csv`` from the resolved cells.

    Idempotent and resilient (F17a): ``cells`` is expected to be the FULL
    expansion with ``run_dir`` already resolved for every cell (preflight),
    so every cell in the grid appears -- ``missing`` if nothing usable is
    on disk -- even when the run stopped partway through the grid.
    """
    # Column layout matches the phase B4 contract (``sld.experiments.loading
    # ._INDEX_COLUMNS``): the column is named "structure" (not "constraint"
    # -- that name lied about what it carried after B1 repointed the axis)
    # and holds ``cell["structure"]``'s value ("off"/"on"). Indices already
    # on disk under the old "constraint" name (results/, results_amended/,
    # reports/historical/mu_j/traces/) are frozen and never rewritten;
    # ``loading.read_index`` tolerates that legacy name on read.
    index_dir = results_root / exp_name
    index_dir.mkdir(parents=True, exist_ok=True)
    with open(index_dir / "index.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_dir", "regime", "family", "structure", "seed", "status", "final_val_loss"])
        for cell in cells:
            run_dir = cell["run_dir"]
            status, val_loss = _cell_status(run_dir)
            writer.writerow(
                [run_dir, cell["dataset"], cell["family"], cell["structure"], cell["seed"], status, val_loss]
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sld-factorial")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.dry_run:
        cells = expand(config, dry_run=True)
        for cell in cells:
            print(
                f"[{cell['index']}] {cell['dataset']} {cell['backbone']} {cell['structure']} "
                f"{cell['family']} seed={cell['seed']} mu_j={cell['mu_j']}"
            )
        print(f"total: {len(cells)} runs")
        return 0

    cells = expand(config, dry_run=False)
    exp_name = config["exp_name"]
    results_root = Path(config.get("base", {}).get("results_root", "results"))
    datasets = {name: _build_dataset_for(cfg) for name, cfg in config["datasets"].items()}

    # every cell's Trainer + run_dir resolved up front, and any identity
    # collision rejected before touching disk (F8c) -- nothing has been
    # attempted yet at this point, so no index is written on this path,
    # same as any other pre-training config error (expand() already
    # behaves this way)
    _preflight_resolve_and_check_collisions(cells, datasets)

    any_failed = False
    n = len(cells)
    try:
        for cell in cells:
            trainer = cell["trainer"]
            run_dir = cell["run_dir"]
            data = datasets[cell["dataset"]]
            label = (
                f"[{cell['index'] + 1}/{n}] {cell['dataset']} {cell['backbone']} "
                f"{cell['structure']} {cell['family']} seed={cell['seed']}"
            )
            if _validate_completion_sentinel(run_dir) is not None:
                print(f"{label} -> {run_dir} (SKIP, resume)")
                continue
            if (run_dir / FAILURE_FILENAME).exists():
                # PROTOCOL.md section 11.7: a divergence is TERMINAL --
                # never re-seeded, never silently retrained on the next
                # invocation. (11.7 makes no determinism claim: the runs are
                # seeded, but deterministic algorithms across execution
                # environments are not established in this repo.) Reported
                # and moved on, exactly like the resume skip above; the
                # overall exit code still reflects that this cell did not
                # succeed (no CLI flag to force a retrain -- out of scope,
                # and 11.7 deliberately restricts replay to a separate
                # namespace).
                any_failed = True
                print(f"{label} -> {run_dir} (SKIP, divergence)")
                continue
            t0 = time.perf_counter()
            try:
                result = trainer.fit(data)
            except RuntimeError as exc:
                any_failed = True
                print(f"{label} -> {run_dir} (FAILED: {exc})")
                continue
            dt = time.perf_counter() - t0
            val_loss = result["metrics"]["final_val_loss"]
            print(f"{label} -> {run_dir} (OK, val_loss={val_loss:.4f}, {dt:.1f}s)")
    except BaseException:
        # F17d: an exception writing the index here must never REPLACE the
        # original exception from the grid loop above -- best-effort write,
        # then always re-raise the original
        try:
            _write_index(results_root, exp_name, cells)
        except Exception as index_exc:
            print(f"warning: failed to write index.csv after an earlier error: {index_exc}")
        raise

    _write_index(results_root, exp_name, cells)
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
