"""Config-driven JEPA training loop, one code path for all predictor families.

A run is ``config + seed -> results/<exp_name>/<hash>/`` (REGLES section 6):
deterministic, regenerable, and returning the public
``{"model", "history", "metrics", "paths"}`` contract shared with
``sld-factorial`` so ``viz/`` can read any family's results without knowing
which one produced them.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import yaml
from torch import nn
from tqdm import tqdm

from sld.encoders.frame_mlp import FrameMLPEncoder
from sld.objectives.jac_reg import Gauge, TotalJacobianPenalty, gauge_digest
from sld.objectives.jepa import EMATarget, JEPALoss
from sld.objectives.vicreg import covariance_loss, variance_loss
from sld.predictors.base import PredictorState
from sld.predictors.registry import STRUCTURED_FAMILIES, get_predictor
from sld.training.seeding import set_seed

if TYPE_CHECKING:
    # deferred to a LAZY import inside Trainer.__init__ (see there): a
    # module-level import here would close a real cycle --
    # sld.encoders.pretrain imports sld.training.seeding, which (as part of
    # importing the sld.training package) imports this very module. Typing
    # only, never evaluated at runtime (PEP 563, `from __future__ import
    # annotations` above already stringifies every annotation in this file).
    from sld.encoders.pretrain import FrozenRepresentationArtifact

# resolved for any key the caller omits; unknown extra keys the caller DOES
# pass are kept as-is (they still flow into the config hash and config.yaml)
_DEFAULTS = {
    "epochs": 50,
    "batch_episodes": 8,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "tau_ema": 0.99,
    "lambda_c": 0.0,
    "w_tf": 1.0,
    "w_fr": 1.0,
    "h_fr": 5,
    "w_var": 1.0,
    "w_cov": 1.0,
    "mu_j": 0.0,
    "k_probes": 2,
    "step_subsample": 0.25,
    "encoder_hidden": [64, 64],
    "target_params": 10_000,
    "results_root": "results",
}

# model-building keys every run must declare (a run is config + seed, so the
# models MUST be reconstructible from the config alone -- REGLES section 6)
_REQUIRED_KEYS = ("exp_name", "seed", "family", "d_x", "d_z", "d_u")

# model-defining config keys accepted PER FAMILY: a key present in the config
# but not applicable to the declared family is REJECTED at construction --
# silently ignoring it would let distinct run identities build identical
# models (adversarial review round 2)
#
# The three structured arms (PROTOCOL.md section 13.1) share ONE key set
# plus their backbone's own extras -- the same shape as their shared
# constructor signature, so canonicalization below is one branch for the
# three, not three.
# ``radius`` is the third level of the structure factor (PROTOCOL section 23),
# "free" or "bounded". It is a PARAMETRISATION, not a capacity knob, so it is
# config-settable and hashed into the run identity like ``delta``.
# ``lift`` is the third level's knob (PROTOCOL 25.3), and like ``radius`` it is
# NOT canonicalized into the identity below: the family branch there injects
# its defaults BEFORE hashing, so resolving `lift` there would add the key to
# every existing structured run's config and move all 134 config_hashes. Absent
# means "mlp", decided at construction in budget._kwargs_for, exactly as
# ``radius`` already works.
_STRUCTURED_KEYS = frozenset({"delta", "r", "radius", "lift"})
_FAMILY_CONFIG_KEYS: dict[str, frozenset[str]] = {
    "mlp": frozenset(),
    "transformer": frozenset({"n_heads", "max_len"}),
    "min_ssm": frozenset({"delta_min", "delta_max"}),
    "koopman_ssm": frozenset(
        {"delta", "regime", "constrained", "n_pairs", "delta_min", "delta_max"}
    ),
    "mlp_k": _STRUCTURED_KEYS,
    "transformer_k": _STRUCTURED_KEYS | {"n_heads", "max_len"},
    "ssm_k": _STRUCTURED_KEYS | {"delta_min", "delta_max"},
}
_MODEL_KEYS = frozenset().union(*_FAMILY_CONFIG_KEYS.values())

# capacity knobs are OWNED by match_budget (the fairness contract sizes them
# from target_params): a config naming one directly would be hashed into the
# run identity yet silently ignored by the forwarding above -- reject them
# outright (adversarial review round 3)
_BUDGET_OWNED_KEYS = frozenset(
    {"d_model", "d_ff", "n_layers", "hidden", "d_state", "psi_hidden", "pi_hidden", "width"}
)

# derived or policy-owned constructor names, rejected for every family
# (round 4 -- closes the schema over ALL public predictor-constructor names):
# d_h is derived (d_z for plain families, 2*n_pairs for koopman_ssm); dtype
# is the Trainer's precision policy (CPU float32, declared in the class
# docstring), not a per-run knob. Classification of every constructor name:
#   forwarded: d_z, d_u (required keys) + _FAMILY_CONFIG_KEYS per family
#   budget-owned (rejected): hidden, d_model, d_ff, n_layers, d_state,
#     psi_hidden, pi_hidden
#   derived/policy (rejected): d_h, dtype, gauge_digest
#
# gauge_digest is DERIVED from the out-of-band ``gauge=`` constructor
# argument (PROTOCOL.md section 10.3's frozen reference scale), never
# declared by a caller: it is injected into the resolved config below, so a
# config that also NAMED it would hash a value the Trainer is about to
# overwrite. The schema stays closed at every level -- the factorial's
# ``_trainer_allowed_keys`` derives its allow-list from ``_DEFAULTS``, which
# deliberately does NOT carry ``gauge_digest``, so a YAML naming it is
# rejected there too.
_DERIVED_OR_POLICY_KEYS = frozenset({"d_h", "dtype", "gauge_digest", "representation_digest"})

# objective weights: must be finite and >= 0 (a negative weight silently
# behaving as "disabled" was adversarial review round 2's finding), and at
# least one PREDICTION objective must be active or there is nothing to train
_WEIGHT_KEYS = ("w_tf", "w_fr", "w_var", "w_cov")

_HISTORY_DIAGNOSTIC_KEYS = ("jac_margin", "r_ctrl", "lip_pi", "vpt")

# the ONLY trustworthy completion signal for a run (adversarial review F9):
# ckpt.pt is written directly to its final path mid-``fit``, so a kill
# between that write and the end of ``fit`` leaves a plausible-looking but
# incomplete artifact set. This file is written LAST, atomically (temp file
# + os.replace in the same directory -- atomic on POSIX), strictly after
# ckpt.pt and audit.json. Its presence -- not ckpt.pt's -- is what resume
# detection and the factorial index's "complete" status key on.
RUN_COMPLETE_FILENAME = "run_complete.json"

# the run's TERMINAL failure sentinel (PROTOCOL.md section 11.7): written
# when -- and only when -- training aborts on a non-finite loss, so the
# divergence is a DURABLE, distinguishable result about this (family,
# regime, seed, LR), never indistinguishable from a cell that was simply
# never attempted. Written the SAME way as RUN_COMPLETE_FILENAME (temp file
# + os.replace in the same directory), so a kill mid-write never leaves a
# half-written failure record either. Unlike RUN_COMPLETE_FILENAME, this
# file is deliberately NOT among the stale artifacts ``fit`` clears at the
# top of a run (see the cleanup loop below) -- it must survive a cleanup
# pass rather than be silently wiped, because 11.7 forbids re-seeding or
# silently retrying a divergence. ``sld-factorial`` (``factorial.py``)
# checks for it before ever calling ``fit`` again on this identity.
FAILURE_FILENAME = "failure.json"

# PROTOCOL.md section 14.7: crash-resume. A snapshot (model + optimizer +
# scheduler + RNG state) written atomically every this many optimizer
# updates -- and in this repo epochs ARE updates, one batch/one
# optimizer.step() per epoch (Trainer docstring). A MODULE CONSTANT, never a
# config key: only results_root is excluded from the config hash, so a
# config key for this would move every config_hash, therefore every run
# directory, therefore make every sentinel already on disk unreachable.
RESUME_CHECKPOINT_INTERVAL = 100

# Deliberately NOT ckpt.pt: this is a different object with a different
# lifetime -- a rolling mid-run snapshot, overwritten every
# RESUME_CHECKPOINT_INTERVAL updates and DELETED entirely on successful
# completion (a finished run leaves no resume state), whereas ckpt.pt is the
# final weights, written once at the very end and -- like this file -- never
# itself a completion signal (RUN_COMPLETE_FILENAME is).
RESUME_CHECKPOINT_FILENAME = "resume_ckpt.pt"

# a resume checkpoint missing any of these is corrupt/truncated and must be
# REFUSED -- never silently adopted, never silently deleted (section 14.7)
_RESUME_REQUIRED_KEYS = frozenset(
    {
        "config_hash",
        "epoch",
        "encoder",
        "predictor",
        "ema_target",
        "optimizer",
        "scheduler",
        "torch_rng_state",
        "numpy_rng_state",
        "python_rng_state",
    }
)

# per-run progress log (section 14.7): resolved config + config_hash + run
# dir + budget first, then one line per epoch. Joins the fresh-start cleanup
# list below -- a killed run's log must not survive into the next attempt at
# the same identity.
TRAIN_LOG_FILENAME = "train.log"


def _save_resume_checkpoint(
    path: Path,
    config_hash: str,
    epoch: int,
    encoder: nn.Module,
    predictor: nn.Module,
    ema: EMATarget,
    optimizer: torch.optim.Optimizer,
    scheduler,
) -> None:
    """Atomic write (temp file + ``os.replace`` in the same directory --
    identical discipline to ``RUN_COMPLETE_FILENAME``/``FAILURE_FILENAME``)
    of everything needed to resume training bit-identically from the END of
    ``epoch``: model/optimizer/scheduler state, and every RNG global
    ``fit``'s loop draws from -- torch, numpy, and python's ``random``
    (``set_seed`` seeds all three; a resume that restores only some of them
    is not bit-identical resume, it is a second, undeclared experiment)."""
    state = {
        "config_hash": config_hash,
        "epoch": epoch,
        "encoder": encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "ema_target": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def _load_resume_checkpoint(path: Path, config_hash: str) -> dict:
    """Load and validate a resume checkpoint, or refuse it loudly.

    Two distinct refusals, per PROTOCOL.md section 14.7 -- neither is ever
    silently adopted nor silently deleted: unreadable/missing-key ("corrupt
    or unreadable" / "corrupt: missing keys" in the message) and identity
    mismatch ("identity mismatch" in the message, when the checkpoint's own
    ``config_hash`` disagrees with this run's).
    """
    try:
        # weights_only=False is required, not a default left unexamined: the
        # payload carries optimizer/scheduler state and numpy's RNG state
        # tuple, none of which is a bare tensor. Trust boundary: this file is
        # written ONLY by _save_resume_checkpoint above, into a run directory
        # this Trainer instance owns -- never sourced from a peer, a
        # download, or any other untrusted origin.
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(
            f"resume checkpoint at {path} is corrupt or unreadable ({exc!r}) -- "
            "refusing to adopt or silently discard it"
        ) from exc
    missing = _RESUME_REQUIRED_KEYS - set(state) if isinstance(state, dict) else _RESUME_REQUIRED_KEYS
    if missing:
        raise RuntimeError(
            f"resume checkpoint at {path} is corrupt: missing keys {sorted(missing)} -- "
            "refusing to adopt or silently discard it"
        )
    if state["config_hash"] != config_hash:
        raise RuntimeError(
            f"resume checkpoint identity mismatch: checkpoint config_hash="
            f"{state['config_hash']!r} != this run's config_hash={config_hash!r} -- "
            "refusing to adopt or silently discard it"
        )
    return state


def _truncate_metrics_csv(path: Path, n_rows: int) -> None:
    """Truncate ``metrics.csv`` to its header plus exactly ``n_rows`` data rows.

    Resume-only: a process killed between a resume checkpoint save and the
    next several epoch writes leaves rows the checkpoint does not cover,
    possibly with a torn last row. ``n_rows`` is the checkpoint's own epoch
    count, so this discards exactly what the checkpoint cannot vouch for,
    torn or not, and never more or less.
    """
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)[:n_rows]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _metrics_csv_defect(path: Path, n_rows: int) -> str | None:
    """Why ``metrics.csv`` cannot back a resume of ``n_rows`` epochs, or ``None``.

    NOT a completeness check -- ``factorial._validate_completion_sentinel`` is
    the ONE validator for "is this run finished"; this answers a narrower
    question, asked only once ``fit`` has already decided to resume: can
    these bytes be parsed at all, and are the ``n_rows`` epochs the resume
    checkpoint claims actually on disk. Never grow this into a second
    completeness validator.

    Incident (2026-08-11): five run_dirs kept a metrics.csv full of NUL bytes
    (blocks allocated by the filesystem, contents never flushed) after a
    kill; resume's history rebuild crashed with ``ValueError: invalid
    literal for int() with base 10: '\\x00\\x00...'``, taking down a whole
    shard instead of retraining the one cell. Discarding here is safe no
    matter what: no ``run_complete.json`` means the run is incomplete by
    definition, so its metrics.csv certifies nothing -- there is nothing to
    lose by restarting the cell from epoch 0.

    The row-count check below leans on the same invariant as the
    truncate-then-rebuild above (section 14.7): ``fit`` flushes each epoch's
    metrics row strictly BEFORE saving the resume checkpoint that claims it,
    so a genuine resume checkpoint for epoch ``e`` always finds at least
    ``e + 1`` rows already on disk -- fewer means the tail never reached it.

    Rows at index >= ``n_rows`` are deliberately NOT value-checked:
    ``_truncate_metrics_csv`` discards exactly that tail by design (its own
    docstring: "possibly with a torn last row"), so a killed writer's torn or
    garbage tail beyond what the checkpoint claims is the EXPECTED shape of
    an ordinary crash, not a reason to throw away completed epochs the
    checkpoint already covers. Only ``rows[:n_rows]`` -- what the real resume
    path in ``fit`` actually value-parses after truncating -- is checked
    row-by-row; the full file is still read once (below) so a csv-level
    structural failure (e.g. an unterminated quoted field anywhere in the
    file) is caught exactly where ``_truncate_metrics_csv``'s own
    ``list(reader)`` would hit it, before any truncation happens.
    """
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return "metrics.csv has no header (file is empty)"
            rows = list(reader)
            for row in rows[:n_rows]:
                int(row["epoch"])
                float(row["train_loss"])
                float(row["val_loss"])
    except (OSError, csv.Error, ValueError, TypeError, KeyError) as exc:
        return f"{type(exc).__name__}: {exc}"[:200]
    if len(rows) < n_rows:
        return f"metrics.csv has {len(rows)} row(s), resume checkpoint claims {n_rows} epoch(s)"
    return None


# metrics.csv's exact header, written by fit() below and re-checked (schema,
# not just column presence) by the factorial validator's sentinel-binding
# check (M3, round 4) -- one shared constant so the two can never drift apart
#
# STATUS (jalon B2, demotion explicite) -- SECONDARY -- not primary thesis.
# Every mu_j hook in this file (config validation below, the "jac_pen"
# column, _composite_loss's mu_j branch, _jac_state_penalty) is the
# now-retired mu_J penalty's training-side wiring. It REMAINS FUNCTIONAL
# and REMAINS TESTED (tests/training/test_trainer.py, 44 tests) as a
# secondary regression path -- never a gate of phases C-E. Removing these
# hooks would still break any config that sets mu_j > 0 (mu_j defaults to
# 0.0 and is a no-op then, per the "OFF cell" doctrine below); this is
# exactly why the verdict here is DEMOTE, not delete.
#
# CONTRACT NOTE: "jac_pen" is the TRAIN-side mu_j diagnostic only. "val_loss"
# does NOT include mu_j * jac_pen -- the penalty is not evaluated on the val
# split at all (Trainer._composite_loss(..., compute_jac_penalty=False)),
# so val_loss is TF + FR + VICReg exclusively, even when mu_j > 0. This is a
# deliberate contract change: the penalty is unused downstream of val
# anywhere in this repo, evaluating it there cost 23-38% of epoch time, and
# it silently advanced the GLOBAL RNG that the FOLLOWING epoch's training
# probes draw from (TotalJacobianPenalty's generator is None by design).
#
# PROTOCOL.md 17.1 (Amendment 8) adds "grad_norm": the PRE-clip global
# gradient-norm reading (_global_grad_norm), recorded every epoch whether or
# not this run's config carries grad_clip -- an unmeasured guard is a silent
# knob.
METRICS_HEADER = (
    "epoch",
    "train_loss",
    "val_loss",
    "lr",
    "pred",
    "consistency",
    "var",
    "cov",
    "jac_pen",
    "grad_norm",
)

# A run's IDENTITY (config_hash) never included the metrics.csv SCHEMA, so a
# run written before Amendment 8 is COMPLETE with its own era's header --
# missing "grad_norm" is not corruption, it is simply older than the column.
# METRICS_HEADER_LEGACY is that prior header, pinned EXACTLY (never derived
# from METRICS_HEADER, which would make this constant track its own future
# changes instead of anchoring the one era it actually was). A CLOSED tuple
# of every header this repo has ever written to disk, oldest first --
# factorial._validate_completion_sentinel accepts a metrics.csv whose
# fieldnames equal EXACTLY one member of this tuple, never a prefix/subset
# match (a header that is neither is genuinely foreign or corrupt). Extend
# this tuple, never rewrite an existing entry, the next time METRICS_HEADER
# gains or loses a column.
METRICS_HEADER_LEGACY = (
    "epoch",
    "train_loss",
    "val_loss",
    "lr",
    "pred",
    "consistency",
    "var",
    "cov",
    "jac_pen",
)
METRICS_HEADER_GENERATIONS = (METRICS_HEADER_LEGACY, METRICS_HEADER)


def _sha1_file(path: Path) -> str:
    """Streamed SHA1 hex digest of ``path``'s bytes (M3, adversarial review
    round 3): the completion sentinel binds ``ckpt.pt`` by this digest, not
    by its mere existence, so a substituted checkpoint of the same shape is
    rejected by the factorial validator. Read in chunks -- ``torch.load``
    is never used here, in the trainer or the validator, just to fingerprint
    a file (REGLES section 5: no cheap check may become an expensive one)."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _time_major(split: dict, idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Episodes ``idx`` of a ``build_dataset`` split -> time-major float32 tensors.

    ``split["x"]`` is (n_ep, T+1, d_x), ``split["u"]`` is (n_ep, T, d_u), both
    float64 numpy; cast to float32 and transpose to (T[+1], B, d) here, once,
    at the single point where numpy meets torch.
    """
    x = torch.as_tensor(split["x"][idx], dtype=torch.float32).transpose(0, 1).contiguous()
    u = torch.as_tensor(split["u"][idx], dtype=torch.float32).transpose(0, 1).contiguous()
    return x, u


def _global_grad_norm(params) -> float:
    """The global L2 gradient norm of ``params``, PRE-clip (PROTOCOL.md
    17.1: measured every epoch, clipped or not -- "a guard whose bite is
    not measured would be a silent knob"). Same formula
    ``torch.nn.utils.clip_grad_norm_`` uses internally (norm_type=2.0,
    combine per-parameter norms via a second L2 norm), factored out here so
    it can be computed WITHOUT ever calling that function: when
    ``grad_clip`` is absent, gradients must stay bitwise untouched, so
    ``fit`` calls this for the measurement and skips ``clip_grad_norm_``
    entirely on that path -- a structural guarantee, not a numerical one.
    """
    grads = [p.grad.detach() for p in params if p.grad is not None]
    if not grads:
        return 0.0
    return float(torch.norm(torch.stack([torch.norm(g, 2) for g in grads]), 2))


def data_fingerprint(data: dict) -> str:
    """The dataset generation spec's 8-hex fingerprint -- the ``data_hash``
    half of a run identity (REGLES section 6, adversarial review round 2).

    Module-level and public because two callers need the SAME number:
    ``Trainer.resolve_run_dir`` (which builds the run directory from it) and
    the ``sld-factorial`` runner (which keys a stratum's frozen
    representation by the dataset it was pretrained on). A second
    implementation of the same quantity is exactly what R2 forbids -- two
    hashes that agree today and drift apart on the next schema change.

    ``paths`` is excluded: it is infrastructure (where the data landed), not
    the spec (what the data IS).
    """
    if "meta" not in data:
        raise ValueError(
            "data must carry its build_dataset 'meta' block: the dataset "
            "generation spec is part of the run identity"
        )
    meta = {k: v for k, v in data["meta"].items() if k != "paths"}
    return hashlib.sha1(json.dumps(meta, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:8]


class Trainer:
    """Config-driven JEPA trainer: wires an encoder + one predictor family
    (of the four in ``sld.predictors``) through ONE code path -- teacher-forced
    (TF) and free-running (FR) prediction, plus VICReg anti-collapse.

    Family dispatch is a single line: ``lifted = hasattr(predictor, "lift")``.
    Lifted families (``koopman_ssm``) carry a separate H-space reached by
    ``predictor.lift``/``predictor.project``; plain families operate directly
    in z-space, so lift/project are the identity. ``JEPALoss`` (see
    ``objectives/jepa.py``) then supplies both the projection-space ``pred``
    term (always) and, only when a family is lifted, the lifted-space
    ``consistency`` term.

    TF/FR mix is entirely config-declared (``w_tf``, ``w_fr``, ``h_fr``) and
    every value is printed into ``config.yaml`` -- the factorial study's
    comparability requirement is that no family gets a silently different
    schedule. TF threads ``predictor.step`` over the TRUE prefix (state
    accumulates across ground-truth tokens; the model never feeds its own
    predictions back in -- that is what teacher forcing means, including for
    a transformer's KV cache). FR uses ``predictor.rollout`` from ``h0`` for
    ``h_fr <= T`` steps.

    v0 batching (declared, not a stand-in for something more elaborate
    later in this wave): one sampled batch of ``batch_episodes`` train
    episodes per epoch, no dataloader machinery. Val loss is evaluated on
    the WHOLE val split every epoch under ``torch.no_grad()`` -- EXCLUDING
    the mu_j penalty (contract change, see ``METRICS_HEADER`` and
    ``_composite_loss``): ``val_loss`` in ``metrics.csv``/``history`` is
    TF + FR + VICReg only, never TF + FR + VICReg + mu_j * jac_pen. The
    penalty is not used off of the train split anywhere in this repo, and
    evaluating it there cost 23-38% of epoch time and silently advanced the
    GLOBAL RNG that the following epoch's training-time probes draw from.

    mu_J gauge (PROTOCOL.md section 10.3). ``Trainer(config, gauge=...)``
    takes the frozen diagonal reference scale OUT-OF-BAND, and applies it to
    the penalty, so what training OPTIMIZES is the declared metric
    ``sigma_max(S_out^-1 J^{q,excl} S_in)`` -- the identical object the
    offline audit measures. It is a constructor argument rather than a
    config key on purpose: a ``Gauge`` is tensors, not YAML, and it is
    measured from a phase-OFF checkpoint rather than declared. Its CONTENT
    still reaches the run identity, as the derived ``gauge_digest`` key
    injected into the hashed config -- two runs under different gauges
    optimize different objectives and can never share a run directory. On an
    OFF cell (``mu_j == 0``) the penalty branch is never evaluated, so the
    gauge is dropped and the key stays absent: inert for identity, exactly
    like ``k_probes``/``step_subsample`` there.

    Frozen shared representation (Wave D2b, PROTOCOL.md sections 14.9-14.11).
    ``Trainer(config, representation=...)`` takes a ``FrozenRepresentationArtifact``
    OUT-OF-BAND, exactly like ``gauge`` above and for the same reason: it is
    tensors and a manifest, not YAML, and it is pretrained once per
    ``(system, regime)`` stratum, shared by all six arms, never per run. When
    supplied, ``fit`` loads the encoder from the artifact instead of
    constructing and training one: the encoder is frozen (``eval()``,
    every parameter ``requires_grad=False``, zero entries in the optimizer's
    parameter groups), the EMA target machinery never updates it, and the
    preflight ``validate_shared_representation`` runs at construction time --
    a mismatched stratum, a tampered digest, or an unusable ``e_recon``
    refuses loudly before any model is built. The stratum it checks against
    comes from config keys ``stratum_system``/``stratum_regime`` (required
    only when a representation is supplied) -- named with a ``stratum_``
    prefix rather than bare ``system``/``regime`` because ``regime`` already
    names a koopman_ssm-only constructor key (its ZOH discretization regime)
    and would otherwise collide with the family-applicability check above.
    Its CONTENT still reaches the
    run identity, as the derived ``representation_digest`` key -- two runs
    loading different representations build different models and can never
    share a run directory. On a run with no representation the key stays
    ABSENT (never a null value), for the identical reason ``gauge_digest``
    does: only ``results_root`` is otherwise excluded from the hash, so an
    unconditional key would move every existing config_hash, therefore every
    run directory, therefore make every sentinel already on disk unreachable.

    BatchNorm-free assumption: encoders in this repo are plain MLPs; the EMA
    target's buffer-copy policy is declared in ``EMATarget`` (jepa.py) and is
    a no-op here. Runs on CPU, float32, by default -- data arrives as
    float64 numpy and is cast to float32 once, at ``_time_major``.
    """

    def __init__(
        self,
        config: dict,
        gauge: Gauge | None = None,
        representation: FrozenRepresentationArtifact | None = None,
    ) -> None:
        # lazy import -- see the TYPE_CHECKING guard at the top of this
        # module for why this cannot be a module-level import
        from sld.encoders.pretrain import representation_digest, validate_shared_representation

        missing = [k for k in _REQUIRED_KEYS if k not in config]
        if missing:
            raise ValueError(f"config must include {missing}")
        self.config = {**_DEFAULTS, **config}

        family = self.config["family"]
        if family not in _FAMILY_CONFIG_KEYS:
            raise ValueError(
                f"unknown family: {family!r}, expected one of {tuple(_FAMILY_CONFIG_KEYS)}"
            )
        # every model-defining key present must be applicable to the family:
        # applied or rejected, never silently ignored
        inapplicable = sorted((_MODEL_KEYS - _FAMILY_CONFIG_KEYS[family]) & set(config))
        if inapplicable:
            raise ValueError(
                f"config keys {inapplicable} are not applicable to family {family!r}"
            )
        budget_owned = sorted(_BUDGET_OWNED_KEYS & set(config))
        if budget_owned:
            raise ValueError(
                f"config keys {budget_owned} are capacity knobs owned by match_budget -- "
                "set target_params instead (params +/-5% is the fairness contract)"
            )
        derived = sorted(_DERIVED_OR_POLICY_KEYS & set(config))
        if derived:
            raise ValueError(
                f"config keys {derived} are derived or policy-owned (d_h follows from "
                "d_z/n_pairs; precision is the Trainer's declared CPU/float32 policy; "
                "gauge_digest is derived from the out-of-band gauge= argument; "
                "representation_digest is derived from the out-of-band representation= argument)"
            )
        # loop integers must be usable BEFORE any artifact is created: an
        # empty epoch loop would still write success-only artifacts and then
        # crash on the empty history (round 4)
        for key in ("epochs", "batch_episodes", "h_fr"):
            v = self.config[key]
            if not (isinstance(v, int) and not isinstance(v, bool) and v >= 1):
                raise ValueError(f"{key} must be an integer >= 1, got {v!r}")
        # family defaults are canonicalized INTO the config before hashing:
        # an omitted default and an explicitly-supplied default value are the
        # SAME model, so they must be the SAME run identity (round 5). An
        # explicit None means "use the default" and is normalized the same
        # way -- setdefault alone would hash null as a distinct identity
        # (round 6)
        def _resolve(key: str, default) -> None:
            if self.config.get(key) is None:
                self.config[key] = default

        # The chain is CLOSED (see its else): a family present in
        # _FAMILY_CONFIG_KEYS but absent here would never be canonicalized,
        # and "default omitted" vs "default written out" would hash as two
        # run identities for one and the same model -- a silent identity bug
        # that no green gate would catch.
        if family == "mlp":
            pass  # no family-specific keys, nothing to canonicalize
        elif family == "transformer":
            _resolve("n_heads", 2)
            _resolve("max_len", 512)
        elif family == "min_ssm":
            _resolve("delta_min", 1e-4)
            _resolve("delta_max", 10.0)
        elif family in STRUCTURED_FAMILIES:
            # ONE branch for mlp_k/transformer_k/ssm_k: delta is the shared
            # physical ZOH step (required, like koopman_ssm's), r the shared
            # lifted width (d_h = 2r, defaulting to the same 2x lifting
            # ratio koopman_ssm gets from n_pairs=d_z). Only the backbone's
            # own extras differ, and they are the SAME defaults their
            # unstructured counterparts use.
            if self.config.get("delta") is None:
                raise ValueError(f"family {family!r} requires delta in the config")
            _resolve("r", self.config["d_z"])
            if family == "transformer_k":
                _resolve("n_heads", 2)
                _resolve("max_len", 512)
            elif family == "ssm_k":
                _resolve("delta_min", 1e-4)
                _resolve("delta_max", 10.0)
        elif family == "koopman_ssm":
            # conditional requiredness is schema validation, not a fit-time
            # surprise (round 5): delta is koopman's physical ZOH step
            if self.config.get("delta") is None:
                raise ValueError("family 'koopman_ssm' requires delta in the config")
            _resolve("regime", "bilinear")
            _resolve("constrained", True)
            _resolve("n_pairs", self.config["d_z"])
            if self.config["regime"] == "selective":
                _resolve("delta_min", 1e-4)
                _resolve("delta_max", 10.0 * self.config["delta"])
        else:  # unreachable while every registered family has a branch above
            raise ValueError(
                f"no canonicalization branch for family {family!r} -- a family accepted "
                "by _FAMILY_CONFIG_KEYS must resolve its defaults here, or two configs "
                "describing the same model would hash to two run identities"
            )
        # selective-only knobs on a fixed regime would hash into the identity
        # while constructing identical dynamics (round 3); the check reads the
        # USER-supplied keys (canonicalization never adds them to fixed regimes)
        if family == "koopman_ssm" and self.config["regime"] != "selective":
            selective_only = sorted({"delta_min", "delta_max"} & set(config))
            if selective_only:
                raise ValueError(
                    f"config keys {selective_only} apply only to regime='selective'"
                )
        for key in _WEIGHT_KEYS:
            w = self.config[key]
            if not (isinstance(w, (int, float)) and math.isfinite(w) and w >= 0):
                raise ValueError(f"{key} must be finite and >= 0, got {w!r}")
        if self.config["w_tf"] <= 0 and self.config["w_fr"] <= 0:
            raise ValueError("at least one prediction objective (w_tf or w_fr) must be > 0")
        # mu_j: the total-Jacobian constraint factor shared by every family
        # in the factorial (objectives/jac_reg.py). Finite/>=0 like the other
        # thresholds (the finite-metrics law: a NaN threshold silently
        # disabling the penalty is a bug, not "off"). v0 scope (declared): the
        # penalty is evaluated on the TF-threaded states only, so it needs a
        # full-length TF thread to have anything to evaluate -- rejected at
        # construction rather than silently computing on an empty thread.
        mu_j = self.config["mu_j"]
        if not (isinstance(mu_j, (int, float)) and math.isfinite(mu_j) and mu_j >= 0):
            raise ValueError(f"mu_j must be finite and >= 0, got {mu_j!r}")
        if mu_j > 0 and self.config["w_tf"] <= 0:
            raise ValueError("mu_j > 0 requires w_tf > 0 -- the penalty needs the TF-threaded state")
        # k_probes / step_subsample: pre-registered mu_j estimator constants
        # (PROTOCOL.md section 5: k_probes=2, step_subsample=0.25), now
        # config-pilotable instead of hardcoded at TotalJacobianPenalty() --
        # defaults are UNCHANGED so a config that omits them reproduces the
        # old behavior exactly. Type/finiteness are checked here; the RANGE
        # bounds (k_probes >= 1, step_subsample in (0, 1]) are validated by
        # TotalJacobianPenalty.__init__ itself (jac_reg.py:66-69) -- checked
        # unconditionally of mu_j below, so a bad value is rejected at
        # construction regardless of whether the penalty ever runs (round 4
        # doctrine: schema errors caught before any artifact, not only on
        # the path that would use them), never a double source of truth for
        # the range check.
        k_probes = self.config["k_probes"]
        if not (isinstance(k_probes, int) and not isinstance(k_probes, bool)):
            raise ValueError(f"k_probes must be an integer, got {k_probes!r}")
        step_subsample = self.config["step_subsample"]
        if not (isinstance(step_subsample, (int, float)) and math.isfinite(step_subsample)):
            raise ValueError(f"step_subsample must be finite, got {step_subsample!r}")
        # PROTOCOL.md 17.1 (Amendment 8): grad_clip is an OPTIONAL config key,
        # deliberately ABSENT from _DEFAULTS -- _DEFAULTS is merged into every
        # config BEFORE hashing (self.config = {**_DEFAULTS, **config} above),
        # so a default here would move config_hash (therefore run_dir,
        # therefore every sentinel already on disk) for every run that has
        # ever existed. Absent: self.config never gains the key, so the
        # hashed dict and config_hash are byte-identical to before this
        # amendment. Present: validated (finite, strictly > 0 -- a guard that
        # cannot bite, or that silently disables itself on a bad value, is a
        # bug, not a lenient setting) and hashed automatically, since it is
        # already a key of self.config once the caller supplies it.
        if "grad_clip" in self.config:
            grad_clip = self.config["grad_clip"]
            if not (isinstance(grad_clip, (int, float)) and math.isfinite(grad_clip) and grad_clip > 0):
                raise ValueError(f"grad_clip must be finite and > 0, got {grad_clip!r}")
        # The frozen reference scale (PROTOCOL.md section 10.3) travels
        # OUT-OF-BAND, as this constructor argument, never as a config value:
        # it is a bundle of tensors, not YAML, and it is MEASURED from a
        # checkpoint rather than declared. It is applied to the penalty here
        # so the quantity training optimizes is the SAME metric-normalized
        # object the audit measures (jac_reg module docstring; the
        # elementwise-JVP and dense-Jacobian sides of that identity are
        # locked by tests). Passing no gauge keeps the RAW pre-amendment
        # estimator, bit-for-bit -- every existing caller and config is
        # unaffected.
        self.gauge = gauge if mu_j > 0 else None
        # RNG for the penalty is the GLOBAL one (generator=None): determinism
        # comes from set_seed(cfg["seed"]) like everywhere else in fit(); mu_j
        # itself is already part of the hashed config, so no cross-run leakage
        self._jac_penalty = TotalJacobianPenalty(
            k_probes=k_probes, step_subsample=step_subsample, gauge=self.gauge
        )
        if mu_j == 0:
            # k_probes/step_subsample only reach anything through the mu_j
            # penalty, which is entirely skipped when mu_j == 0 (l. 327
            # disabled-branch doctrine) -- canonicalize them to the shared
            # default so two mu_j=0 configs differing ONLY in these
            # otherwise-inert knobs share one run identity. Same technique
            # the family defaults use for an explicit None (l. 224): resolve
            # into self.config BEFORE hashing, so config.yaml and the hash
            # agree with each other and with what actually ran.
            #
            # The gauge is inert for EXACTLY the same reason, and is
            # canonicalized the same way -- by staying ABSENT from the hashed
            # config (``self.gauge`` is already None above, so it cannot
            # reach the penalty either). Absence rather than a null value is
            # the same convention _FAMILY_CONFIG_KEYS already uses for a key
            # that does not apply to this run, and it keeps every OFF cell's
            # identity bit-identical to what it was before the gauge existed.
            self.config["k_probes"] = _DEFAULTS["k_probes"]
            self.config["step_subsample"] = _DEFAULTS["step_subsample"]
        elif self.gauge is not None:
            # mu_j > 0 AND a gauge: the penalized object depends on this
            # gauge's CONTENT, so two runs under different gauges are
            # different runs and must not share a directory. The digest is
            # bit-stable across processes (see gauge_digest) and is the only
            # thing about the gauge that enters the identity.
            self.config["gauge_digest"] = gauge_digest(self.gauge)

        # The frozen shared representation (Wave D2b, PROTOCOL.md sections
        # 14.9-14.11) travels OUT-OF-BAND, exactly like the gauge above, and
        # for the same reason: it is tensors and a manifest, measured once
        # per stratum, not declared per run. Unlike the gauge it has no
        # on/off knob -- if it is supplied at all, it is always used, so the
        # preflight and the derived key below are unconditional on
        # `representation is not None`, never on another config value.
        self.representation = representation
        if representation is not None:
            # "stratum_system"/"stratum_regime" are ordinary caller-supplied
            # config keys (the doctrine already documented at the top of
            # this file: an unknown extra key the caller passes is kept
            # as-is and flows into the hash) -- required HERE, and only
            # here, so the preflight below has something to check the
            # artifact's own manifest against. A representation-less run
            # never needs them. Named with a "stratum_" prefix, NOT plain
            # "system"/"regime": "regime" already names a koopman_ssm-only
            # constructor key (its ZOH discretization regime, "bilinear" or
            # "selective", part of _FAMILY_CONFIG_KEYS/_MODEL_KEYS) -- a bare
            # "regime" key here would collide with that and be REJECTED by
            # the family-applicability check above for every other family
            # (caught by hand: `Trainer(cfg, representation=art)` on family
            # "mlp" raised "config keys ['regime'] are not applicable").
            if "stratum_system" not in self.config or "stratum_regime" not in self.config:
                raise ValueError(
                    "representation requires 'stratum_system' and 'stratum_regime' keys "
                    "in config, so the preflight (validate_shared_representation) can "
                    "check the artifact was pretrained for THIS run's stratum"
                )
            # preflight: digest tamper check, stratum match, e_recon
            # usability -- all three BEFORE any model is built or any batch
            # is drawn (round 4 doctrine: schema/artifact errors caught
            # before any artifact, never a fit-time surprise). Raises
            # ValueError, never coerces, never proceeds on a bad artifact.
            validate_shared_representation(
                representation,
                system=self.config["stratum_system"],
                regime=self.config["stratum_regime"],
            )
            # its CONTENT still reaches the run identity, as this derived
            # key -- two runs loading different representations build
            # different models and can never share a run directory. Stays
            # ABSENT (never a null value) when representation is None: only
            # results_root is otherwise excluded from the hash, so an
            # unconditional key would move every existing config_hash,
            # therefore every run directory, therefore make every sentinel
            # already on disk unreachable -- the exact trap gauge_digest
            # above already avoids.
            self.config["representation_digest"] = representation_digest(representation)

        # canonical config string: sort_keys=True makes it a pure function of
        # the resolved dict's content, not of insertion order; this EXACT
        # string is both what gets hashed and what is written to config.yaml.
        # results_root is EXCLUDED from the hashed identity: it is
        # infrastructure (where artifacts land), not science (what the run
        # is) -- adversarial review round 1.
        hashed = {k: v for k, v in self.config.items() if k != "results_root"}
        self._canonical_yaml = yaml.safe_dump(hashed, sort_keys=True)
        self.config_hash = hashlib.sha1(self._canonical_yaml.encode("utf-8")).hexdigest()[:8]
        # the FULL run identity also covers the dataset's generation spec
        # (adversarial review round 2): run_dir is resolved in fit(), once
        # the data -- and hence its fingerprint -- is known
        self.run_dir: Path | None = None
        self.data_hash: str | None = None

    def _composite_loss(
        self,
        encoder: nn.Module,
        predictor,
        ema: EMATarget,
        jepa: JEPALoss,
        x_bt: torch.Tensor,
        u_bt: torch.Tensor,
        lifted: bool,
        lift_fn,
        project_fn,
        h_fr: int,
        compute_jac_penalty: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """One composite loss evaluation: TF + FR + VICReg, on one episode batch.

        ``compute_jac_penalty=False`` (val split only, see ``fit``) treats
        the mu_j branch exactly like a zero-weighted one -- not evaluated at
        all, same doctrine as the ``w_fr == 0`` skip below -- regardless of
        ``cfg["mu_j"]``. This is what keeps validation from spending 23-38%
        of an epoch on a diagnostic used nowhere, and from consuming the
        GLOBAL RNG (``TotalJacobianPenalty``'s probes/subsampling, generator
        ``None``) and shifting the probes every subsequent epoch's TRAINING
        step draws.
        """
        cfg = self.config
        T = u_bt.shape[0]
        z = encoder(x_bt)  # (T+1, B, d_z) online
        z_tgt = ema(x_bt)  # (T+1, B, d_z) EMA target, no grad by construction

        # zero-weighted branches are NOT evaluated at all: a diverging FR
        # rollout must not poison a TF-only run through 0 * NaN (adversarial
        # review round 1), and skipping them also skips their compute. The
        # placeholder zero is EXPLICITLY float32: created under the restored
        # ambient dtype it would promote the whole loss accumulation to
        # float64 and change history low bits across ambients (round 7)
        zero = torch.zeros((), dtype=torch.float32)

        if cfg["w_tf"] > 0:
            # -- teacher-forced term: step over the TRUE prefix h_true_t = lift(z_t) --
            h_true = lift_fn(z[:T])  # (T, B, d_h)
            state = PredictorState()
            h_hats = []
            # jac_steps captures the INPUT (h_t, m_t, u_t) of every step BEFORE
            # it runs, entirely skipped when mu_j == 0 (doctrine: a disabled
            # branch is not evaluated, l.258 zero placeholder)
            jac_steps = [] if (cfg["mu_j"] > 0 and compute_jac_penalty) else None
            for t in range(T):
                if jac_steps is not None:
                    jac_steps.append((h_true[t], state.m, u_bt[t]))
                h_next, state = predictor.step(h_true[t], u_bt[t], state)
                h_hats.append(h_next)
            h_hat_tf = torch.stack(h_hats, dim=0)  # (T, B, d_h)
            pred_z_tf = project_fn(h_hat_tf)  # (T, B, d_z)
            tf_out = jepa(
                pred_z_tf,
                z_tgt[1:],
                h_hat=h_hat_tf if lifted else None,
                lift=lift_fn if lifted else None,
            )
            jac_pen = (
                self._jac_state_penalty(predictor, jac_steps) if jac_steps is not None else zero
            )
        else:
            tf_out = {"total": zero, "pred": zero, "consistency": zero}
            jac_pen = zero

        if cfg["w_fr"] > 0:
            # -- free-running term: rollout h_fr <= T steps from h0 = lift(z_0) --
            h0 = lift_fn(z[0])  # (B, d_h)
            h_hat_fr = predictor.rollout(h0, u_bt[:h_fr])  # (h_fr, B, d_h)
            pred_z_fr = project_fn(h_hat_fr)  # (h_fr, B, d_z)
            fr_out = jepa(
                pred_z_fr,
                z_tgt[1 : h_fr + 1],
                h_hat=h_hat_fr if lifted else None,
                lift=lift_fn if lifted else None,
            )
        else:
            fr_out = {"total": zero, "pred": zero, "consistency": zero}

        if cfg["w_var"] > 0 or cfg["w_cov"] > 0:
            # -- VICReg anti-collapse on the lifted online embeddings (every arm) --
            zl = lift_fn(z)  # (T+1, B, d_h); the losses flatten leading dims themselves
            var = variance_loss(zl) if cfg["w_var"] > 0 else zero
            cov = covariance_loss(zl) if cfg["w_cov"] > 0 else zero
        else:
            var = cov = zero

        total = (
            cfg["w_tf"] * tf_out["total"]
            + cfg["w_fr"] * fr_out["total"]
            + cfg["w_var"] * var
            + cfg["w_cov"] * cov
            + cfg["mu_j"] * jac_pen
        )
        # diagnostic breakdown for metrics.csv: TF+FR pred/consistency summed
        # unweighted, var/cov/jac_pen are the raw values (weights live in config.yaml)
        diag = {
            "pred": tf_out["pred"].item() + fr_out["pred"].item(),
            "consistency": tf_out["consistency"].item() + fr_out["consistency"].item(),
            "var": var.item(),
            "cov": cov.item(),
            "jac_pen": jac_pen.item(),
        }
        return total, diag

    def _jac_state_penalty(
        self, predictor, jac_steps: list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]]
    ) -> torch.Tensor:
        """Dispatch the mu_j penalty by presence of a threaded state ``m``.

        SECONDARY -- not primary thesis (jalon B2 demotion): this is the
        branch point of the retired mu_J penalty's training-side wiring
        (module CONTRACT NOTE above). Functional and tested, never a gate.

        Family-agnostic (``predictors.base`` doctrine): reads what each
        captured step ACTUALLY carries, never the family name. Every
        ``m_t is None`` across the thread -- true for ``mlp``/``koopman_ssm``,
        which never populate ``PredictorState.m`` -- uses the tensor-state
        path (``TotalJacobianPenalty.__call__``); any tensor ``m_t`` -- true
        for ``min_ssm``/``transformer`` past their first step -- uses the
        pytree path (``on_state_steps``). Accepted corner case: a T=1 TF
        thread on a state-threading family captures only the initial
        ``m=None`` and falls to the tensor path, missing the m-Jacobian
        entries for that single step -- a consequence of dispatching on the
        captured value rather than the family, not a bug.
        """
        if all(m is None for _, m, _ in jac_steps):
            q_seq = torch.stack([h for h, _, _ in jac_steps], dim=0)
            u_seq = torch.stack([u for _, _, u in jac_steps], dim=0)
            step_fn = lambda q, u: predictor.step(q, u, PredictorState())[0]  # noqa: E731
            return self._jac_penalty(step_fn, q_seq, u_seq)

        def step_fn(h, m, u):
            h_next, state = predictor.step(h, u, PredictorState(m=m))
            return h_next, state.m

        return self._jac_penalty.on_state_steps(step_fn, jac_steps)

    def resolve_run_dir(self, data: dict) -> Path:
        """Resolve (never create) ``self.run_dir`` from ``data``, and return it.

        Pulled out of ``fit`` so callers -- namely the ``sld-factorial``
        runner -- can compute the SAME run directory ``fit`` would use
        WITHOUT training anything, purely to check whether a matching run
        already exists on disk (resume detection). Writes nothing: no
        directory, no files. ``data`` without a ``meta`` block is rejected,
        exactly as ``fit`` rejects it (the dataset generation spec is part
        of the run identity -- REGLES section 6, adversarial review round 2).
        """
        if "meta" not in data:
            raise ValueError(
                "data must carry its build_dataset 'meta' block: the dataset "
                "generation spec is part of the run identity"
            )
        data_hash = data_fingerprint(data)
        self.data_hash = data_hash
        self.run_dir = (
            Path(self.config["results_root"])
            / self.config["exp_name"]
            / f"{self.config_hash}-{data_hash}"
        )
        return self.run_dir

    def fit(self, data: dict) -> dict:
        """Build the models FROM the config, then train on ``data``.

        A run is ``config + seed + dataset spec`` (REGLES section 6): ``fit``
        seeds the global RNG FIRST and only then constructs the encoder
        (``FrameMLPEncoder(d_x, d_z, encoder_hidden)``) and the
        parameter-matched predictor (``get_predictor(family, d_z, d_u,
        target_params, ...)``), so identical config + seed reconstructs
        identical initial weights -- models built by the caller with
        unseeded RNG would break run-identity (adversarial review round 1).
        The run directory is ``<results_root>/<exp_name>/<config_hash>-
        <data_hash>`` where ``data_hash`` fingerprints ``data["meta"]`` (the
        ``build_dataset`` generation spec: system/regime and their params,
        seed, dt, T, episode splits, observation maps) -- scientifically
        different datasets can never share a run identity (adversarial
        review round 2). ``data`` without a ``meta`` block is rejected.
        Declared scope: the fingerprint covers the generation SPEC; arrays
        mutated out-of-band without touching ``meta`` are outside it (the
        replay contract in ``sld.data`` is what ties spec to arrays).
        Encoder scope at v0: FRAME encoder only; the windowed-observation
        data path (sliding windows feeding ``WindowMLPEncoder``) arrives
        with Wave 6 and is NOT wired here.

        Returns the REGLES section 6 public contract:
        ``{"model", "history", "metrics", "paths"}``. Writes artifacts to
        ``self.run_dir``: ``config.yaml``, ``metrics.csv``, ``ckpt.pt``,
        ``audit.json`` (placeholder ``{}``, filled by the Wave 4 audit
        offline), ``figs/`` (empty). Any nonfinite train/val loss ABORTS
        the run with ``RuntimeError`` (finite-metrics law: NaN is a bug,
        never a result) -- no SUCCESS artifact (``ckpt.pt``, ``audit.json``,
        ``run_complete.json``) is written for an aborted run, and the
        offending metrics.csv row is never written either (metrics.csv only
        ever carries finite values). Instead (PROTOCOL.md section 11.7),
        ``run_dir / "failure.json"`` (``FAILURE_FILENAME``) is written,
        atomically, BEFORE the ``RuntimeError`` is raised: it is this run's
        own terminal, durable record of the divergence -- config_hash,
        data_hash, the failing epoch, which of train/val loss was
        non-finite and its value, and the exception class -- so the
        offending values are recoverable from failure.json even though they
        never reach metrics.csv. See ``FAILURE_FILENAME``'s module-level
        docstring for why it survives the stale-artifact cleanup below.

        Contract extension (adversarial review F9): ``paths["complete"]``
        points to ``run_dir / "run_complete.json"``, the run's completion
        sentinel -- ``{"config_hash", "data_hash", "epochs_completed",
        "final_train_loss", "final_val_loss", "ckpt_sha1", "metrics_rows"}``
        (M3, round 3: the last two BIND the sentinel to the artifacts it
        certifies -- ``ckpt_sha1`` is a streamed digest of ``ckpt.pt``
        (``_sha1_file``, never ``torch.load``), ``metrics_rows`` is the
        number of data rows written to ``metrics.csv`` -- so the factorial
        validator can recompute both from disk and refuse a substituted or
        truncated artifact instead of trusting names/existence alone). It is
        written LAST, after ``ckpt.pt`` and ``audit.json``, atomically (temp
        file + ``os.replace`` in the same directory). ``ckpt.pt``'s mere
        existence is NOT a completion signal -- it is written directly to
        its final path mid-``fit``, so a process kill between that write and
        the sentinel leaves a plausible but incomplete artifact set. Callers
        resuming or indexing a run (``sld-factorial``) must check the
        sentinel, never ``ckpt.pt`` alone. Any stale sentinel from a
        PREVIOUS run at this identity is removed up front, alongside
        ``ckpt.pt``/``audit.json``.

        Crash-resume (PROTOCOL.md section 14.7): every ``RESUME_CHECKPOINT_
        INTERVAL`` updates, ``fit`` atomically writes ``run_dir /
        RESUME_CHECKPOINT_FILENAME`` -- model, optimizer, scheduler, epoch
        counter, and torch/numpy/python RNG state. On entry, if that file
        exists and its ``config_hash`` matches this run's, ``fit`` loads it
        and continues from the next epoch, truncating ``metrics.csv`` to
        exactly the rows the checkpoint covers first; a MISMATCHED or
        corrupt/truncated file is refused with ``RuntimeError``, never
        adopted and never silently deleted. The bit-identity requirement:
        a run of N updates killed at N/2 and resumed produces a
        ``metrics.csv`` byte-identical to an uninterrupted run of N
        updates. ``ckpt.pt`` and ``run_complete.json`` keep their existing
        meanings unchanged -- a checkpoint is still not a completion
        signal. The resume checkpoint is deleted on successful completion
        (a finished run leaves no resume state) and is one of the stale
        artifacts a fresh start clears, alongside the new per-run
        ``train.log`` (config, ``config_hash``, run dir, budget, then one
        line per epoch).
        """
        cfg = self.config
        # dataset spec enters the run identity BEFORE anything is written
        self.resolve_run_dir(data)

        set_seed(cfg["seed"])  # seed BEFORE model construction (run identity)

        # the declared CPU/float32 policy is enforced AT CONSTRUCTION, not by
        # casting afterwards: init draws consume the RNG differently per
        # dtype, so float64-then-cast is a DIFFERENT seeded model than native
        # float32, and casting would be exactly the dtype narrowing the
        # bounds contract declares unsupported (adversarial review round 6)
        ambient = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            encoder = FrameMLPEncoder(cfg["d_x"], cfg["d_z"], tuple(cfg["encoder_hidden"]))
            if self.representation is not None:
                # Wave D2b: the trainer does NOT train this encoder -- load
                # the frozen weights from the artifact and freeze them here,
                # at the single point construction happens, so every
                # downstream reader (optimizer param list below, EMA target,
                # ckpt.pt's saved state) sees the SAME frozen object. eval()
                # and requires_grad=False are both set (neither alone is
                # trusted -- belt and suspenders, same doctrine EMATarget
                # already uses for its own stop-gradient target).
                encoder.load_state_dict(self.representation.encoder_state_dict)
                encoder.eval()
                for p in encoder.parameters():
                    p.requires_grad_(False)
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

        lifted = hasattr(predictor, "lift")
        lift_fn = predictor.lift if lifted else (lambda z: z)
        project_fn = predictor.project if lifted else (lambda h: h)

        ema = EMATarget(encoder, tau=cfg["tau_ema"])
        jepa = JEPALoss(lambda_c=cfg["lambda_c"])

        # a frozen encoder contributes ZERO members to the optimizer -- this
        # is the mutant that matters most (module docstring, Wave D2b): a
        # re-introduced `encoder.parameters()` here would silently start
        # training weights the run identity and the preflight both declare
        # frozen.
        if self.representation is not None:
            params = list(predictor.parameters())
        else:
            params = list(encoder.parameters()) + list(predictor.parameters())
        optimizer = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

        train_split, val_split = data["train"], data["val"]
        n_train = train_split["x"].shape[0]
        T = train_split["u"].shape[1]
        h_fr = min(cfg["h_fr"], T)
        val_idx = np.arange(val_split["x"].shape[0])

        self.run_dir.mkdir(parents=True, exist_ok=True)

        resume_path = self.run_dir / RESUME_CHECKPOINT_FILENAME
        complete_path = self.run_dir / RUN_COMPLETE_FILENAME
        # A resume checkpoint coexisting with a completion sentinel is
        # STALE, not a run to continue: the only way both exist together is
        # the narrow window between writing run_complete.json and deleting
        # the resume checkpoint at the very end of a successful fit (below)
        # being interrupted mid-way. Treated as a fresh start, which wipes
        # both -- resuming a run that has already certified itself complete
        # would silently re-extend a finished result.
        resuming = resume_path.exists() and not complete_path.exists()
        metrics_defect = None

        resume_state = None
        if resuming:
            # PROTOCOL.md section 14.7: identity MATCH -> continue; identity
            # MISMATCH or an unreadable/truncated file -> refuse loudly
            # (RuntimeError from _load_resume_checkpoint), never adopted and
            # never silently deleted.
            resume_state = _load_resume_checkpoint(resume_path, self.config_hash)
            # Incident 2026-08-11 -- see _metrics_csv_defect: a resume
            # checkpoint can be perfectly valid while metrics.csv, the OTHER
            # half of resume state, is not. Treated the same as a fresh
            # start below: the checkpoint alone cannot rebuild `history`, and
            # a run with no run_complete.json certifies nothing about its
            # metrics.csv either.
            metrics_defect = _metrics_csv_defect(
                self.run_dir / "metrics.csv", resume_state["epoch"] + 1
            )
            if metrics_defect is not None:
                resuming = False  # falls through to the fresh-start wipe below
                resume_state = None

        if resume_state is not None:
            encoder.load_state_dict(resume_state["encoder"])
            predictor.load_state_dict(resume_state["predictor"])
            ema.load_state_dict(resume_state["ema_target"])
            optimizer.load_state_dict(resume_state["optimizer"])
            scheduler.load_state_dict(resume_state["scheduler"])
            torch.set_rng_state(resume_state["torch_rng_state"])
            np.random.set_state(resume_state["numpy_rng_state"])
            random.setstate(resume_state["python_rng_state"])
            start_epoch = resume_state["epoch"] + 1
        else:
            # Every artifact this fit writes is cleared up front.  Besides
            # preventing stale success artifacts from certifying an aborted
            # rerun, this makes a validator-requested retrain recover if a
            # corrupt artifact is a directory: writing config.yaml/metrics.csv
            # would otherwise raise IsADirectoryError, as would recreating figs/
            # if it had become a regular file.  Remove each stale path however it
            # exists: shutil.rmtree for a directory, unlink otherwise.
            # FAILURE_FILENAME is DELIBERATELY absent from this list (PROTOCOL.md
            # section 11.7): a divergence is a terminal result, not a partial
            # artifact to be wiped and silently retried -- see FAILURE_FILENAME's
            # docstring above. RESUME_CHECKPOINT_FILENAME and TRAIN_LOG_FILENAME
            # join this list (section 14.7): a killed run's resume state and log
            # must not survive into the next attempt at the same identity.
            for stale in (
                "config.yaml",
                "metrics.csv",
                "ckpt.pt",
                "audit.json",
                RUN_COMPLETE_FILENAME,
                "figs",
                RESUME_CHECKPOINT_FILENAME,
                TRAIN_LOG_FILENAME,
            ):
                stale_path = self.run_dir / stale
                if stale_path.is_dir():
                    shutil.rmtree(stale_path)
                else:
                    stale_path.unlink(missing_ok=True)
            (self.run_dir / "figs").mkdir()
            start_epoch = 0

        config_path = self.run_dir / "config.yaml"
        config_path.write_text(self._canonical_yaml)

        # Per-run log (section 14.7). A logger unique to this run directory,
        # with its OWN FileHandler removed in `finally`: a shared/global
        # logger would leak every subsequent fit() call's lines into every
        # earlier run's train.log, since repeated fit() calls happen
        # in-process (this test suite is the proof).
        logger = logging.getLogger(f"sld.training.trainer.run.{self.run_dir}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        log_handler = logging.FileHandler(
            self.run_dir / TRAIN_LOG_FILENAME, mode="a" if resuming else "w"
        )
        log_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(log_handler)

        history = {k: [] for k in ("epoch", "train_loss", "val_loss", *_HISTORY_DIAGNOSTIC_KEYS)}
        metrics_path = self.run_dir / "metrics.csv"

        try:
            logger.info(
                "config_hash=%s data_hash=%s run_dir=%s", self.config_hash, self.data_hash, self.run_dir
            )
            logger.info(
                "budget: epochs=%d (resume checkpoint every %d updates)",
                cfg["epochs"],
                RESUME_CHECKPOINT_INTERVAL,
            )
            logger.info("config:\n%s", self._canonical_yaml.rstrip())

            if metrics_defect is not None:
                # ALSO on stdout, not only in train.log: `logger.propagate` is
                # False just above and this logger carries a FileHandler only,
                # so a campaign operator watching a shard would never see it.
                # A restart from zero changes WHAT THE CELL MEASURED -- 8000
                # fresh updates instead of a resume -- and the final artifacts
                # are indistinguishable, so it must not be discoverable only by
                # opening 60 train.log files after the fact.
                print(
                    f"    RESTART from epoch 0: metrics.csv unusable ({metrics_defect})",
                    flush=True,
                )
                logger.warning(
                    "metrics.csv unusable (%s) and this run has no %s -- it certifies nothing: "
                    "restarting this run from epoch 0 (previous artifacts wiped)",
                    metrics_defect,
                    RUN_COMPLETE_FILENAME,
                )

            if resuming:
                # metrics.csv may hold rows the checkpoint does not cover
                # (possibly a torn last row) -- truncate to exactly what it
                # certifies, then rebuild `history` from what remains so
                # epochs_completed/final_*_loss/best_*_loss reflect the
                # WHOLE run, not just the resumed tail.
                _truncate_metrics_csv(metrics_path, start_epoch)
                with open(metrics_path, newline="") as f:
                    for row in csv.DictReader(f):
                        history["epoch"].append(int(row["epoch"]))
                        history["train_loss"].append(float(row["train_loss"]))
                        history["val_loss"].append(float(row["val_loss"]))
                        for key in _HISTORY_DIAGNOSTIC_KEYS:
                            history[key].append(None)  # never recomputed, see Wave 4 note above
                logger.info(
                    "resuming from epoch %d (%d rows kept in metrics.csv)", start_epoch, start_epoch
                )
                metrics_mode = "a"
            else:
                metrics_mode = "w"

            t_start = time.monotonic()
            with open(metrics_path, metrics_mode, newline="") as f:
                writer = csv.writer(f)
                if not resuming:
                    writer.writerow(list(METRICS_HEADER))

                epoch_iter = tqdm(
                    range(start_epoch, cfg["epochs"]),
                    total=cfg["epochs"],
                    initial=start_epoch,
                    file=sys.stderr,
                    disable=None,  # auto: disabled whenever stderr is not a TTY
                    desc=f"{cfg['exp_name']} {self.config_hash}",
                )
                for epoch in epoch_iter:
                    # one sampled train batch per epoch (v0 scale, declared above)
                    batch_size = min(cfg["batch_episodes"], n_train)
                    idx = np.random.choice(n_train, size=batch_size, replace=False)
                    x_bt, u_bt = _time_major(train_split, idx)

                    optimizer.zero_grad()
                    train_total, diag = self._composite_loss(
                        encoder, predictor, ema, jepa, x_bt, u_bt, lifted, lift_fn, project_fn, h_fr
                    )
                    train_total.backward()
                    # PROTOCOL.md 17.1: the global gradient norm is measured
                    # PRE-clip, unconditionally (clipped or not -- a guard
                    # whose bite is not measured would be a silent knob), via
                    # the separate _global_grad_norm helper -- never by
                    # calling clip_grad_norm_ itself. Absent grad_clip: the
                    # guard's own clipping call is never made, so every
                    # gradient reaching optimizer.step() stays bitwise
                    # untouched (a structural equivalence with pre-amendment
                    # runs, not merely a numerical one).
                    grad_norm = _global_grad_norm(params)
                    if "grad_clip" in cfg:
                        torch.nn.utils.clip_grad_norm_(params, max_norm=cfg["grad_clip"])
                    lr_used = optimizer.param_groups[0]["lr"]  # rate actually used for this step
                    optimizer.step()
                    if self.representation is None:
                        ema.update(encoder)  # EMA target refreshed after EVERY optimizer step
                    # else: Wave D2b -- a frozen encoder is never fed to
                    # ema.update(); `ema.target` was deep-copied from the
                    # already-frozen encoder at construction and is left
                    # exactly there, never touched again.
                    scheduler.step()

                    with torch.no_grad():
                        x_v, u_v = _time_major(val_split, val_idx)
                        # mu_j is NOT evaluated on val (see METRICS_HEADER note
                        # and _composite_loss docstring): unused downstream and
                        # 23-38% of epoch time when it was
                        val_total, _ = self._composite_loss(
                            encoder,
                            predictor,
                            ema,
                            jepa,
                            x_v,
                            u_v,
                            lifted,
                            lift_fn,
                            project_fn,
                            h_fr,
                            compute_jac_penalty=False,
                        )

                    train_loss, val_loss = float(train_total.item()), float(val_total.item())
                    # finite-metrics law (REGLES section 6): a nonfinite loss is a
                    # bug, never a result -- abort loudly instead of writing a
                    # completed-looking run (adversarial review round 1)
                    if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
                        # PROTOCOL.md section 11.7: the divergence is reported,
                        # not just raised -- failure.json is this run's terminal
                        # sentinel, written atomically (temp file + os.replace,
                        # same discipline as RUN_COMPLETE_FILENAME) BEFORE the
                        # abort. The offending values are recorded HERE, as
                        # strings ("nan"/"inf"/"-inf") rather than raw floats --
                        # json.dumps would otherwise emit the bare NaN/Infinity
                        # literals Python's own json module accepts but strict
                        # JSON does not -- since this row is never written to
                        # metrics.csv (metrics.csv only ever carries finite
                        # values; this is the recovery path for it, chosen over
                        # writing the nonfinite row into metrics.csv itself).
                        nonfinite_fields = [
                            name
                            for name, value in (("train_loss", train_loss), ("val_loss", val_loss))
                            if not math.isfinite(value)
                        ]
                        failure = {
                            "config_hash": self.config_hash,
                            "data_hash": self.data_hash,
                            "epoch": epoch,
                            "reason": "non-finite loss",
                            "nonfinite_fields": nonfinite_fields,
                            "train_loss": str(train_loss),
                            "val_loss": str(val_loss),
                            "exception_class": "RuntimeError",
                        }
                        failure_path = self.run_dir / FAILURE_FILENAME
                        failure_tmp = failure_path.with_name(failure_path.name + ".tmp")
                        failure_tmp.write_text(json.dumps(failure))
                        os.replace(failure_tmp, failure_path)
                        raise RuntimeError(
                            f"nonfinite loss at epoch {epoch}: train={train_loss}, "
                            f"val={val_loss} -- run aborted, artifacts incomplete "
                            f"(terminal failure recorded at {failure_path})"
                        )
                    history["epoch"].append(epoch)
                    history["train_loss"].append(train_loss)
                    history["val_loss"].append(val_loss)
                    for key in _HISTORY_DIAGNOSTIC_KEYS:
                        history[key].append(None)  # arrives with the Wave 4 audit, never omitted

                    writer.writerow(
                        [
                            epoch,
                            train_loss,
                            val_loss,
                            lr_used,
                            diag["pred"],
                            diag["consistency"],
                            diag["var"],
                            diag["cov"],
                            diag["jac_pen"],
                            grad_norm,
                        ]
                    )
                    f.flush()  # live progress feed: today's buffering hides it entirely

                    elapsed = time.monotonic() - t_start
                    progress = (epoch + 1 - start_epoch) / max(cfg["epochs"] - start_epoch, 1)
                    eta = elapsed / progress - elapsed if progress > 0 else float("nan")
                    logger.info(
                        "epoch=%d train_loss=%.6g val_loss=%.6g lr=%.6g elapsed=%.2fs ETA=%.2fs",
                        epoch,
                        train_loss,
                        val_loss,
                        lr_used,
                        elapsed,
                        eta,
                    )

                    # crash-resume checkpoint (section 14.7): every
                    # RESUME_CHECKPOINT_INTERVAL updates, strictly AFTER this
                    # epoch's metrics row is written -- so the checkpoint's
                    # epoch count and metrics.csv's row count always agree,
                    # which is what makes the truncate-on-resume above exact.
                    if (epoch + 1) % RESUME_CHECKPOINT_INTERVAL == 0:
                        _save_resume_checkpoint(
                            resume_path,
                            self.config_hash,
                            epoch,
                            encoder,
                            predictor,
                            ema,
                            optimizer,
                            scheduler,
                        )
        finally:
            logger.removeHandler(log_handler)
            log_handler.close()

        ckpt_path = self.run_dir / "ckpt.pt"
        torch.save(
            {
                "encoder": encoder.state_dict(),
                "predictor": predictor.state_dict(),
                "ema_target": ema.state_dict(),
                "config": cfg,
            },
            ckpt_path,
        )

        audit_path = self.run_dir / "audit.json"
        # placeholder: filled by the Wave 4 offline audit, never computed here
        audit_path.write_text(json.dumps({}))

        # completion sentinel (F9): written LAST, strictly after ckpt.pt and
        # audit.json above -- its presence, not ckpt.pt's, is the public
        # "this run finished" signal. Atomic write: a temp file in the SAME
        # directory (same filesystem -- os.replace is then atomic on POSIX)
        # renamed onto the final name, so a kill mid-write never leaves a
        # half-written sentinel that reads as complete.
        complete_path = self.run_dir / RUN_COMPLETE_FILENAME
        completion = {
            "config_hash": self.config_hash,
            "data_hash": self.data_hash,
            "epochs_completed": len(history["epoch"]),
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            # M3 (round 3): binds the sentinel to the artifacts it certifies
            # -- ckpt_sha1 streams ckpt.pt (just written above) through
            # SHA1, metrics_rows counts exactly one row per epoch (this loop
            # writes exactly that). Both are recomputed from disk, not
            # trusted, by the factorial validator.
            "ckpt_sha1": _sha1_file(ckpt_path),
            "metrics_rows": len(history["epoch"]),
        }
        tmp_path = complete_path.with_name(complete_path.name + ".tmp")
        tmp_path.write_text(json.dumps(completion))
        os.replace(tmp_path, complete_path)

        # section 14.7: a finished run leaves no resume state. Deleted LAST,
        # strictly after the completion sentinel above -- the narrow window
        # this leaves (sentinel written, resume checkpoint not yet deleted)
        # is exactly the race `resuming`'s definition above accounts for.
        resume_path.unlink(missing_ok=True)

        paths = {
            "config": str(config_path.resolve()),
            "metrics_csv": str(metrics_path.resolve()),
            "audit": str(audit_path.resolve()),
            "ckpt": str(ckpt_path.resolve()),
            "figs": str((self.run_dir / "figs").resolve()),
            "complete": str(complete_path.resolve()),
        }
        metrics = {
            "final_train_loss": history["train_loss"][-1],
            "best_train_loss": min(history["train_loss"]),
            "final_val_loss": history["val_loss"][-1],
            "best_val_loss": min(history["val_loss"]),
            "seed": int(cfg["seed"]),
            "config_hash": self.config_hash,
        }
        return {
            "model": {"encoder": encoder, "predictor": predictor, "ema_target": ema},
            "history": history,
            "metrics": metrics,
            "paths": paths,
        }
