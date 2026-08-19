"""Free-running evaluation path (Wave D3a): from a trained run and its frozen
shared representation to the ``e_h`` curve in TRUE-STATE coordinates, the two
trivial baselines, the per-episode VPT and the positive-control verdict
(PROTOCOL.md sections 12 and 14).

This module NEVER reimplements a normalized error, a baseline, a VPT reducer
or the gate arithmetic -- every one of those already exists, once, in
``sld.audit.rollout`` and is imported and called here. This module's own job
is exactly two things:

1. Run the rollout GENUINELY free (PROTOCOL.md's rule D6): after the initial
   condition, no ground truth and no future
   embedding reaches the predictor -- not as input, not as state, not through
   a cache. ``evaluate_free_rollout`` calls ``predictor.rollout(h0, u_seq)``
   exactly ONCE; every downstream array (the model's ``e_h``, the per-episode
   curves the VPT is computed from) is a pure function of that one call's
   output. ``u_seq`` carries the dataset's own action sequence -- an
   exogenous, pre-recorded control signal, never the state being predicted --
   the same object training's own free-running term (``Trainer._composite_loss``)
   already threads through ``predictor.rollout`` un-audited-differently.
2. Wire the FROZEN readout and FROZEN ``sigma_train`` from the
   ``FrozenRepresentationArtifact`` into those primitives -- never a readout
   or a scale refit on the evaluation split -- and refuse an artifact whose
   digest does not match the digest the run recorded at train time
   (``evaluate_run_free_rollout``).

This file gained a second half (the gate PERSISTENCE + CAMPAIGN section
below): the verdict ``evaluate_free_rollout``/``evaluate_run_free_rollout``
compute was, until then, always calculated and immediately discarded -- no
``gate.json``, no per-run persistence, no campaign, nothing indexed. The
split is structural, not just a comment: everything ABOVE this point is pure
computation (no ``json``/``os``/``csv``/``yaml`` touched), everything BELOW
reads or writes disk.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from sld.audit._common import require_cpu_float64_module
from sld.audit.rollout import (
    PositiveControlResult,
    evaluate_positive_control,
    linear_extrapolation_rollout,
    normalized_state_error,
    persistence_rollout,
    vpt,
)
from sld.encoders.pretrain import FrozenRepresentationArtifact, apply_readout, representation_digest
from sld.encoders.pretrain import validate_shared_representation
from sld.experiments import protocol as protocol_mod
from sld.experiments.factorial import (
    _build_dataset_for,
    _parse_shard,
    _preflight_resolve_and_check_collisions,
    _resolve_stratum_representations,
    _shard_cells,
    _validate_completion_sentinel,
    expand,
)
from sld.experiments.loading import load_run


class RolloutDivergence(ValueError):
    """The model's own free rollout (``model_pred_x``) left the finite domain
    during evaluation -- PROTOCOL.md section 15.3's ``divergence (evaluation)``
    terminal class.

    Typed at the ONE site that knows which tensor produced the non-finite
    values (right after ``model_pred_x`` is decoded, before it ever reaches
    ``normalized_state_error``), so "the model diverged" no longer depends on
    string-matching that function's message -- which also fires, with the
    same substring, for a corrupted ``target`` or ``sigma_train``, neither of
    which is a model divergence (section 15.3's own motivation: probe round 1
    conflated the two with an ad hoc ``"non-finite" in str(exc)`` filter).

    Attributes:
        first_step: the first free-running step (1-indexed, matching
            ``e_h``'s own ``h = 1 .. H`` convention), over ANY episode or
            coordinate, at which ``model_pred_x`` is non-finite.
    """

    def __init__(self, first_step: int, message: str | None = None):
        self.first_step = first_step
        super().__init__(
            message or f"model rollout diverged (non-finite) at free-running step {first_step}"
        )


# Index convention for one evaluation episode's true-state trajectory
# (z_true, shape (B, T+1, d)) -- PINNED here, once, because it is an easy
# off-by-one trap:
#   index 0       -> x_{-1}  (one step before the rollout's initial condition;
#                              feeds ONLY the linear baseline, never the model)
#   index 1       -> x_0     (the rollout's initial condition, h=0; encoded
#                              from the OBSERVATION at this same index)
#   index 2..H+1  -> x_1..x_H (the H targets the model/baselines are scored
#                              against; e_h's index 0 is x_1, matching
#                              normalized_state_error's own convention)
# A split needs T+1 >= _LOOKBACK + 1 + H physical steps; anything shorter is
# a refusal (PROTOCOL.md section 12.3 forbids a silent min(H, T)).
_LOOKBACK = 1
_T0 = 1


@dataclass(frozen=True)
class FreeRolloutResult:
    """Everything Wave D3a's evaluation path produces for one run, one held-out
    split, one horizon -- the ONE object the gate, the VPT and any future
    figure all read from (never a second, independently recomputed array).

    Attributes:
        e_h: shape ``(H,)``, the model's normalized state error, aggregated
            over episodes (PROTOCOL.md section 12.1) -- what
            ``positive_control`` was computed from.
        persistence_e_h, linear_e_h: shape ``(H,)``, the two trivial
            baselines, same episodes/coordinates/``sigma_train`` as ``e_h``.
        per_episode_e_h: shape ``(B, H)``, the model's normalized error with
            NO episode aggregation -- the single source ``vpt_value`` below
            was reduced from, and the source ``e_h`` is itself derivable
            from (RMS-combine over axis 0 reproduces ``e_h`` exactly).
        model_pred_x: shape ``(B, H, d)``, the free rollout decoded to
            true-state coordinates through the FROZEN readout -- what
            ``per_episode_e_h``/``e_h`` were computed from, kept for a
            future figure so it is read once, not rebuilt.
        vpt_value, vpt_status: shape ``(B,)`` -- PROTOCOL.md section 14.5's
            per-episode VPT, ``inf``/``"censored"`` kept, never fabricated.
        vpt_median: median of ``vpt_value`` (``inf`` entries included, per
            section 14.5 -- no mean is taken at D, and no censored value is
            filtered before the median).
        vpt_censored: count of ``"censored"`` episodes.
        horizon: ``H``, the exact number of free-running steps scored.
        positive_control: PROTOCOL.md section 12.3's gate, computed from
            ``e_h``/``persistence_e_h``/``linear_e_h`` above and nothing else.
    """

    e_h: np.ndarray
    persistence_e_h: np.ndarray
    linear_e_h: np.ndarray
    per_episode_e_h: np.ndarray
    model_pred_x: np.ndarray
    vpt_value: np.ndarray
    vpt_status: np.ndarray
    vpt_median: float
    vpt_censored: int
    horizon: int
    positive_control: PositiveControlResult


def _lift_project(predictor: nn.Module):
    """Same rule ``Trainer._composite_loss`` already applies (trainer.py):
    a lifted family (``koopman_ssm`` and the three structured arms) exposes
    ``lift``/``project``; every other family is identity in both directions.
    """
    lifted = hasattr(predictor, "lift")
    lift_fn = predictor.lift if lifted else (lambda z: z)
    project_fn = predictor.project if lifted else (lambda h: h)
    return lift_fn, project_fn


def evaluate_free_rollout(
    encoder: nn.Module,
    predictor: nn.Module,
    eval_split: dict,
    *,
    W: np.ndarray,
    sigma_train: np.ndarray,
    horizon: int,
    tau: float,
    h_gate: int,
    gamma: float,
    tau_1: float,
) -> FreeRolloutResult:
    """The free-running evaluation core (PROTOCOL.md sections 12, 14.5).

    Args:
        encoder, predictor: already-resolved, CPU FLOAT64 modules -- audit
            doctrine (``sld.audit._common``, REGLES section 5): this
            evaluation runs offline, in float64, never in the training-time
            float32 forward pass (float32 would cap the exact-ZOH oracle
            test at ~1e-7, not the 1e-12 this evaluation requires). Enforced
            loudly here via ``require_cpu_float64_module``, the SAME helper
            every other audit entry point in this repo already uses --
            never a silent cast, so a float32 model reaching this function
            is a caller bug, not a value this function repairs.
            ``evaluate_run_free_rollout`` is the caller that makes the
            float64 copy, from a run's own (float32) checkpoint.
        eval_split: a ``{"x", "u", "z_true"}`` dict in the ``build_dataset``
            shape (``x``/``z_true``: ``(B, T+1, d)``, ``u``: ``(B, T, d_u)``)
            -- the HELD-OUT split; the caller decides which one.
        W: the FROZEN readout matrix (``artifact.readout.W``) -- never a
            readout refit on ``eval_split``.
        sigma_train: the FROZEN train-split scale (``artifact.sigma_train``)
            -- never recomputed on ``eval_split``.
        horizon: ``H``, exact -- a split shorter than required is refused.
        tau, h_gate, gamma, tau_1: PROTOCOL.md sections 12.2/12.3's own
            thresholds, always supplied by the caller (never defaulted here,
            same discipline ``vpt``/``evaluate_positive_control`` already
            use for a pre-registered constant).

    Returns:
        A ``FreeRolloutResult``.

    Raises:
        ValueError: ``horizon < 1``; ``eval_split`` missing a required key;
            fewer physical steps or fewer actions than ``horizon`` needs
            (refused outright, never reduced via ``min(horizon, T)``); the
            free rollout or either baseline produces a non-finite value
            (propagated from ``normalized_state_error``'s own refusal).
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    for key in ("x", "u", "z_true"):
        if key not in eval_split:
            raise ValueError(f"eval_split must include {key!r}")

    x = np.asarray(eval_split["x"])
    u = np.asarray(eval_split["u"])
    z_true = np.asarray(eval_split["z_true"], dtype=np.float64)

    required_steps = _LOOKBACK + 1 + horizon  # indices 0 .. _T0+horizon
    if z_true.shape[1] < required_steps:
        raise ValueError(
            f"eval_split has {z_true.shape[1]} physical steps per episode, fewer than "
            f"the {required_steps} required for horizon={horizon} (index 0 = x_-1, "
            f"index {_T0} = x_0, indices {_T0 + 1}..{_T0 + horizon} = targets) -- "
            "refused outright, never silently reduced via min(horizon, T)"
        )
    required_actions = _T0 + horizon
    if u.shape[1] < required_actions:
        raise ValueError(
            f"eval_split has {u.shape[1]} actions per episode, fewer than the "
            f"{required_actions} required for horizon={horizon} -- refused outright, "
            "never silently reduced"
        )

    require_cpu_float64_module(encoder, "encoder")
    require_cpu_float64_module(predictor, "predictor")
    encoder.eval()
    predictor.eval()
    lift_fn, project_fn = _lift_project(predictor)

    x0_obs = torch.as_tensor(x[:, _T0, :], dtype=torch.float64)
    u_seq = torch.as_tensor(
        u[:, _T0 : _T0 + horizon, :], dtype=torch.float64
    ).transpose(0, 1).contiguous()  # (H, B, d_u), time-major -- rollout's own contract

    with torch.no_grad():
        z0 = encoder(x0_obs)  # (B, d_z) -- the ONLY encoder call; nothing after t0 is ever encoded
        h0 = lift_fn(z0)  # (B, d_h)
        h_hat = predictor.rollout(h0, u_seq)  # (H, B, d_h) -- the ONE free-rollout call, D6
        pred_z = project_fn(h_hat)  # (H, B, d_z), READ-ONLY projection (base.py contract)

    pred_x = apply_readout(pred_z.numpy(), W)  # (H, B, d) -- FROZEN W, already float64
    model_pred_x = np.ascontiguousarray(np.transpose(pred_x, (1, 0, 2)))  # (B, H, d)

    if not np.isfinite(model_pred_x).all():
        # ONLY model_pred_x is checked here -- target/sigma_train non-finite
        # values are still caught, unchanged, by normalized_state_error's own
        # refusal below (a plain ValueError, never RolloutDivergence): this
        # separation IS PROTOCOL.md section 15.3's correction, typed rather
        # than left to prose matching.
        bad = ~np.isfinite(model_pred_x).all(axis=(0, 2))  # (H,), True at a diverged step
        first_step = int(np.argmax(bad)) + 1  # 1-indexed, matches e_h's h=1..H
        raise RolloutDivergence(
            first_step,
            f"model_pred_x contains non-finite values (NaN/inf), first at "
            f"free-running step {first_step}",
        )

    x_minus_1 = z_true[:, 0, :]
    x0_true = z_true[:, _T0, :]
    target = z_true[:, _T0 + 1 : _T0 + 1 + horizon, :]  # (B, H, d)

    sigma = np.asarray(sigma_train, dtype=np.float64)  # FROZEN sigma_train, never recomputed here

    # The ONE per-episode source (requirement 7): B calls to the ONE existing
    # normalized_state_error, each on a single-episode slice of the SAME
    # model_pred_x/target/sigma computed above -- never a second rollout.
    B = model_pred_x.shape[0]
    per_episode_e_h = np.stack(
        [
            normalized_state_error(model_pred_x[b : b + 1], target[b : b + 1], sigma)
            for b in range(B)
        ],
        axis=0,
    )  # (B, H)
    e_h = normalized_state_error(model_pred_x, target, sigma)  # (H,), same three inputs, aggregated

    persistence_pred = persistence_rollout(x0_true, horizon)
    linear_pred = linear_extrapolation_rollout(x_minus_1, x0_true, horizon)
    persistence_e_h = normalized_state_error(persistence_pred, target, sigma)
    linear_e_h = normalized_state_error(linear_pred, target, sigma)

    # Requirement 7, structurally: the VPT is reduced from the SAME array the
    # result returns, not from a second one built alongside it. This is not a
    # convention -- `per_episode_e_h` is passed here and returned unchanged,
    # so a caller can re-derive the VPT from the returned table and get the
    # same answer. A second, separately-built array would be invisible to any
    # test that only checks the two agree in shape.
    vpt_result = vpt(per_episode_e_h, tau)
    vpt_median = float(np.median(vpt_result.value))  # inf entries included, per section 14.5
    vpt_censored = int(np.sum(vpt_result.status == "censored"))

    positive_control = evaluate_positive_control(
        e_h, persistence_e_h, linear_e_h, h_gate=h_gate, gamma=gamma, tau_1=tau_1
    )

    return FreeRolloutResult(
        e_h=e_h,
        persistence_e_h=persistence_e_h,
        linear_e_h=linear_e_h,
        per_episode_e_h=per_episode_e_h,
        model_pred_x=model_pred_x,
        vpt_value=vpt_result.value,
        vpt_status=vpt_result.status,
        vpt_median=vpt_median,
        vpt_censored=vpt_censored,
        horizon=horizon,
        positive_control=positive_control,
    )


def load_run_double(
    run_dir: str | Path, representation: FrozenRepresentationArtifact
) -> tuple[nn.Module, nn.Module, dict]:
    """Load ``run_dir``'s checkpoint, refuse an artifact this run was not
    trained with, and return CPU FLOAT64 copies of its encoder/predictor plus
    the run's own resolved config -- the shared preamble ``evaluate_run_free_rollout``
    below needs, and that ``sld.experiments.oscillator_gold``'s D3 scoring
    pass (PROTOCOL.md section 19.2's spectral/action bounds, which need the
    raw predictor rather than a rollout) reuses rather than reimplements (R2:
    one implementation of "is this artifact the one this run trained with").

    Loads ``{"encoder", "predictor", "config"}`` via ``load_run`` and refuses,
    BEFORE any use, an artifact that is not the one this run actually trained
    with: ``representation``'s own tamper/stratum preflight
    (``validate_shared_representation``, against the run's recorded
    ``stratum_system``/``stratum_regime``) AND a direct digest comparison
    against the run's own recorded ``config["representation_digest"]`` -- two
    distinct checks, because a same-stratum artifact from a DIFFERENT
    training run (different seed, different data) passes the first and fails
    the second.

    ``load_run`` reconstructs the checkpoint's own float32 (training-time)
    modules; this function makes the float64 COPY audit doctrine requires
    (``sld.audit._common``) -- a fresh ``copy.deepcopy(...).double()``, never
    mutating the checkpoint's own objects in place.

    Raises:
        ValueError: the run's config has no recorded ``representation_digest``
            (a representation-less run has nothing to evaluate against here);
            ``representation``'s own digest does not match it; or
            ``validate_shared_representation`` refuses (tamper/stratum
            mismatch/unusable ``e_recon``).
    """
    run = load_run(run_dir)
    cfg = run["config"]

    recorded_digest = cfg.get("representation_digest")
    if recorded_digest is None:
        raise ValueError(
            f"run at {run_dir} has no recorded representation_digest -- it was not "
            "trained with a frozen shared representation, so there is nothing for "
            "this representation-based evaluation path to check it against"
        )
    computed_digest = representation_digest(representation)
    if computed_digest != recorded_digest:
        raise ValueError(
            f"representation digest mismatch: the supplied artifact hashes to "
            f"{computed_digest!r}, but run at {run_dir} recorded "
            f"{recorded_digest!r} -- refusing to evaluate this run against a "
            "representation it was not trained with"
        )
    validate_shared_representation(
        representation, system=cfg["stratum_system"], regime=cfg["stratum_regime"]
    )

    encoder64 = copy.deepcopy(run["encoder"]).double().eval()
    predictor64 = copy.deepcopy(run["predictor"]).double().eval()
    return encoder64, predictor64, cfg


def evaluate_run_free_rollout(
    run_dir: str | Path,
    representation: FrozenRepresentationArtifact,
    eval_split: dict,
    *,
    horizon: int,
    tau: float,
    h_gate: int,
    gamma: float,
    tau_1: float,
) -> FreeRolloutResult:
    """``evaluate_free_rollout``, resolved from a finished run directory.

    The run-vs-artifact preamble (``load_run`` + the digest/stratum refusal +
    the float64 copy) is ``load_run_double`` above; this function is now
    exactly that plus the ``evaluate_free_rollout`` call.

    Raises:
        ValueError: the run's config has no recorded ``representation_digest``
            (a representation-less run has nothing to evaluate against here);
            ``representation``'s own digest does not match it; or
            ``validate_shared_representation`` refuses (tamper/stratum
            mismatch/unusable ``e_recon``).
    """
    encoder64, predictor64, _cfg = load_run_double(run_dir, representation)

    return evaluate_free_rollout(
        encoder64,
        predictor64,
        eval_split,
        W=representation.readout.W,
        sigma_train=representation.sigma_train,
        horizon=horizon,
        tau=tau,
        h_gate=h_gate,
        gamma=gamma,
        tau_1=tau_1,
    )


# ==============================================================================
# PERSISTENCE (I/O starts here). Nothing above this line touches disk; every
# function below does. ``gate.json`` is this module's own ``run_complete.json``
# -- same discipline, deliberately copied: written LAST, atomically (temp file
# + ``os.replace`` in the SAME directory), and read back through a validator
# that trusts nothing about the bytes on disk (``sld.training.trainer``'s
# ``RUN_COMPLETE_FILENAME``/``_validate_completion_sentinel`` doctrine, reused
# rather than reinvented -- R2).
# ==============================================================================

_SCHEMA_VERSION = "gate/1"
_STATUSES = frozenset({"learns", "does_not_learn", "divergence"})

# closed schemas, exactly like factorial._SENTINEL_KEYS: a missing OR an
# extra key both fail validation, never silently tolerated.
_TOP_KEYS = frozenset(
    {"schema", "identity", "digests", "split", "protocol", "measures", "status", "first_step"}
)
_IDENTITY_KEYS = frozenset({"cell_id", "arm", "seed", "stratum_system", "stratum_regime", "run_dir"})
_DIGEST_KEYS = frozenset({"config_hash", "data_hash", "representation_digest"})
# every threshold the verdict was ACTUALLY computed from, plus what tau_1 was
# derived from (PROTOCOL.md section 15.1) -- so a later change to any one of
# them is detectable against a gate.json already on disk (read_gate_artifact's
# ``expected_protocol``).
_PROTOCOL_KEYS = frozenset(
    {"h_gate", "gamma", "tau_1", "tau_vpt", "horizon", "gamma_1", "e_1_pers", "e_recon", "attainable"}
)
_MEASURE_KEYS = frozenset(
    {
        "e_h",
        "persistence_e_h",
        "linear_e_h",
        "e_1",
        "e_h_gate",
        "m_low",
        "m_high",
        "lower_bound",
        "upper_bound",
        "near_miss",
        "vpt_median",
        "vpt_censored",
        "per_episode_vpt",
    }
)


def _read_run_digests(run_dir: Path) -> dict:
    """``{"config_hash", "data_hash", "representation_digest"}``, RE-READ from
    ``run_dir``'s own artifacts on disk -- never from caller memory.
    ``config_hash``/``data_hash`` come from a freshly re-validated completion
    sentinel (``sld.experiments.factorial._validate_completion_sentinel``,
    the exact same validator ``sld-factorial`` itself trusts, reused rather
    than reimplemented); ``representation_digest`` comes from the run's own
    ``config.yaml``. A caller's in-memory bookkeeping could be stale (an
    artifact rebuilt between resolving the cell and writing a verdict); the
    bytes on disk cannot lie about themselves this way.

    Shared by ``write_gate_artifact`` below and
    ``sld.experiments.oscillator_gold``'s D3 ``d3_gold.json`` writer -- R2:
    one implementation of "what digests does this run's own disk state
    carry", not a second copy that could drift.

    Raises:
        ValueError: ``run_dir`` has no valid completion sentinel, or
            ``config.yaml`` is unreadable.
    """
    sentinel = _validate_completion_sentinel(run_dir)
    if sentinel is None:
        raise ValueError(
            f"run at {run_dir} has no valid completion sentinel (run_complete.json) -- "
            "refusing to read digests for a run that is not verified complete"
        )
    try:
        run_config = yaml.safe_load((run_dir / "config.yaml").read_text())
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"run at {run_dir}: config.yaml is unreadable ({exc!r})") from exc
    return {
        "config_hash": sentinel["config_hash"],
        "data_hash": sentinel["data_hash"],
        "representation_digest": run_config.get("representation_digest") if isinstance(run_config, dict) else None,
    }


def gate_filename(split: str) -> str:
    """The artifact filename for ``split``. The gate's own verdict keeps
    ``gate.json``; a 16.4 selection reading on val lands in ``gate_val.json``
    beside it, never over it."""
    return "gate.json" if split == "test" else f"gate_{split}.json"


def write_gate_artifact(
    run_dir: str | Path,
    result: FreeRolloutResult | RolloutDivergence,
    *,
    identity: dict,
    protocol: dict,
    split: str,
) -> Path:
    """Persist one run's positive-control verdict to ``run_dir/gate.json``.

    ``result`` is either the successful ``FreeRolloutResult``
    ``evaluate_run_free_rollout`` returned, or the ``RolloutDivergence`` it
    raised -- the caller catches that exception and hands the exception
    OBJECT straight to this function rather than re-deriving a status from a
    caught message, so ``first_step`` is carried through structurally.

    ``identity`` (``cell_id``/``arm``/``seed``/``stratum_system``/
    ``stratum_regime``/``run_dir``) is an opaque echo of whatever the caller
    (the campaign) already resolved for this cell -- it is NOT re-derived
    here, because none of it (which grid cell this is, which backbone/
    structure arm) is recoverable from the run's own artifacts alone.

    ``digests`` (``config_hash``/``data_hash``/``representation_digest``),
    by contrast, are NEVER taken from the caller: they are re-read from the
    run's own artifacts on disk right here -- ``config_hash``/``data_hash``
    from a freshly re-validated completion sentinel
    (``sld.experiments.factorial._validate_completion_sentinel``, the exact
    same validator ``sld-factorial`` itself trusts, reused rather than
    reimplemented), ``representation_digest`` from the run's own
    ``config.yaml``. A caller's in-memory bookkeeping could be stale (an
    artifact rebuilt between resolving the cell and writing the gate); the
    bytes on disk cannot lie about themselves this way.

    Raises:
        ValueError: ``identity`` or ``protocol`` does not carry exactly its
            required keys; a ``protocol`` threshold is non-finite; ``run_dir``
            has no valid completion sentinel (refuses to gate an unfinished
            or unverified run); ``config.yaml`` is unreadable; ``result`` is
            neither a ``FreeRolloutResult`` nor a ``RolloutDivergence``.
    """
    run_dir = Path(run_dir)

    if set(identity) != _IDENTITY_KEYS:
        raise ValueError(f"identity must carry exactly {sorted(_IDENTITY_KEYS)}, got {sorted(identity)}")
    if set(protocol) != _PROTOCOL_KEYS:
        raise ValueError(f"protocol must carry exactly {sorted(_PROTOCOL_KEYS)}, got {sorted(protocol)}")
    for key, value in protocol.items():
        if key == "attainable":
            continue
        if not math.isfinite(float(value)):
            raise ValueError(f"protocol[{key!r}] = {value!r} must be finite")

    digests = _read_run_digests(run_dir)

    if isinstance(result, RolloutDivergence):
        status = "divergence"
        first_step = result.first_step
        measures = None
    elif isinstance(result, FreeRolloutResult):
        pc = result.positive_control
        status = pc.verdict
        first_step = None
        measures = {
            "e_h": result.e_h.tolist(),
            "persistence_e_h": result.persistence_e_h.tolist(),
            "linear_e_h": result.linear_e_h.tolist(),
            "e_1": float(result.e_h[0]),
            "e_h_gate": float(result.e_h[protocol["h_gate"] - 1]),
            "m_low": pc.m_low,
            "m_high": pc.m_high,
            "lower_bound": pc.lower_bound,
            "upper_bound": pc.upper_bound,
            "near_miss": pc.near_miss,
            "vpt_median": result.vpt_median,
            "vpt_censored": result.vpt_censored,
            "per_episode_vpt": result.vpt_value.tolist(),
        }
    else:
        raise ValueError(
            f"result must be a FreeRolloutResult or a RolloutDivergence, got {type(result)!r}"
        )

    payload = {
        "schema": _SCHEMA_VERSION,
        "identity": identity,
        "digests": digests,
        "split": split,
        "protocol": protocol,
        "measures": measures,
        "status": status,
        "first_step": first_step,
    }

    # The FILENAME carries the split, so a selection reading can never be
    # mistaken for a gate verdict: 15.5/16.4 put the gate on test and every
    # SELECTION on val, and `sld.viz.ingest.load_gate` globs `gate.json`
    # only. A val reading therefore lands beside the verdict, never over it.
    gate_path = run_dir / gate_filename(split)
    tmp_path = gate_path.with_name(gate_path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload))
    os.replace(tmp_path, gate_path)
    return gate_path


def read_gate_artifact(
    run_dir: str | Path,
    *,
    expected_protocol: dict | None = None,
    expected_digests: dict | None = None,
    split: str = "test",
) -> dict:
    """Load and VALIDATE ``run_dir/gate.json`` -- trusts nothing about the
    bytes on disk, same doctrine as ``factorial._validate_completion_sentinel``.

    ``expected_protocol``/``expected_digests`` are the caller's OWN current
    values (e.g. the live ``protocol.py`` constants, or a freshly recomputed
    representation digest); any key present there that disagrees with what
    is recorded in ``gate.json`` is a refusal naming that key -- this is
    what makes "a threshold changed after the run" or "the representation
    moved since this verdict was written" detectable rather than silently
    stale.

    Raises:
        ValueError: ``gate.json`` is missing, corrupt/unreadable, not schema
            ``"gate/1"``, missing a required key at any level, carries an
            unknown ``status``, carries a non-finite value where one is
            required (NaN is refused everywhere; the VPT fields additionally
            permit ``inf`` -- a censored VPT is a legitimate value, per
            PROTOCOL.md section 14.5), is internally inconsistent (a
            ``"divergence"`` status with non-null ``measures``, or a non-
            ``"divergence"`` status with a non-null ``first_step``), or
            disagrees with a caller-supplied ``expected_protocol``/
            ``expected_digests`` entry.
    """
    run_dir = Path(run_dir)
    path = run_dir / gate_filename(split)
    if not path.exists():
        raise ValueError(f"no {path.name} at {path}")
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeError) as exc:
        raise ValueError(f"gate.json at {path} is corrupt or unreadable ({exc!r})") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"gate.json at {path} is not a JSON object")

    if payload.get("schema") != _SCHEMA_VERSION:
        raise ValueError(
            f"gate.json at {path} has schema {payload.get('schema')!r}, expected {_SCHEMA_VERSION!r}"
        )
    missing_top = _TOP_KEYS - set(payload)
    if missing_top:
        raise ValueError(f"gate.json at {path} is missing top-level keys {sorted(missing_top)}")

    identity = payload["identity"]
    if not isinstance(identity, dict) or set(identity) != _IDENTITY_KEYS:
        raise ValueError(f"gate.json at {path}: identity must carry exactly {sorted(_IDENTITY_KEYS)}")

    digests = payload["digests"]
    if not isinstance(digests, dict) or set(digests) != _DIGEST_KEYS:
        raise ValueError(f"gate.json at {path}: digests must carry exactly {sorted(_DIGEST_KEYS)}")

    protocol = payload["protocol"]
    if not isinstance(protocol, dict) or set(protocol) != _PROTOCOL_KEYS:
        raise ValueError(f"gate.json at {path}: protocol must carry exactly {sorted(_PROTOCOL_KEYS)}")
    for key, value in protocol.items():
        if key == "attainable":
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"gate.json at {path}: protocol[{key!r}] = {value!r} is not a finite number")

    status = payload["status"]
    if status not in _STATUSES:
        raise ValueError(f"gate.json at {path}: unknown status {status!r}")

    first_step = payload["first_step"]
    measures = payload["measures"]
    if status == "divergence":
        if measures is not None:
            raise ValueError(f"gate.json at {path}: status is 'divergence' but measures is not null")
        if not isinstance(first_step, int) or isinstance(first_step, bool) or first_step < 1:
            raise ValueError(
                f"gate.json at {path}: status is 'divergence', first_step must be a positive "
                f"int, got {first_step!r}"
            )
    else:
        if first_step is not None:
            raise ValueError(
                f"gate.json at {path}: status is {status!r} but first_step is not null "
                f"(got {first_step!r})"
            )
        if not isinstance(measures, dict) or set(measures) != _MEASURE_KEYS:
            raise ValueError(f"gate.json at {path}: measures must carry exactly {sorted(_MEASURE_KEYS)}")
        for key in ("e_h", "persistence_e_h", "linear_e_h"):
            values = measures[key]
            if not isinstance(values, list) or not values:
                raise ValueError(f"gate.json at {path}: measures[{key!r}] must be a non-empty list")
            if not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                for v in values
            ):
                raise ValueError(f"gate.json at {path}: measures[{key!r}] contains a non-finite value")
        for key in ("e_1", "e_h_gate", "m_low", "m_high"):
            value = measures[key]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"gate.json at {path}: measures[{key!r}] = {value!r} is not a finite number")
        for key in ("lower_bound", "upper_bound", "near_miss"):
            if not isinstance(measures[key], bool):
                raise ValueError(f"gate.json at {path}: measures[{key!r}] must be a boolean")
        vpt_median = measures["vpt_median"]
        if (
            not isinstance(vpt_median, (int, float))
            or isinstance(vpt_median, bool)
            or math.isnan(vpt_median)
        ):
            raise ValueError(f"gate.json at {path}: measures['vpt_median'] = {vpt_median!r} is NaN or invalid")
        vpt_censored = measures["vpt_censored"]
        if not isinstance(vpt_censored, int) or isinstance(vpt_censored, bool) or vpt_censored < 0:
            raise ValueError(
                f"gate.json at {path}: measures['vpt_censored'] must be a non-negative int, "
                f"got {vpt_censored!r}"
            )
        per_episode_vpt = measures["per_episode_vpt"]
        if not isinstance(per_episode_vpt, list) or not per_episode_vpt:
            raise ValueError(f"gate.json at {path}: measures['per_episode_vpt'] must be a non-empty list")
        if any(
            not isinstance(v, (int, float)) or isinstance(v, bool) or math.isnan(v)
            for v in per_episode_vpt
        ):
            raise ValueError(f"gate.json at {path}: measures['per_episode_vpt'] contains NaN")

    if expected_protocol is not None:
        for key, expected_value in expected_protocol.items():
            if key not in _PROTOCOL_KEYS:
                raise ValueError(f"expected_protocol names unknown threshold {key!r}")
            if protocol[key] != expected_value:
                raise ValueError(
                    f"gate.json at {path}: protocol threshold {key!r} changed since this run's "
                    f"gate was written -- recorded {protocol[key]!r}, expected {expected_value!r}"
                )

    if expected_digests is not None:
        for key, expected_value in expected_digests.items():
            if key not in _DIGEST_KEYS:
                raise ValueError(f"expected_digests names unknown digest {key!r}")
            if digests[key] != expected_value:
                raise ValueError(
                    f"gate.json at {path}: digest {key!r} changed since this run's gate was "
                    f"written -- recorded {digests[key]!r}, expected {expected_value!r}"
                )

    return payload


# ------------------------------------------------------------------------
# The campaign: the same experiment YAML sld-factorial trains from, walked
# again -- read-only over the trained runs, never a second training path.
# ------------------------------------------------------------------------


def _gate_cell_status(run_dir: Path, split: str = "test") -> tuple[str, str, str]:
    """``(status, m_low, m_high)`` for one cell's ``run_dir``, read from
    ``gate.json`` alone -- mirrors ``factorial._cell_status``'s discipline:
    read from DISK, never from what this process happened to compute, so a
    shard that never touched this cell still reports another shard's
    already-written verdict correctly."""
    gate_path = Path(run_dir) / gate_filename(split)
    if not gate_path.exists():
        return "missing", "", ""
    try:
        payload = read_gate_artifact(run_dir, split=split)
    except ValueError:
        # present but untrustworthy -- distinct from both a valid verdict
        # and a cell that was never gated at all (same M9 doctrine
        # factorial._cell_status applies to an invalid run_complete.json)
        return "invalid_gate", "", ""
    status = payload["status"]
    measures = payload.get("measures")
    if not measures:
        return status, "", ""
    return status, str(measures["m_low"]), str(measures["m_high"])


def _write_gate_index(results_root: Path, exp_name: str, cells: list[dict], split: str = "test") -> None:
    """Regenerate ``results_root/exp_name/gate_index.csv`` from the full,
    resolved cell list -- ``factorial._write_index``'s own contract, copied:
    ALWAYS the full expected grid (never just this shard's slice), atomic
    write (temp file + ``os.replace`` in the same directory). A cell this
    invocation never touched, or never trained at all, appears with status
    ``"missing"`` rather than being silently absent from the file.
    """
    index_dir = results_root / exp_name
    index_dir.mkdir(parents=True, exist_ok=True)
    final_path = index_dir / ("gate_index.csv" if split == "test" else f"gate_index_{split}.csv")
    tmp_path = final_path.with_name(final_path.name + ".tmp")
    with open(tmp_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_dir", "regime", "family", "structure", "seed", "status", "m_low", "m_high"])
        for cell in cells:
            run_dir = cell["run_dir"]
            status, m_low, m_high = _gate_cell_status(run_dir, split)
            writer.writerow(
                [run_dir, cell["dataset"], cell["family"], cell["structure"], cell["seed"], status, m_low, m_high]
            )
    os.replace(tmp_path, final_path)


def _evaluate_gate_cell(
    cell: dict,
    datasets: dict[str, dict],
    representations: dict[str, FrozenRepresentationArtifact],
    results_root: Path,
    n: int,
    split: str = "test",
) -> dict:
    """Gate one cell: refuse loudly if its run is not complete, evaluate the
    free rollout on ``split``, write the matching artifact, and return a small
    summary dict.

    ``split`` defaults to ``"test"`` and every GATE reading keeps it there:
    PROTOCOL.md 15.5/16.4 fix "the gate scores on test; anything that
    *selects* reads val". 23.3's per-arm rate selection is such a selection,
    so it asks for ``"val"`` explicitly and the SAME
    ``evaluate_run_free_rollout`` call is made with the split swapped -- 16.4
    already declared that change mechanical ("no new measurement is
    defined"), and it is the identical criterion ``probe.py::_val_reading``
    computes for round 3. The artifact lands in ``gate_val.json``, never in
    ``gate.json``, so no val reading can ever be read back as a verdict.

    NEVER raises for a merely-incomplete or diverged cell -- both are
    reported, by name, and the campaign loop continues (``factorial.main``'s
    own F17d discipline: one bad cell must not silently drop every other
    cell's already-computed verdict from the index).
    """
    run_dir = cell["run_dir"]
    label = (
        f"[{cell['index'] + 1}/{n}] {cell['dataset']} {cell['backbone']} "
        f"{cell['structure']} {cell['family']} seed={cell['seed']}"
    )
    sentinel = _validate_completion_sentinel(run_dir)
    if sentinel is None:
        print(f"{label} -> {run_dir} (REFUSED: no valid run_complete.json -- run is not complete)")
        return {"index": cell["index"], "run_dir": str(run_dir), "status": "missing"}

    representation = representations[cell["dataset"]]
    eval_split = datasets[cell["dataset"]][split]
    e_1_pers = representation.manifest["e_1_pers"]
    e_recon = representation.e_recon
    tau_1_value = protocol_mod.tau_1(e_1_pers)
    attainable = protocol_mod.tau1_validity(e_recon, e_1_pers)["attainable"]

    protocol = {
        "h_gate": protocol_mod.H_GATE,
        "gamma": protocol_mod.GAMMA,
        "tau_1": tau_1_value,
        "tau_vpt": protocol_mod.TAU_VPT,
        # no separate "evaluation horizon" constant is declared anywhere in
        # this repo -- H_GATE is scored at exactly, never a longer horizon
        # padded on top of it (ponytail: the smallest horizon that satisfies
        # evaluate_positive_control's own length requirement).
        "horizon": protocol_mod.H_GATE,
        "gamma_1": protocol_mod.GAMMA_1,
        "e_1_pers": e_1_pers,
        "e_recon": e_recon,
        "attainable": attainable,
    }
    identity = {
        "cell_id": cell["index"],
        "arm": {"backbone": cell["backbone"], "structure": cell["structure"], "family": cell["family"]},
        "seed": cell["seed"],
        "stratum_system": cell["config"]["stratum_system"],
        "stratum_regime": cell["config"]["stratum_regime"],
        "run_dir": str(run_dir.relative_to(results_root)),
    }

    try:
        result = evaluate_run_free_rollout(
            run_dir,
            representation,
            eval_split,
            horizon=protocol_mod.H_GATE,
            tau=protocol_mod.TAU_VPT,
            h_gate=protocol_mod.H_GATE,
            gamma=protocol_mod.GAMMA,
            tau_1=tau_1_value,
        )
    except RolloutDivergence as exc:
        write_gate_artifact(run_dir, exc, identity=identity, protocol=protocol, split=split)
        print(f"{label} -> {run_dir} (DIVERGENCE at free-running step {exc.first_step})")
        return {"index": cell["index"], "run_dir": str(run_dir), "status": "divergence"}

    gate_path = write_gate_artifact(run_dir, result, identity=identity, protocol=protocol, split=split)
    print(f"{label} -> {run_dir} (OK, verdict={result.positive_control.verdict})")
    return {
        "index": cell["index"],
        "run_dir": str(run_dir),
        "status": result.positive_control.verdict,
        "gate_path": str(gate_path),
    }


def evaluate_gate_campaign(
    config_path: str | Path, *, shard: tuple[int, int] | None = None, split: str = "test"
) -> list[dict]:
    """Score every cell of ``config_path``'s grid against its already-trained
    run, writing one ``gate.json`` per cell plus one campaign-wide
    ``gate_index.csv``.

    Reuses ``sld.experiments.factorial.expand`` to build the cell list --
    never a parallel expansion that could drift from what ``sld-factorial``
    itself trained (R2) -- and the SAME global preflight
    (``_preflight_resolve_and_check_collisions``) over the FULL grid, before
    ``shard`` narrows which cells THIS invocation evaluates. ``shard`` never
    narrows the expansion, the preflight, or ``gate_index.csv``'s coverage --
    only the evaluation loop, exactly like ``sld-factorial --shard``
    (PROTOCOL.md section 14.7).

    Args:
        config_path: the experiment YAML -- the SAME file (and schema)
            ``sld-factorial --config`` trained the grid from.
        shard: ``(i, k)``, 1-indexed shard ``i`` of ``k``, or ``None`` to
            evaluate every cell.

    Returns:
        One summary dict per evaluated cell (``run_cells`` order): at least
        ``{"index", "run_dir", "status"}``, plus ``"gate_path"`` when a
        ``gate.json`` was written. ``status`` is one of ``"learns"``,
        ``"does_not_learn"``, ``"divergence"``, or ``"missing"`` (the run at
        this cell has no valid completion sentinel -- reported here, never
        silently dropped from the list, so nothing downstream can aggregate
        over ``run_cells`` and mistake a missing seed for one that was
        evaluated and passed).

    Raises:
        ValueError: the config carries no ``representation:`` block (the
            gate always scores against the frozen shared representation a
            run was trained with); anything ``expand``/the preflight/
            ``evaluate_run_free_rollout`` itself refuses (a digest mismatch,
            an insufficient split, a config schema error) -- these are
            campaign-wide identity/data problems, not a per-seed fluke, so
            they abort the whole campaign rather than being swallowed
            per-cell.
    """
    config_path = Path(config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    cells = expand(config, dry_run=False)
    if config.get("representation") is None:
        raise ValueError(
            f"{config_path}: evaluate_gate_campaign requires a `representation:` block -- "
            "the positive-control gate always scores a run against the frozen shared "
            "representation it was trained with (PROTOCOL.md sections 14.9-14.11)"
        )

    exp_name = config["exp_name"]
    results_root = Path(config.get("base", {}).get("results_root", "results"))
    datasets = {name: _build_dataset_for(cfg) for name, cfg in config["datasets"].items()}
    representations = _resolve_stratum_representations(config, results_root, exp_name, datasets)

    # global preflight over the FULL grid (F8/F17, mirrored from
    # factorial.main): every cell's run_dir resolved and collision-checked
    # BEFORE any evaluation, and BEFORE sharding narrows the loop below.
    _preflight_resolve_and_check_collisions(cells, datasets, representations)

    run_cells = _shard_cells(cells, shard)
    n = len(cells)
    results = [
        _evaluate_gate_cell(cell, datasets, representations, results_root, n, split)
        for cell in run_cells
    ]

    _write_gate_index(results_root, exp_name, cells, split)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sld-gate")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--shard",
        type=_parse_shard,
        default=None,
        metavar="i/k",
        help=(
            "score only cells i-1::k of the grid (1<=i<=k); expansion and the collision "
            "preflight stay global, same convention as sld-factorial --shard "
            "(PROTOCOL.md section 14.7)."
        ),
    )
    parser.add_argument(
        "--split",
        choices=("test", "val"),
        default="test",
        help=(
            "which split to score. test (default) is the GATE reading; val is a "
            "SELECTION reading (PROTOCOL.md 15.5/16.4, 23.3's per-arm rate argmin) "
            "and writes gate_val.json, never gate.json."
        ),
    )
    args = parser.parse_args(argv)
    results = evaluate_gate_campaign(args.config, shard=args.shard, split=args.split)
    any_failed = any(r["status"] in ("missing", "divergence") for r in results)
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
