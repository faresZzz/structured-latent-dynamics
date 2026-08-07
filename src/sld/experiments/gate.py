"""Free-running evaluation path (Wave D3a): from a trained run and its frozen
shared representation to the ``e_h`` curve in TRUE-STATE coordinates, the two
trivial baselines, the per-episode VPT and the positive-control verdict
(PROTOCOL.md sections 12 and 14).

This module NEVER reimplements a normalized error, a baseline, a VPT reducer
or the gate arithmetic -- every one of those already exists, once, in
``sld.audit.rollout`` and is imported and called here. This module's own job
is exactly two things:

1. Run the rollout GENUINELY free (PROTOCOL.md's rule D6, phaseD-apprentissage-
   plan.md): after the initial condition, no ground truth and no future
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
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
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
# (z_true, shape (B, T+1, d)) -- PINNED here, once, because it is the trap
# the wave brief names explicitly:
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
            test at ~1e-7, not the 1e-12 the wave brief requires). Enforced
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

    Loads ``{"encoder", "predictor", "config"}`` via ``load_run`` and refuses,
    BEFORE any rollout, an artifact that is not the one this run actually
    trained with: ``representation``'s own tamper/stratum preflight
    (``validate_shared_representation``, against the run's recorded
    ``stratum_system``/``stratum_regime``) AND a direct digest comparison
    against the run's own recorded ``config["representation_digest"]`` -- two
    distinct checks, because a same-stratum artifact from a DIFFERENT
    training run (different seed, different data) passes the first and fails
    the second.

    ``load_run`` reconstructs the checkpoint's own float32 (training-time)
    modules; this function makes the float64 COPY audit doctrine requires
    (``sld.audit._common``) -- a fresh ``copy.deepcopy(...).double()``, never
    mutating the checkpoint's own objects in place -- before handing them to
    ``evaluate_free_rollout``.

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
