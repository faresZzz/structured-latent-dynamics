"""PROTOCOL.md sections 2 and 5: LR sweep, mu_j calibration, and the bounded
phase-4 consistency check, all under the ``pendulum_core_selection``
namespace.

This module is the missing piece of infrastructure that runs those two
sections literally: ``sld-factorial`` (``experiments/factorial.py``) cannot
do it -- it expects a CLOSED single-value-per-arm schema and rejects a
multi-candidate sweep -- so this module builds ``Trainer`` configs directly
in Python, one candidate at a time.

Four ordered phases per family (PROTOCOL section 2):

1. **Phase OFF** -- sweep ``lr in {1e-4, 3e-4, 1e-3}`` on the OFF cell
   (``mu_j=0``), 200 optimizer updates, seed 0, both regimes ->
   ``LR_off(family)``.
2. **Initial calibration** -- PROTOCOL section 5 steps 1-6 at
   ``LR_off(family)``, OU regime only, 200 optimizer updates ->
   ``mu_j(family)``.
3. **Phase ON** -- the same sweep procedure as phase 1, on the ON cell at
   ``mu_j(family)`` fixed -> ``LR_on(family)``.
4. **Bounded consistency check** -- if ``LR_on == LR_off``, done. Otherwise
   re-measure the section-5 proxy at ``(LR_on, mu_j)``; if outside the
   band, up to two cycles of {recalibrate mu_j at the fixed LR -> re-sweep
   phase ON -> re-measure if the LR moved}, else ``"selection not
   converged"``.

Two post-hoc measurements this repo does not otherwise compute are the
core primitives here:

- ``_measure_fr_val_loss``: free-running validation loss at ``h_fr=5``
  (PROTOCOL section 2 point 4's selection metric). Honesty declaration:
  ``sld.experiments.loading.load_run`` does not reload ``ckpt.pt``'s EMA
  target (it returns only ``{"encoder", "predictor", "config"}``), so this
  uses the ONLINE encoder itself as the regression target -- a declared
  post-hoc simplification of the training-time EMA-target loss, not a
  hidden discrepancy. Plain MSE in projection space (``JEPALoss``'s
  ``pred`` term with no lifted-space consistency term), computed over the
  WHOLE val split, never a single batch.
- ``_measure_sigma_max_median``: the RAW post-hoc audit (PROTOCOL section
  5 step 3, superseded as the DECISION quantity by section 10.3) -- threads
  ``predictor.step`` over the TRUE prefix on the val split (mirroring
  ``Trainer._composite_loss``'s TF branch exactly), capturing
  ``(h_t, m_t, u_t)`` before every step, and takes the median of
  ``sigma_max(J^q_t)`` over all of them: ``sld.audit.jacobian.total_jacobian``
  when ``m_t`` is absent or fixed-size (``mlp``, ``koopman_ssm``, and
  ``min_ssm`` whenever its state is populated), and
  ``augmented_jacobian_excluding_identity`` for ``transformer`` alone (the
  only family with growing memory, PROTOCOL section 5's declared scope).
  It is kept as the un-normalized COMPARISON quantity only; nothing in the
  selection path reads it.
- ``estimate_gauge`` + ``measure_sigma_max_median_normalized``: the DECLARED
  metric (PROTOCOL section 10.3 amendment), median of
  ``sigma_max(S_out^-1 . J_t^{q,excl} . S_in)``. ``S`` is estimated ONCE per
  family, on the RETAINED phase-OFF / OU / seed-0 checkpoint, BEFORE the
  mu_j sweep, and then frozen: the same ``Gauge`` object is threaded through
  the whole sweep, the refinement point, the phase-4 re-measurements and both
  bounded-iteration cycles (``run_family_selection`` -> ``_calibrate_mu_j``/
  ``_measure_proxy_at`` -> ``_measure_mu_j_candidate``). It is never
  re-estimated from a penalized run -- a per-candidate gauge would be a
  MOVING gauge, and the declared invariance would be vacuous.
  The same frozen object is ALSO passed to ``Trainer`` for every ON-cell
  run this module launches (``_fit_and_load(..., gauge)``), which is what
  makes "applied identically at training time and at audit time"
  (section 10.3) true rather than aspirational: without it the penalty
  would optimize the raw Jacobian while the calibration graded the
  normalized one.

Every run this module trains is published in ``results/<results_root>/
pendulum_core_selection/selection_report.{json,csv}`` (``run_all_families``)
-- full trace, never summarized away. ``apply_selection_to_pendulum_core``
patches ``pendulum_core.yaml``'s ``lr``/``mu_j`` nulls from a finished
report; it is a SEPARATE function, never called at import time or by
``run_all_families`` itself -- the supervisor calls it only after manually
reviewing the report.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
from pathlib import Path

import torch
import yaml

from sld.audit.jacobian import (
    augmented_jacobian_excluding_identity,
    augmented_sigma_max_normalized,
    total_jacobian,
)
from sld.audit.spectral import operator_norm
from sld.experiments.factorial import _build_dataset_for
from sld.experiments.loading import load_run
from sld.objectives.jac_reg import Gauge, MetricIndeterminate, gauge_digest
from sld.predictors.base import PredictorState
from sld.training.trainer import Trainer

LR_GRID: tuple[float, ...] = (1e-4, 3e-4, 1e-3)
MU_J_GRID: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0)
BAND: tuple[float, float] = (0.90, 1.00)
# PROTOCOL section 10.3, "Indeterminacy, not repair": a terminal verdict
# reported VERBATIM, exactly like section 5's "band not reached" and section
# 2's "selection not converged". It is the value of a median that has no
# defined normalization (a zero/non-finite reference variance, or a cache
# position beyond the frozen reference rollout), and it is never floored,
# extrapolated, or silently dropped from the grid.
METRIC_INDETERMINATE = "metric indeterminate"
# PROTOCOL section 10.2 (amendment 1, 2026-07-30) supersedes section 5's
# 10-epoch calibration budget and section 2's 50-epoch sweep budget with a
# single 200-OPTIMIZER-UPDATE budget, applied identically to the LR sweeps,
# the mu_j calibration and the retained/core achieved-margin runs -- the
# per-arm fairness law (REGLES section 4) requires one shared budget, and
# 10 updates were measured to move sigma_max by ~1% (the transient regime
# where mu_j is inert under AdamW). The Trainer takes exactly one optimizer
# step per epoch (trainer.py, "one sampled train batch per epoch"), so an
# epoch count IS an update count here; the names below are kept for
# continuity with the config key.
CORE_EPOCHS = 200
CALIBRATION_EPOCHS = 200
# PROTOCOL section 10.2: step_subsample 0.25 -> 1.0 (every teacher-forced
# step kept; the Bernoulli step-mask was the dominant variance floor).
# k_probes stays at the Trainer default of 2 -- measured CV over 256 draws
# at step_subsample=1.0 is 0.075% (mlp) / 0.139% (min_ssm), two orders of
# magnitude inside the 2.5% requirement, so raising it buys nothing.
STEP_SUBSAMPLE = 1.0
H_FR = 5
SELECTION_SEED = 0
FAMILIES: tuple[str, ...] = ("mlp", "transformer", "min_ssm", "koopman_ssm")
SELECTION_EXP_NAME = "pendulum_core_selection"
# only the transformer's KV cache grows across the rollout (PROTOCOL
# section 5 "Scope"); min_ssm's state is fixed-size, mlp/koopman_ssm never
# populate it at all.
GROWING_MEMORY_FAMILIES = frozenset({"transformer"})

_RUN_FIELDS = (
    "family",
    "phase",
    "constraint",
    "regime",
    "seed",
    "lr",
    "mu_j",
    "epochs",
    "run_dir",
    "fr_val_loss",
    "sigma_max_median",
    # PROTOCOL section 10.3: which frozen gauge this run was TRAINED under
    # (empty on OFF cells, where the penalty -- and hence the gauge -- is
    # inert). It is the same digest the run's own config.yaml carries under
    # ``gauge_digest``, so a reviewer holding only the report and the run
    # directories can check, from disk, that the gauge the calibration
    # measured with is the gauge the training optimized under.
    "gauge_digest",
)


# --------------------------------------------------------------------------
# config construction
# --------------------------------------------------------------------------


def _arm_overrides(calibration_config: dict, family: str, constraint: str) -> dict:
    """Family/constraint arm's static overrides from the YAML, minus ``lr``/
    ``mu_j`` (those are this module's job to fill, never copied from the
    placeholder null/zero the YAML carries)."""
    for arm in calibration_config["arms"]:
        if arm["family"] == family and arm["constraint"] == constraint:
            return {k: v for k, v in arm["overrides"].items() if k not in ("lr", "mu_j")}
    raise ValueError(f"no arm family={family!r} constraint={constraint!r} in calibration config")


def _build_config(
    calibration_config: dict,
    family: str,
    constraint: str,
    *,
    lr: float,
    mu_j: float,
    epochs: int,
    seed: int = SELECTION_SEED,
) -> dict:
    """One ``Trainer`` config for ``family``'s ``constraint`` cell, built
    from ``calibration_config``'s ``base`` + that arm's static overrides
    (parsed from the YAML, never hand-duplicated) plus this call's
    lr/mu_j/epochs. Enforces the ``pendulum_core_selection`` namespace
    (PROTOCOL section 2: "Run namespaces")."""
    if calibration_config["exp_name"] != SELECTION_EXP_NAME:
        raise ValueError(
            f"calibration_config['exp_name'] must be {SELECTION_EXP_NAME!r}, "
            f"got {calibration_config['exp_name']!r}"
        )
    base = calibration_config["base"]
    overrides = _arm_overrides(calibration_config, family, constraint)
    return {
        **base,
        **overrides,
        "exp_name": calibration_config["exp_name"],
        "seed": seed,
        "family": family,
        "lr": lr,
        "mu_j": mu_j,
        "epochs": epochs,
        # PROTOCOL section 10.2. Set explicitly rather than left to the
        # Trainer default so the amended constant is recorded in every run's
        # config.yaml and enters the run identity hash on ON cells; on OFF
        # cells (mu_j == 0) the Trainer canonicalizes it back to the default
        # before hashing, since the penalty branch is never evaluated there.
        "step_subsample": STEP_SUBSAMPLE,
    }


def _fit_and_load(
    config: dict, data: dict, results_root: Path, gauge: Gauge | None = None
) -> tuple[dict, Path]:
    """Train one candidate and reload it. ``gauge`` is the frozen phase-OFF
    reference scale (PROTOCOL section 10.3), handed to the ``Trainer``
    OUT-OF-BAND so the mu_j penalty OPTIMIZES the same metric-normalized
    object this module then AUDITS with
    ``measure_sigma_max_median_normalized``. Passing it here (rather than
    only at measurement time) is the whole point: a run trained on the raw
    quantity and calibrated on the normalized one would be a
    training/audit mismatch, which is what the amendment exists to close.
    ``None`` on phase-OFF cells, where ``mu_j == 0`` makes the penalty --
    and hence the gauge -- inert."""
    trainer = Trainer(dict(config, results_root=str(results_root)), gauge=gauge)
    result = trainer.fit(data)
    run_dir = Path(result["paths"]["ckpt"]).parent
    return load_run(run_dir), run_dir


# --------------------------------------------------------------------------
# post-hoc measurements
# --------------------------------------------------------------------------


def _measure_fr_val_loss(
    encoder: torch.nn.Module,
    predictor,
    lifted: bool,
    val_split: dict,
    h_fr: int = H_FR,
) -> float:
    """Free-running validation loss at ``h_fr`` (PROTOCOL section 2 point 4).

    Honesty declaration (see module docstring): the EMA target is not
    reloaded from ``ckpt.pt`` by ``load_run``, so this regresses against the
    ONLINE encoder's own embeddings of the true future, not an EMA target --
    a declared post-hoc simplification. Plain MSE in projection space
    (``JEPALoss``'s ``pred`` term, ``lambda_c=0``): FR only, never TF, never
    the training-time composite loss. Computed over the WHOLE val split.
    """
    lift_fn = predictor.lift if lifted else (lambda z: z)
    project_fn = predictor.project if lifted else (lambda h: h)
    x = torch.as_tensor(val_split["x"], dtype=torch.float32).transpose(0, 1).contiguous()
    u = torch.as_tensor(val_split["u"], dtype=torch.float32).transpose(0, 1).contiguous()
    T = u.shape[0]
    h_fr = min(h_fr, T)
    with torch.no_grad():
        z = encoder(x)
        h0 = lift_fn(z[0])
        h_hat_fr = predictor.rollout(h0, u[:h_fr])
        pred_z = project_fn(h_hat_fr)
        target_z = z[1 : h_fr + 1]
        loss = torch.nn.functional.mse_loss(pred_z, target_z)
    return float(loss.item())


def _sigma_max_at_step(
    predictor_d,
    family: str,
    h_t: torch.Tensor,
    m_t: torch.Tensor | None,
    u_t: torch.Tensor,
) -> float:
    """``sigma_max(J^q_t)`` at one captured TF step, dispatched by what the
    step actually carries (mirrors ``Trainer._jac_state_penalty``'s
    dispatch doctrine, applied per-step instead of per-thread):

    - ``m_t is None`` (every family's first step, and every step of
      ``mlp``/``koopman_ssm``, which never populate ``state.m``):
      ``augmented_jacobian_excluding_identity(step_fn, h_t, None, u_t)``.
      This is the AUDITED object at a memory-creating step (``min_ssm``'s
      and ``transformer``'s first step): with no old ``m`` to differentiate
      against, ``d(h_next, m_next)/d(h_t)`` is a genuine RECTANGULAR
      Jacobian, and it is exactly what training penalizes there too --
      ``Trainer._jac_state_penalty`` dispatches the WHOLE TF thread to
      ``on_state_steps`` as soon as ANY step in it carries a tensor ``m``
      (true for ``min_ssm``/``transformer`` past their first step), so the
      first, ``m=None`` step of that same thread is included in
      ``on_state_steps``, not split off into the tensor-only path -- its
      own ``m_t is None`` branch differentiates ``(h_next, m_next)`` w.r.t.
      ``h_t`` alone (``jac_reg.TotalJacobianPenalty.on_state_steps``, the
      ``leaves = (h_t,) if m_t is None else (h_t, m_t)`` branch). Bug fixed
      here (Wave 5 round 7 review): this used to call ``total_jacobian`` on
      ``h_next`` ALONE, silently dropping the ``m_next`` rows, so it
      audited ``sigma_max(d h_next / d h)`` at exactly the step where
      training penalized ``||d(h_next, m_next)/d h||^2`` -- a different
      object.
      ``augmented_jacobian_excluding_identity`` is the right tool for this,
      not a repurposing of it: its own exclusion gate only fires when
      ``n_m > 0 and n_m_next > n_m`` (see its docstring's Returns section);
      with ``m=None`` here, ``n_m == 0``, so the gate never fires and it
      returns the FULL, un-excluded ``(h_next, m_next)`` Jacobian -- which
      is exactly the rectangular object above, and which collapses to the
      ordinary square ``d(h_next)/d(h)`` for ``mlp``/``koopman_ssm``, whose
      ``step_fn`` leaves ``m_next`` ``None`` too.
    - ``m_t`` a tensor, family not in ``GROWING_MEMORY_FAMILIES``
      (``min_ssm``, whose state is fixed-size): dense ``total_jacobian`` on
      the augmented ``q = concat(h_t, m_t)`` -- safe because ``m_t``'s size
      never changes for this family.
    - ``m_t`` a tensor, family in ``GROWING_MEMORY_FAMILIES``
      (``transformer``): ``augmented_jacobian_excluding_identity``, the
      identity-block-excluded object PROTOCOL section 5 declares for this
      family alone -- here the gate DOES fire (``m_next`` strictly grows),
      so the old-``m`` identity rows are found and removed.
    """

    def step_fn(h: torch.Tensor, m: torch.Tensor | None, u: torch.Tensor):
        h_next, state = predictor_d.step(h, u, PredictorState(m=m))
        return h_next, state.m

    if m_t is None:
        return augmented_jacobian_excluding_identity(step_fn, h_t, None, u_t).sigma_max

    if family in GROWING_MEMORY_FAMILIES:
        return augmented_jacobian_excluding_identity(step_fn, h_t, m_t, u_t).sigma_max

    h_shape, m_shape, u_shape = h_t.shape, m_t.shape, u_t.shape
    n_h = h_t.numel()

    def step_fn_flat(q: torch.Tensor, uu: torch.Tensor) -> torch.Tensor:
        h_part = q[:n_h].reshape(h_shape)
        m_part = q[n_h:].reshape(m_shape)
        h_next, state = predictor_d.step(h_part, uu.reshape(u_shape), PredictorState(m=m_part))
        return torch.cat([h_next.reshape(-1), state.m.reshape(-1)])

    q = torch.cat([h_t.reshape(-1), m_t.reshape(-1)])
    J = total_jacobian(step_fn_flat, q, u_t.reshape(-1))
    return operator_norm(J)


def _measure_sigma_max_median(
    encoder: torch.nn.Module,
    predictor,
    lifted: bool,
    family: str,
    val_split: dict,
) -> float:
    """RAW (un-normalized) median ``sigma_max(J^q_t)`` over states visited in
    teacher-forcing on the val split.

    NOT the decisional quantity of the protocol. PROTOCOL section 10.3
    supersedes section 5 step 3's raw margin with the metric-normalized
    ``sigma_max(S_out^-1 . J^{q,excl}_t . S_in)``, computed by
    ``measure_sigma_max_median_normalized``; that is the ONLY quantity the
    calibration path compares to ``BAND``. This function is retained purely
    as the un-normalized comparison reading -- it is what the pre-amendment
    campaign measured, it is what ``_private/research/mu-j-calibration/
    run_criterion3.py`` reports, and keeping it makes "raw vs declared"
    auditable side by side instead of unmeasurable. Do not reintroduce it
    into a selection decision.

    Threads ``predictor.step``
    over the TRUE prefix, one val episode at a time, capturing
    ``(h_t, m_t, u_t)`` BEFORE every step -- exactly
    ``Trainer._composite_loss``'s TF branch, just unbatched and in CPU
    float64 (audit doctrine, ``sld.audit.jacobian``). ``encoder``/
    ``predictor`` are deep-copied and cast to double here; the caller's
    (float32) objects are never mutated. States are detached across steps
    (state-advance runs under ``torch.no_grad()``), matching the "detached"
    honesty declaration in PROTOCOL section 5 step 3.
    """
    encoder_d = copy.deepcopy(encoder).double().eval()
    predictor_d = copy.deepcopy(predictor).double().eval()
    lift_fn_d = predictor_d.lift if lifted else (lambda z: z)

    x = torch.as_tensor(val_split["x"], dtype=torch.float64)  # (n_ep, T+1, d_x)
    u = torch.as_tensor(val_split["u"], dtype=torch.float64)  # (n_ep, T, d_u)
    n_ep, T = u.shape[0], u.shape[1]

    with torch.no_grad():
        z = encoder_d(x)  # (n_ep, T+1, d_z)
        h_true = lift_fn_d(z[:, :T])  # (n_ep, T, d_h)

    sigmas: list[float] = []
    for ep in range(n_ep):
        state = PredictorState()
        for t in range(T):
            h_t = h_true[ep, t].unsqueeze(0)
            u_t = u[ep, t].unsqueeze(0)
            m_t = state.m
            sigmas.append(_sigma_max_at_step(predictor_d, family, h_t, m_t, u_t))
            with torch.no_grad():
                _, state = predictor_d.step(h_t, u_t, state)

    # Conventional median (stdlib, not a hand-rolled order statistic): for an
    # EVEN count, the mean of the two central values, not the upper of the
    # two. ``sigmas`` is a plain Python float list -- no float64/dtype
    # concern that would push this back to a numpy/torch reimplementation.
    return statistics.median(sigmas)


# --------------------------------------------------------------------------
# PROTOCOL.md section 10.3 amendment: the declared, gauge-normalized metric
# --------------------------------------------------------------------------


def _centered_std(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """PROTOCOL.md section 10.3's "centered per-coordinate standard
    deviations": Bessel-corrected (``unbiased=True``) sample std along
    ``dim``, mean-subtracted. With too few samples along ``dim`` (e.g. a
    single reference episode), this returns ``NaN`` by construction
    (``n - 1 == 0``) -- never floored or special-cased here: it is stored
    as-is in the ``Gauge`` and left to trigger ``MetricIndeterminate`` at
    the point of use, per the "indeterminacy, not repair" discipline."""
    return torch.std(x, dim=dim, unbiased=True)


def estimate_gauge(
    encoder: torch.nn.Module,
    predictor,
    lifted: bool,
    family: str,
    val_split: dict,
) -> Gauge:
    """PROTOCOL.md section 10.3's "Reference scales": ``S`` estimated on
    the retained phase-OFF, seed-0, OU-regime TF validation trajectories,
    BEFORE the mu_j sweep, frozen for the whole sweep -- the caller must
    pass the OFF-cell checkpoint's own ``encoder``/``predictor`` (never an
    ON/penalized run's).

    Threads ``predictor.step`` over the TRUE prefix exactly like
    ``_measure_sigma_max_median`` (the SAME TF distribution the penalty
    targets), in CPU float64, on deep copies -- the caller's live objects
    are never mutated.

    ``h``'s scale is one frozen per-coordinate std, pooled over every
    ``(episode, timestep)`` visited -- ``h`` never grows. A FIXED-SIZE
    ``m`` (``min_ssm``'s recurrent state) is estimated the same way,
    pooled over ``(episode, timestep)``. A GROWING ``m``
    (``transformer``'s KV cache, the only family in
    ``GROWING_MEMORY_FAMILIES``) needs one frozen per-position block
    instead (PROTOCOL section 10.3's "Growing memory"): each position's
    scale is estimated ACROSS EPISODES ONLY, at the exact step that WRITES
    it (one creation event per position per episode), using the
    transformer's own declared cache layout ``(n_layers, 2, B, n_heads, t,
    d_head)`` (``predictors.transformer`` module docstring) -- knowledge
    that legitimately lives HERE (family-aware orchestration,
    ``GROWING_MEMORY_FAMILIES`` dispatch already exists in this module),
    never in ``audit/`` or ``objectives/`` (which stay layout-agnostic).

    Returns:
        A frozen ``Gauge``. Any zero-variance or non-finite coordinate is
        NOT floored here -- stored as-is, so a later ``Gauge.h_scale()``/
        ``Gauge.m_scale()`` call raises ``MetricIndeterminate`` at the
        point of use.
    """
    encoder_d = copy.deepcopy(encoder).double().eval()
    predictor_d = copy.deepcopy(predictor).double().eval()
    lift_fn_d = predictor_d.lift if lifted else (lambda z: z)

    x = torch.as_tensor(val_split["x"], dtype=torch.float64)
    u = torch.as_tensor(val_split["u"], dtype=torch.float64)
    n_ep, T = u.shape[0], u.shape[1]

    with torch.no_grad():
        z = encoder_d(x)
        h_true = lift_fn_d(z[:, :T])  # (n_ep, T, d_h)

    s_h = _centered_std(h_true.reshape(-1, h_true.shape[-1]), dim=0).reshape(1, -1)

    if family not in GROWING_MEMORY_FAMILIES:
        m_samples: list[torch.Tensor] = []
        for ep in range(n_ep):
            state = PredictorState()
            for t in range(T):
                h_t = h_true[ep, t].unsqueeze(0)
                u_t = u[ep, t].unsqueeze(0)
                with torch.no_grad():
                    _, state = predictor_d.step(h_t, u_t, state)
                if state.m is not None:
                    m_samples.append(state.m)
        if not m_samples:
            return Gauge(s_h=s_h)  # mlp / koopman_ssm: never populate m
        m_stack = torch.cat(m_samples, dim=0)  # (n_ep*T, d_state)
        s_m = _centered_std(m_stack, dim=0).reshape(1, -1)
        return Gauge(s_h=s_h, s_m=s_m)

    # Growing memory (transformer): one frozen per-position block, labeled
    # structurally (layer, K/V, head, feature), each estimated at the step
    # that CREATES it, pooled over episodes only.
    position_slices: list[list[torch.Tensor]] = [[] for _ in range(T)]
    for ep in range(n_ep):
        state = PredictorState()
        for t in range(T):
            h_t = h_true[ep, t].unsqueeze(0)
            u_t = u[ep, t].unsqueeze(0)
            with torch.no_grad():
                _, state = predictor_d.step(h_t, u_t, state)
            # (n_layers, 2, B, n_heads, t_len, d_head); the step just
            # written is always the LAST slot along the position axis (4).
            new_slice = state.m[:, :, :, :, -1, :]  # (n_layers, 2, B, n_heads, d_head)
            position_slices[t].append(new_slice)

    per_position_scale = []
    for t in range(T):
        stacked = torch.cat(position_slices[t], dim=2)  # (n_layers, 2, n_ep, n_heads, d_head)
        scale_t = _centered_std(stacked, dim=2)  # std ACROSS EPISODES -- (n_layers, 2, n_heads, d_head)
        per_position_scale.append(scale_t.unsqueeze(2).unsqueeze(4))  # -> (n_layers, 2, 1, n_heads, 1, d_head)
    s_m_positions = torch.cat(per_position_scale, dim=4)  # (n_layers, 2, 1, n_heads, T, d_head)
    return Gauge(s_h=s_h, s_m_positions=s_m_positions, position_axis=4)


def measure_sigma_max_median_normalized(
    encoder: torch.nn.Module,
    predictor,
    lifted: bool,
    family: str,
    val_split: dict,
    gauge: Gauge,
) -> float | str:
    """PROTOCOL.md section 10.3's declared metric: the median, over states
    visited in teacher-forcing on the val split, of
    ``sigma_max(S_out^-1 . J_t^{q,excl} . S_in)`` -- the SAME aggregation
    as ``_measure_sigma_max_median`` (same TF distribution, same per-step
    dispatch to ``audit.jacobian``), but the DECLARED, gauge-normalized
    quantity instead of the raw one, via
    ``audit.jacobian.augmented_sigma_max_normalized`` (which already
    handles all four families uniformly -- fixed-size, growing, or no
    ``m`` at all -- so this needs no per-family branch, unlike
    ``_sigma_max_at_step``'s raw dispatch).

    A ``MetricIndeterminate`` at ANY visited step makes the WHOLE median
    indeterminate: the median summarizes one margin, and one undefined
    reference-scale coordinate leaves that step's normalized Jacobian
    undefined, with no principled way to drop just that step (PROTOCOL
    section 10.3: "Indeterminacy, not repair"). Returns the literal string
    ``"metric indeterminate"`` in that case, never a float -- the same
    verbatim-verdict convention this module already uses for "band not
    reached"/"selection not converged".

    Args:
        encoder, predictor, lifted, family, val_split: as
            ``_measure_sigma_max_median``.
        gauge: the frozen reference scale, from ``estimate_gauge`` on the
            retained phase-OFF checkpoint -- never re-estimated here.
    """
    encoder_d = copy.deepcopy(encoder).double().eval()
    predictor_d = copy.deepcopy(predictor).double().eval()
    lift_fn_d = predictor_d.lift if lifted else (lambda z: z)

    x = torch.as_tensor(val_split["x"], dtype=torch.float64)
    u = torch.as_tensor(val_split["u"], dtype=torch.float64)
    n_ep, T = u.shape[0], u.shape[1]

    with torch.no_grad():
        z = encoder_d(x)
        h_true = lift_fn_d(z[:, :T])

    def step_fn(h: torch.Tensor, m: torch.Tensor | None, u: torch.Tensor):
        h_next, state = predictor_d.step(h, u, PredictorState(m=m))
        return h_next, state.m

    sigmas: list[float] = []
    for ep in range(n_ep):
        state = PredictorState()
        for t in range(T):
            h_t = h_true[ep, t].unsqueeze(0)
            u_t = u[ep, t].unsqueeze(0)
            m_t = state.m
            try:
                sigmas.append(augmented_sigma_max_normalized(step_fn, h_t, m_t, u_t, gauge))
            except MetricIndeterminate:
                return METRIC_INDETERMINATE
            with torch.no_grad():
                _, state = predictor_d.step(h_t, u_t, state)

    return statistics.median(sigmas)


# --------------------------------------------------------------------------
# decision logic (pure -- no training here, unit-testable in isolation)
# --------------------------------------------------------------------------


def _select_lr(scores: dict[float, float]) -> float:
    """PROTOCOL section 2 point 5: smallest LR among exact ties at the
    minimum averaged score."""
    min_score = min(scores.values())
    tied = sorted(lr for lr, s in scores.items() if s == min_score)
    return tied[0]


def _in_band(median: float) -> bool:
    """``median`` inside the calibration band. Refuses a non-float outright:
    ``METRIC_INDETERMINATE`` is a TERMINAL verdict, not a value that may be
    compared, floored or ordered, so any caller that reaches a band test
    holding one has failed to terminate and must fail loudly here rather
    than produce a silent verdict (in Python 3 ``0.9 <= "metric
    indeterminate"`` raises, but ``sorted``/``all`` over mixed values can
    fail far more quietly)."""
    _require_determinate(median)
    return BAND[0] <= median <= BAND[1]


def _require_determinate(*medians: float) -> None:
    for median in medians:
        if not isinstance(median, float):
            raise ValueError(
                f"{median!r} is not a determinate median: an indeterminate "
                "metric is a terminal verdict reported verbatim (PROTOCOL "
                "section 10.3), never an input to a band test, an ordering, "
                "a monotonicity check or a geometric mean"
            )


def _select_mu_j(medians: dict[float, float]) -> tuple[float | None, str]:
    """PROTOCOL section 5 step 5 (retention only, no refinement): smallest
    grid value whose median falls in ``BAND``. Returns
    ``(mu_j, "selected")`` or ``(None, "none")``."""
    _require_determinate(*medians.values())
    in_band = sorted(mu for mu, med in medians.items() if BAND[0] <= med <= BAND[1])
    if in_band:
        return in_band[0], "selected"
    return None, "none"


def _refinement_candidate(medians: dict[float, float]) -> float | None:
    """PROTOCOL section 5 step 6's four-branch refinement rule, given the
    ORIGINAL (un-refined) grid's medians. Returns the one point to
    re-measure, or ``None`` for the non-monotone abstention branch.

    "Monotone" is read verbatim from the protocol text ("the medians
    decrease monotonically as mu_j increases"): NON-INCREASING, equality
    admitted -- a plateau is monotone, not a violation. A strict ``>``
    reading (fixed here, Wave 5 round 7 review) misclassified an exact
    plateau (e.g. two adjacent grid points with the identical measured
    median, which real 10-epoch proxy runs do produce) as "non-monotone"
    and abstained from the one refinement point the protocol's own
    monotone-straddle branch would have taken. The four branches stay
    exhaustive and mutually exclusive exactly as before: every set of
    medians is all-above, all-below, or (not all-above and not all-below)
    -- and within that last set, either non-increasing (monotone) or not.
    """
    _require_determinate(*medians.values())
    grid = sorted(medians)
    values = [medians[g] for g in grid]
    if all(v > BAND[1] for v in values):
        return grid[-1] * 10
    if all(v < BAND[0] for v in values):
        return grid[0] / 10
    monotone_non_increasing = all(values[i] >= values[i + 1] for i in range(len(values) - 1))
    if not monotone_non_increasing:
        return None  # non-monotone -> deterministic abstention
    for i in range(len(grid) - 1):
        if values[i] > BAND[1] and values[i + 1] < BAND[0]:
            return (grid[i] * grid[i + 1]) ** 0.5
    return None  # defensive: monotone but no bracketing pair (shouldn't occur)


# --------------------------------------------------------------------------
# training-driving phases
# --------------------------------------------------------------------------


def _sweep_lr(
    family: str,
    constraint: str,
    mu_j: float,
    epochs: int,
    calibration_config: dict,
    datasets: dict[str, dict],
    results_root: Path,
    gauge: Gauge | None = None,
    seed: int = SELECTION_SEED,
) -> dict:
    """PROTOCOL section 2's sweep procedure, one phase, one family: trains
    ``LR_GRID x {"ou", "action_conditioned"}`` (6 ``Trainer.fit`` runs) and
    retains the LR minimizing the unweighted mean of both regimes' post-hoc
    FR validation loss, tie-broken to the smallest LR.

    Also returns ``retained_ou_run`` -- the ``load_run`` dict of the RETAINED
    LR's OU-regime, seed-0 checkpoint. That is exactly the object PROTOCOL
    section 10.3 requires the frozen gauge to be estimated on when this sweep
    is the phase-OFF one, and it is already in memory here: reloading it from
    ``run_dir`` afterwards would re-derive what this loop just built.

    ``gauge`` is ``None`` for the phase-OFF sweep (it does not exist yet --
    this sweep is what produces the checkpoint it is estimated from) and the
    caller's FROZEN gauge for every phase-ON sweep, where ``mu_j > 0`` makes
    the penalty live: those ON runs must be trained under the same declared
    metric their margin is later measured in."""
    runs = []
    scores: dict[float, float] = {}
    ou_runs: dict[float, dict] = {}
    for lr in LR_GRID:
        per_regime = {}
        for regime_name in ("ou", "action_conditioned"):
            config = _build_config(
                calibration_config, family, constraint, lr=lr, mu_j=mu_j, epochs=epochs, seed=seed
            )
            run, run_dir = _fit_and_load(config, datasets[regime_name], results_root, gauge)
            fr_loss = _measure_fr_val_loss(
                run["encoder"], run["predictor"], hasattr(run["predictor"], "lift"),
                datasets[regime_name]["val"],
            )
            per_regime[regime_name] = fr_loss
            if regime_name == "ou":
                ou_runs[lr] = run
            runs.append(
                {
                    "family": family,
                    "phase": "lr_sweep",
                    "constraint": constraint,
                    "regime": regime_name,
                    "seed": seed,
                    "lr": lr,
                    "mu_j": mu_j,
                    "epochs": epochs,
                    "run_dir": str(run_dir),
                    "fr_val_loss": fr_loss,
                    "sigma_max_median": None,
                    "gauge_digest": None if gauge is None else gauge_digest(gauge),
                }
            )
        scores[lr] = 0.5 * (per_regime["ou"] + per_regime["action_conditioned"])
    retained_lr = _select_lr(scores)
    return {
        "retained_lr": retained_lr,
        "scores": scores,
        "runs": runs,
        "retained_ou_run": ou_runs[retained_lr],
    }


def _measure_mu_j_candidate(
    family: str,
    lr: float,
    mu_j_value: float,
    epochs: int,
    calibration_config: dict,
    datasets: dict[str, dict],
    results_root: Path,
    gauge: Gauge,
    seed: int = SELECTION_SEED,
) -> tuple[float | str, dict]:
    """Train the ON cell at ``(lr, mu_j_value)`` for ``epochs`` on the OU
    regime only, and return its post-hoc DECLARED margin plus a trace row.

    The margin is ``measure_sigma_max_median_normalized`` under ``gauge``
    (PROTOCOL section 10.3), NOT the raw ``_measure_sigma_max_median``.
    ``gauge`` is the caller's frozen phase-OFF gauge, passed in explicitly
    and never estimated here: estimating it from this penalized run is
    exactly the moving-gauge failure the amendment forbids.

    The SAME object is handed to the ``Trainer`` that produces the
    checkpoint, so the candidate is TRAINED under the metric it is then
    MEASURED in. Training on the raw quantity and calibrating on the
    normalized one would optimize one object and grade another.

    Returns ``(median, row)`` where ``median`` is a float or the terminal
    string ``METRIC_INDETERMINATE`` -- the caller must dispatch on it before
    any band test."""
    config = _build_config(
        calibration_config, family, "on", lr=lr, mu_j=mu_j_value, epochs=epochs, seed=seed
    )
    run, run_dir = _fit_and_load(config, datasets["ou"], results_root, gauge)
    median = measure_sigma_max_median_normalized(
        run["encoder"], run["predictor"], hasattr(run["predictor"], "lift"),
        family, datasets["ou"]["val"], gauge,
    )
    row = {
        "family": family,
        "phase": "mu_j_calibration",
        "constraint": "on",
        "regime": "ou",
        "seed": seed,
        "lr": lr,
        "mu_j": mu_j_value,
        "epochs": epochs,
        "run_dir": str(run_dir),
        "fr_val_loss": None,
        "sigma_max_median": median,
        "gauge_digest": gauge_digest(gauge),
    }
    return median, row


def _calibrate_mu_j(
    family: str,
    lr: float,
    calibration_config: dict,
    datasets: dict[str, dict],
    results_root: Path,
    gauge: Gauge,
    epochs: int = CALIBRATION_EPOCHS,
    seed: int = SELECTION_SEED,
) -> dict:
    """PROTOCOL section 5 steps 1-6 in full: mu_j sweep, retention rule, and
    at most one deterministic refinement point -- every candidate measured
    under the caller's SINGLE frozen ``gauge`` (section 10.3), including the
    refinement point.

    Indeterminacy is terminal for the whole calibration, not for one grid
    point: an indeterminate median leaves the grid's median vector
    incomplete, and every remaining decision (the band test, the all-above/
    all-below branches, the monotonicity check, the bracketing geometric
    mean) is defined over the COMPLETE vector. Dropping the point would be
    an undeclared candidate removal and flooring it is forbidden, so the
    verdict becomes ``METRIC_INDETERMINATE``, reported verbatim, with the
    trace rows measured so far kept."""
    runs = []
    medians: dict[float, float] = {}
    for mu in MU_J_GRID:
        median, row = _measure_mu_j_candidate(
            family, lr, mu, epochs, calibration_config, datasets, results_root, gauge, seed
        )
        medians[mu] = median
        runs.append(row)
        if median == METRIC_INDETERMINATE:
            return {
                "retained_mu_j": None, "verdict": METRIC_INDETERMINATE,
                "medians": medians, "runs": runs,
            }

    retained_mu_j, verdict = _select_mu_j(medians)
    if verdict == "none":
        candidate = _refinement_candidate(medians)
        if candidate is None:
            retained_mu_j, verdict = None, "band not reached"
        else:
            refined_median, row = _measure_mu_j_candidate(
                family, lr, candidate, epochs, calibration_config, datasets, results_root, gauge, seed
            )
            row["phase"] = "mu_j_calibration_refinement"
            runs.append(row)
            medians[candidate] = refined_median
            if refined_median == METRIC_INDETERMINATE:
                retained_mu_j, verdict = None, METRIC_INDETERMINATE
            elif _in_band(refined_median):
                retained_mu_j, verdict = candidate, "selected"
            else:
                retained_mu_j, verdict = None, "band not reached"

    return {"retained_mu_j": retained_mu_j, "verdict": verdict, "medians": medians, "runs": runs}


def _measure_proxy_at(
    family: str,
    lr: float,
    mu_j: float,
    calibration_config: dict,
    datasets: dict[str, dict],
    results_root: Path,
    gauge: Gauge,
    seed: int = SELECTION_SEED,
    epochs: int = CALIBRATION_EPOCHS,
) -> tuple[float | str, dict]:
    """PROTOCOL section 2 phase 4's conditional re-measurement: the same
    OU-only proxy at the amended 200-update calibration budget (section
    10.2), at a fixed ``(lr, mu_j)`` pair (no fresh mu_j sweep), under the
    SAME frozen phase-OFF ``gauge`` as the sweep that produced ``mu_j`` --
    re-estimating it here would move the metric between the sweep and its
    own consistency check."""
    median, row = _measure_mu_j_candidate(
        family, lr, mu_j, epochs, calibration_config, datasets, results_root, gauge, seed
    )
    row["phase"] = "phase4_remeasure"
    return median, row


def run_family_selection(
    family: str,
    calibration_config: dict,
    results_root: Path,
    core_epochs: int = CORE_EPOCHS,
    calibration_epochs: int = CALIBRATION_EPOCHS,
) -> dict:
    """Orchestrates PROTOCOL sections 2 and 5's four ordered phases for one
    family. Returns ``{"family", "lr_off", "lr_on", "mu_j", "verdict",
    "runs"}`` where ``verdict`` is one of ``"selected"``, ``"band not
    reached"``, ``"selection not converged"``, ``METRIC_INDETERMINATE``
    (PROTOCOL's own terminal states, reported verbatim) and ``"runs"`` is
    the full per-run trace (every ``Trainer.fit`` this call performed).

    The frozen gauge (PROTOCOL section 10.3) is estimated HERE, exactly
    once per family, on phase OFF's retained OU/seed-0 checkpoint, after
    the OFF sweep and BEFORE the mu_j sweep -- then passed down, unchanged,
    to every measurement below it: the initial calibration, its refinement
    point, the phase-4 re-measurement, and both bounded-iteration cycles'
    recalibrations and re-measurements. Nothing under this function ever
    estimates a gauge of its own.

    That same object is also handed to every TRAINING run below it that has
    a live penalty -- the phase-ON LR sweeps and every mu_j candidate -- so
    the constraint those runs optimize and the margin they are graded on are
    one quantity, not two. It reaches ``Trainer`` out-of-band and enters
    each ON run's identity as ``gauge_digest`` (see ``trainer.py``); the
    phase-OFF sweep above runs at ``mu_j = 0``, where the gauge is inert and
    would not exist yet in any case.

    ``core_epochs``/``calibration_epochs`` both default to the amended
    200-update budget (PROTOCOL section 10.2 supersedes the original 50/10)
    but are overridable for tests -- the sweep/calibration/refinement/
    bounded-iteration LOGIC is identical at any budget, only the training
    cost changes.
    """
    results_root = Path(results_root)
    datasets = {name: _build_dataset_for(ds_cfg) for name, ds_cfg in calibration_config["datasets"].items()}
    runs: list[dict] = []

    # Phase OFF
    off_sweep = _sweep_lr(family, "off", 0.0, core_epochs, calibration_config, datasets, results_root)
    runs.extend(off_sweep["runs"])
    lr_off = off_sweep["retained_lr"]

    # The frozen gauge: PROTOCOL section 10.3's "Reference scales" -- ONE
    # estimation per family, on the RETAINED phase-OFF, OU-regime, seed-0
    # checkpoint, BEFORE the mu_j sweep, never re-estimated afterwards.
    off_run = off_sweep["retained_ou_run"]
    gauge = estimate_gauge(
        off_run["encoder"], off_run["predictor"], hasattr(off_run["predictor"], "lift"),
        family, datasets["ou"]["val"],
    )

    # Initial calibration (section 2 phase 2 == section 5 steps 1-6)
    calib = _calibrate_mu_j(
        family, lr_off, calibration_config, datasets, results_root, gauge, calibration_epochs
    )
    runs.extend(calib["runs"])
    if calib["verdict"] != "selected":
        # verbatim propagation: "band not reached" OR METRIC_INDETERMINATE,
        # never collapsed into one another
        return {
            "family": family, "lr_off": lr_off, "lr_on": None, "mu_j": None,
            "verdict": calib["verdict"], "runs": runs,
        }
    mu_j = calib["retained_mu_j"]

    # Phase ON (initial sweep, mu_j fixed)
    on_sweep = _sweep_lr(
        family, "on", mu_j, core_epochs, calibration_config, datasets, results_root, gauge
    )
    runs.extend(on_sweep["runs"])
    lr_on = on_sweep["retained_lr"]

    if lr_on == lr_off:
        return {
            "family": family, "lr_off": lr_off, "lr_on": lr_on, "mu_j": mu_j,
            "verdict": "selected", "runs": runs,
        }

    proxy, row = _measure_proxy_at(
        family, lr_on, mu_j, calibration_config, datasets, results_root, gauge,
        epochs=calibration_epochs,
    )
    runs.append(row)
    if proxy == METRIC_INDETERMINATE:
        return {
            "family": family, "lr_off": lr_off, "lr_on": lr_on, "mu_j": mu_j,
            "verdict": METRIC_INDETERMINATE, "runs": runs,
        }
    if _in_band(proxy):
        return {
            "family": family, "lr_off": lr_off, "lr_on": lr_on, "mu_j": mu_j,
            "verdict": "selected", "runs": runs,
        }

    lr_fixed = lr_on
    for _cycle in (1, 2):
        recalib = _calibrate_mu_j(
            family, lr_fixed, calibration_config, datasets, results_root, gauge, calibration_epochs
        )
        runs.extend(recalib["runs"])
        if recalib["verdict"] != "selected":
            return {
                "family": family, "lr_off": lr_off, "lr_on": lr_fixed, "mu_j": None,
                "verdict": recalib["verdict"], "runs": runs,
            }
        mu_j = recalib["retained_mu_j"]

        re_sweep = _sweep_lr(
            family, "on", mu_j, core_epochs, calibration_config, datasets, results_root, gauge
        )
        runs.extend(re_sweep["runs"])
        lr_new = re_sweep["retained_lr"]

        if lr_new == lr_fixed:
            return {
                "family": family, "lr_off": lr_off, "lr_on": lr_new, "mu_j": mu_j,
                "verdict": "selected", "runs": runs,
            }

        proxy, row = _measure_proxy_at(
            family, lr_new, mu_j, calibration_config, datasets, results_root, gauge,
            epochs=calibration_epochs,
        )
        runs.append(row)
        if proxy == METRIC_INDETERMINATE:
            return {
                "family": family, "lr_off": lr_off, "lr_on": lr_new, "mu_j": mu_j,
                "verdict": METRIC_INDETERMINATE, "runs": runs,
            }
        if _in_band(proxy):
            return {
                "family": family, "lr_off": lr_off, "lr_on": lr_new, "mu_j": mu_j,
                "verdict": "selected", "runs": runs,
            }
        lr_fixed = lr_new

    return {
        "family": family, "lr_off": lr_off, "lr_on": lr_fixed, "mu_j": mu_j,
        "verdict": "selection not converged", "runs": runs,
    }


def run_all_families(
    calibration_config_path: str | Path,
    results_root: str | Path,
    families: tuple[str, ...] = FAMILIES,
    core_epochs: int = CORE_EPOCHS,
    calibration_epochs: int = CALIBRATION_EPOCHS,
) -> dict:
    """Loops ``run_family_selection`` over ``families`` and writes the full
    report (``selection_report.json``/``.csv``) under
    ``<results_root>/pendulum_core_selection/`` -- every run performed,
    never summarized away (task point 9a)."""
    calibration_config_path = Path(calibration_config_path)
    with open(calibration_config_path) as f:
        calibration_config = yaml.safe_load(f)
    results_root = Path(results_root)

    family_results = {}
    all_runs: list[dict] = []
    for family in families:
        res = run_family_selection(family, calibration_config, results_root, core_epochs, calibration_epochs)
        family_results[family] = {k: v for k, v in res.items() if k != "runs"}
        all_runs.extend(res["runs"])

    report_dir = results_root / SELECTION_EXP_NAME
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "selection_report.json").write_text(
        json.dumps({"summary": family_results, "runs": all_runs}, indent=2, sort_keys=True)
    )
    _write_report_csv(report_dir / "selection_report.csv", all_runs)
    return {"summary": family_results, "runs": all_runs}


def _write_report_csv(path: Path, runs: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(_RUN_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in runs:
            writer.writerow({k: row.get(k, "") for k in _RUN_FIELDS})


# --------------------------------------------------------------------------
# pendulum_core.yaml patcher -- NOT auto-invoked, manual review gate
# --------------------------------------------------------------------------


def _patch_arms(doc: dict, summary: dict) -> None:
    for arm in doc["arms"]:
        res = summary.get(arm["family"])
        if res is None:
            continue
        if arm["constraint"] == "off":
            arm["overrides"]["lr"] = res["lr_off"]
        elif arm["constraint"] == "on":
            # A terminal verdict (anything but "selected" -- "band not
            # reached", "selection not converged", METRIC_INDETERMINATE)
            # never carries a usable lr_on/mu_j: run_family_selection
            # returns None for both whenever it returns a non-"selected"
            # verdict. Writing them anyway would silently promote a failed
            # calibration into a runnable core cell (PROTOCOL.md forbids
            # undeclared nearest-value fallback). Reset both to null and
            # refuse BEFORE the caller ever opens the file for writing.
            if res["verdict"] != "selected":
                arm["overrides"]["lr"] = None
                arm["overrides"]["mu_j"] = None
                raise ValueError(
                    f"family {arm['family']!r} ON arm: selection verdict is "
                    f"{res['verdict']!r}, not 'selected' -- refusing to patch "
                    "lr/mu_j from a terminal/failed calibration result"
                )
            arm["overrides"]["lr"] = res["lr_on"]
            arm["overrides"]["mu_j"] = res["mu_j"]


def apply_selection_to_pendulum_core(summary: dict, pendulum_core_yaml_path: str | Path) -> None:
    """Patch ``pendulum_core.yaml``'s ``lr``/``mu_j`` nulls from a finished
    selection ``summary`` (``run_all_families(...)["summary"]``).

    NOT called at import time or by ``run_all_families``: this is a
    separate, deliberate step the supervisor takes only after manually
    reviewing ``selection_report.{json,csv}``. Uses ``ruamel.yaml``
    (round-trip, comment-preserving) if installed; otherwise falls back to
    ``pyyaml``, which DISCARDS every comment in the file -- printed as a
    loud warning, per this task's fallback instruction, since ``ruamel.yaml``
    is not currently a project dependency.
    """
    path = Path(pendulum_core_yaml_path)
    try:
        from ruamel.yaml import YAML

        yaml_rt = YAML()
        yaml_rt.preserve_quotes = True
        with open(path) as f:
            doc = yaml_rt.load(f)
        _patch_arms(doc, summary)
        with open(path, "w") as f:
            yaml_rt.dump(doc, f)
    except ImportError:
        print(
            f"WARNING: ruamel.yaml not installed -- falling back to pyyaml, "
            f"which discards ALL comments in {path}. Install ruamel.yaml "
            "(`uv add ruamel.yaml`) to preserve them before patching a file "
            "meant to stay human-readable."
        )
        with open(path) as f:
            doc = yaml.safe_load(f)
        _patch_arms(doc, summary)
        with open(path, "w") as f:
            yaml.safe_dump(doc, f, sort_keys=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="src/sld/experiments/configs/calibration.yaml",
        help="calibration.yaml path (exp_name must be pendulum_core_selection)",
    )
    parser.add_argument("--results-root", default="results")
    parser.add_argument(
        "--apply", action="store_true",
        help="patch pendulum_core.yaml with the resolved lr/mu_j after the report is "
        "written -- review selection_report.{json,csv} manually first",
    )
    parser.add_argument(
        "--pendulum-core-yaml", default="src/sld/experiments/configs/pendulum_core.yaml"
    )
    args = parser.parse_args(argv)

    result = run_all_families(args.config, args.results_root)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    if args.apply:
        apply_selection_to_pendulum_core(result["summary"], args.pendulum_core_yaml)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
