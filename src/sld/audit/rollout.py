"""Rollout-level offline audit.

Everything that scores a free-running rollout against ground truth
(PROTOCOL.md sections 12.1-12.3) lives in this one file: ``vpt`` (section
12.2); ``fit_train_state_scale`` and ``normalized_state_error`` (section
12.1, the normalized state error ``e_h``); ``persistence_rollout`` and
``linear_extrapolation_rollout`` (section 12.3's two trivial baselines);
and ``evaluate_positive_control`` (section 12.3's gate, pure arithmetic on
three already-computed ``e_h`` curves -- no I/O, no artifact writing, that
is a later wave's job).

``vpt`` implements PROTOCOL.md section 12.2's contract, itself a direct
adoption -- not a house variant -- of Vlachas, Pathak, Hunt, Girvan, Ott &
Koumoutsakos (2020), "Backpropagation algorithms and Reservoir Computing in
Recurrent Neural Networks for the forecasting of complex spatiotemporal
dynamics", Neural Networks 126:191-217, arXiv:1910.05266: the valid
prediction time is the length of the longest prefix on which a per-step
error stays below a threshold,

    VPT_tau(e) = max { h in {0, 1, ..., H} : e_j < tau for all j <= h }

Consequences normative here, not implementation choices (PROTOCOL.md 12.2):

- Validity is STRICT ``<``; a step with ``e_h >= tau`` is invalid, so
  ``e = [0, .2, .4, .6]`` at ``tau = .4`` gives ``VPT = 2`` (``e_3 == tau``
  is NOT valid -- never 3).
- ``e_1 >= tau`` gives ``VPT = 0`` (zero valid steps), never 1 -- a "first
  crossing index" convention would report 1 here and silently mean
  something else (a step being valid, rather than zero being valid).
- If no ``e_j`` ever reaches ``tau`` within the observed horizon ``H``, the
  true crossing time was never observed: this is right-censoring, and
  fabricating a finite substitute (``H``, ``H+1``, ...) reports a
  measurement that was not taken (``H`` in particular is indistinguishable
  from "crossed exactly at H+1"). ``vpt`` instead returns ``value = inf``,
  ``status = "censored"``, and keeps ``horizon = H`` alongside so the
  caller knows how far the non-crossing was actually observed.
- This function performs NO normalization (the caller supplies an already
  per-coordinate-normalized ``e``, PROTOCOL.md section 12.1) and NO
  aggregation over episodes, seeds, or a Lyapunov time scale -- each of
  those is a separate, system- or study-specific decision (section 12.5),
  and folding any of them in here would make this "the VPT of one study"
  instead of the one reducer every study shares.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from torch import Tensor

from sld.audit._common import as_float64_array


class VPTResult(NamedTuple):
    """Output of ``vpt``: parallel arrays sharing ``e``'s leading shape
    (``()`` for a single curve, ``(B,)`` for a ``(B, H)`` batch, and so on).

    Attributes:
        value: float64, the number of valid steps (``VPT_tau``) wherever
            ``status`` is ``"crossed"`` (an exact count, including 0), or
            ``inf`` wherever ``status`` is ``"censored"``.
        status: ``"crossed"`` where some step ``>= tau`` was observed
            inside the horizon, else ``"censored"`` (right-censored: no
            crossing was observed on the available horizon).
        horizon: ``H``, the input's last-axis length -- the horizon each
            ``"censored"`` entry was censored AT. Recorded once (a plain
            int, not an array) since every element of a given call shares
            the same ``H`` by construction (one fixed input shape).
    """

    value: np.ndarray
    status: np.ndarray
    horizon: int


def vpt(e: Tensor | np.ndarray, tau: float) -> VPTResult:
    """Valid Prediction Time, reduced over the last axis of ``e``.

    Signature choice (one reducer for both a single curve and a batch):
    returns one pair of parallel arrays shaped like ``e``'s leading axes
    (a 0-d pair for a lone ``(H,)`` curve, ``(B,)`` for a ``(B, H)``
    batch) rather than a per-element result object, so both callers share
    the same vectorized code path and a batch is reduced in one call
    instead of a Python loop over a list of dataclasses.

    Args:
        e: already-normalized error curve, shape ``(..., H)`` -- e.g. the
            ``e_h`` of PROTOCOL.md section 12.1, with ``H`` physical steps
            on the last axis. Any leading shape, including none (a single
            curve, shape ``(H,)``).
        tau: validity threshold, applied as ``e_h < tau`` (strict). The
            project's declared value is ``0.5`` (PROTOCOL.md section
            12.2), passed in by the caller -- never hardcoded here.

    Returns:
        A ``VPTResult`` (see above).

    Raises:
        ValueError: ``tau`` is non-finite or negative; ``e`` is a 0-d
            scalar (no horizon axis to reduce); ``e``'s last axis (the
            horizon) is empty; or ``e`` contains non-finite values
            (NaN/inf in an error curve is an upstream bug -- never
            silently dropped, censored, or propagated as a NaN "result").
    """
    tau = float(tau)
    if not (math.isfinite(tau) and tau >= 0.0):
        raise ValueError(f"tau must be finite and >= 0, got {tau}")

    e = as_float64_array(e)
    if e.ndim == 0:
        raise ValueError("e must have shape (..., H); got a 0-d scalar with no horizon axis")
    H = e.shape[-1]
    if H == 0:
        raise ValueError("e's last axis (the horizon) is empty")
    if not np.isfinite(e).all():
        raise ValueError(
            "e contains non-finite values (NaN/inf); this is an upstream bug in "
            "the rollout/error computation, never a valid audit outcome -- fix "
            "the curve rather than passing it through"
        )

    invalid = e >= tau
    crossed = invalid.any(axis=-1)
    # argmax on an all-False row returns 0 (spurious); masked by `crossed` below.
    first_invalid = np.argmax(invalid, axis=-1)

    value = np.where(crossed, first_invalid.astype(np.float64), np.inf)
    status = np.where(crossed, "crossed", "censored")
    return VPTResult(value=value, status=status, horizon=H)


def fit_train_state_scale(x_train: Tensor | np.ndarray) -> np.ndarray:
    """Per-coordinate standard deviation on the TRAIN split, frozen before
    any evaluation (PROTOCOL.md section 12.1's ``sigma_i^train``).

    Args:
        x_train: shape ``(..., d)`` -- any number of leading axes (episodes,
            time steps, ...) are pooled into one population per coordinate
            before taking the standard deviation.

    Returns:
        ``sigma``, shape ``(d,)``, float64.

    ``ddof=0`` (population standard deviation, the same convention as
    scikit-learn's ``StandardScaler``): the train split is not a sample
    drawn to estimate some larger population's spread that this quantity
    should stay unbiased toward -- it IS the reference distribution every
    future ``e_h`` is normalized against. ``ddof=1`` would additionally be
    undefined for a coordinate observed only once.

    Policy for a constant coordinate (``sigma_i == 0``) is a REFUSAL, never
    an epsilon substitute: an epsilon divisor would make every downstream
    ``e_h`` at that coordinate a hidden function of the epsilon's magnitude
    rather than of the model, so the verdict would depend on a constant
    this module invented. Raises ``ValueError`` naming the offending
    coordinate index; the caller must drop the coordinate upstream or
    explain why it is constant.

    Raises:
        ValueError: ``x_train`` is not at least 1-d, contains non-finite
            values, or has a zero-variance coordinate.
    """
    x_train = as_float64_array(x_train)
    if x_train.ndim < 1:
        raise ValueError("x_train must have shape (..., d); got a 0-d scalar")
    d = x_train.shape[-1]
    flat = x_train.reshape(-1, d)
    if not np.isfinite(flat).all():
        raise ValueError("x_train contains non-finite values (NaN/inf)")

    sigma = flat.std(axis=0, ddof=0)
    zero_idx = np.flatnonzero(sigma == 0.0)
    if zero_idx.size > 0:
        raise ValueError(
            f"coordinate(s) {zero_idx.tolist()} have zero variance on the "
            "train split (sigma_i == 0); a constant coordinate is refused "
            "outright, never silently replaced by an epsilon (PROTOCOL.md "
            "section 12.1) -- drop the coordinate upstream or explain why "
            "it is constant"
        )
    return sigma


def normalized_state_error(
    pred: Tensor | np.ndarray,
    target: Tensor | np.ndarray,
    sigma_train: Tensor | np.ndarray,
) -> np.ndarray:
    """PROTOCOL.md section 12.1's normalized state error, one value per
    horizon step:

        e_h = sqrt( (1/(B*d)) * sum_b sum_i ((pred[b,h,i]-target[b,h,i])/sigma_i)^2 )

    Args:
        pred: the free-running rollout decoded to true-state coordinates,
            shape ``(B, H, d)``.
        target: the dataset's true state (the RK4 oracle), same shape.
        sigma_train: frozen per-coordinate scale from
            ``fit_train_state_scale``, shape ``(d,)``.

    Returns:
        ``e``, shape ``(H,)``, float64. ``h`` indexes physical steps
        ``1..H`` after the initial state (``h = 0`` is not scored -- the
        rollout starts from the true state, so it is never part of
        ``pred``/``target`` here); this array's index 0 IS ``e_1``.

    Refuses, each with a distinct message: a shape mismatch between
    ``pred`` and ``target``; ``pred``/``target`` not rank-3, or their
    trailing axis disagreeing with ``sigma_train``'s length; non-finite
    values in ``pred``, in ``target``, or in ``sigma_train``; and a
    ``sigma_train`` entry that is zero (re-checked here rather than
    trusted from the caller -- this is the one place a silent
    zero-division would corrupt every downstream ``e_h``).
    """
    pred = as_float64_array(pred)
    target = as_float64_array(target)
    sigma = as_float64_array(sigma_train)

    if pred.ndim != 3 or target.ndim != 3:
        raise ValueError(
            f"pred and target must have shape (B, H, d); got {pred.shape} and "
            f"{target.shape}"
        )
    if pred.shape != target.shape:
        raise ValueError(f"pred and target shapes disagree: {pred.shape} vs {target.shape}")
    if sigma.ndim != 1 or sigma.shape[0] != pred.shape[-1]:
        raise ValueError(
            f"sigma_train must have shape (d,) matching pred's trailing axis "
            f"d={pred.shape[-1]}; got {sigma.shape}"
        )

    if not np.isfinite(pred).all():
        raise ValueError("pred contains non-finite values (NaN/inf)")
    if not np.isfinite(target).all():
        raise ValueError("target contains non-finite values (NaN/inf)")
    if not np.isfinite(sigma).all():
        raise ValueError("sigma_train contains non-finite values (NaN/inf)")
    zero_idx = np.flatnonzero(sigma == 0.0)
    if zero_idx.size > 0:
        raise ValueError(
            f"sigma_train has zero entries at coordinate(s) {zero_idx.tolist()}; "
            "a zero scale is refused, never replaced by an epsilon"
        )

    B, H, d = pred.shape
    diff = (pred - target) / sigma  # broadcasts sigma (d,) over (B, H, d)
    return np.sqrt(np.sum(diff * diff, axis=(0, 2)) / (B * d))


def persistence_rollout(x0: Tensor | np.ndarray, horizon: int) -> np.ndarray:
    """PROTOCOL.md section 12.3's persistence baseline: ``x_h = x0`` for
    every ``h``.

    Args:
        x0: the last observed true state, shape ``(B, d)``.
        horizon: ``H``, number of steps to hold ``x0`` constant over.

    Returns:
        shape ``(B, horizon, d)`` -- ``horizon`` bit-identical copies of
        ``x0`` (a broadcast, not a recomputation, so every step is exactly
        equal to ``x0``, not merely close to it).

    Raises:
        ValueError: ``x0`` is not rank-2, or ``horizon < 1``.
    """
    x0 = as_float64_array(x0)
    if x0.ndim != 2:
        raise ValueError(f"x0 must have shape (B, d); got {x0.shape}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1; got {horizon}")
    return np.repeat(x0[:, None, :], horizon, axis=1)


def linear_extrapolation_rollout(
    x_minus_1: Tensor | np.ndarray,
    x0: Tensor | np.ndarray,
    horizon: int,
) -> np.ndarray:
    """PROTOCOL.md section 12.3's linear-extrapolation baseline:
    ``x_h = x0 + h * (x0 - x_minus_1)``.

    Index convention -- the trap this docstring exists to name: ``h`` runs
    ``1..horizon`` on the RETURNED array's step axis, never ``0..horizon-1``.
    The first returned step (index 0) is therefore
    ``x0 + 1*(x0 - x_minus_1)``, NOT ``x0``: a caller expecting
    ``output[:, 0] == x0`` has the wrong convention. This matches
    ``normalized_state_error``'s own indexing, where ``h = 0`` (the true
    state the rollout starts from) is never scored and index 0 of an
    ``e_h`` curve is already ``e_1``.

    Args:
        x_minus_1: the true state one step before ``x0``, shape ``(B, d)``.
        x0: the last observed true state, shape ``(B, d)``.
        horizon: ``H``, number of steps to extrapolate.

    Returns:
        shape ``(B, horizon, d)``.

    Raises:
        ValueError: ``x_minus_1`` and ``x0`` disagree in shape or are not
            rank-2, or ``horizon < 1``.
    """
    x_minus_1 = as_float64_array(x_minus_1)
    x0 = as_float64_array(x0)
    if x0.ndim != 2 or x_minus_1.shape != x0.shape:
        raise ValueError(
            f"x_minus_1 and x0 must both have shape (B, d) and agree; got "
            f"{x_minus_1.shape} and {x0.shape}"
        )
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1; got {horizon}")

    velocity = x0 - x_minus_1  # (B, d)
    h = np.arange(1, horizon + 1, dtype=np.float64)  # (horizon,), 1..horizon
    return x0[:, None, :] + h[None, :, None] * velocity[:, None, :]


class PositiveControlResult(NamedTuple):
    """Output of ``evaluate_positive_control`` (PROTOCOL.md section 12.3's
    positive-control gate).

    Attributes:
        lower_bound: ``e_{h_gate} < min(e_{h_gate}^pers, e_{h_gate}^lin) /
            gamma`` (strict).
        upper_bound: ``e_1 < tau_1`` (strict).
        m_low: PROTOCOL.md section 12.4's margin ratio for the lower
            bound: ``e_{h_gate} * gamma / min(e_pers, e_lin)``. ``< 1``
            reads as a PASS, and a larger value reads as a WORSE failure
            -- this is 12.4's own direction (distance from passing), not
            its reciprocal. An earlier draft of this file defined the
            reciprocal under this name; that was wrong and is not the
            convention here.
        m_high: section 12.4's margin ratio for the upper bound:
            ``e_1 / tau_1``. Same direction: ``< 1`` is a PASS.
        verdict: ``"learns"`` iff ``lower_bound`` and ``upper_bound`` are
            both ``True``, else ``"does_not_learn"``.
        near_miss: ``True`` iff ``verdict`` is ``"does_not_learn"`` and
            the MAXIMUM of the ratios of the bounds that actually FAILED
            lies in ``(1, 1.5]`` (PROTOCOL.md section 14.4). A cell that
            misses one bound by 1.1x and the other by 5x is NOT a near
            miss -- the 5x bound dominates the max, and 5x sits outside
            ``(1, 1.5]``. A passing bound never enters the max.
    """

    lower_bound: bool
    upper_bound: bool
    m_low: float
    m_high: float
    verdict: str
    near_miss: bool


def evaluate_positive_control(
    model_curve: Tensor | np.ndarray,
    persistence_curve: Tensor | np.ndarray,
    linear_curve: Tensor | np.ndarray,
    *,
    h_gate: int,
    gamma: float,
    tau_1: float,
) -> PositiveControlResult:
    """PROTOCOL.md section 12.3's positive-control gate: pure arithmetic on
    three already-computed ``e_h`` curves. No I/O, no artifact writing --
    that is a later wave's job. The returned ``m_low``/``m_high`` margin
    ratios follow section 12.4's convention (``< 1`` is a pass); see
    ``PositiveControlResult``.

    Args:
        model_curve, persistence_curve, linear_curve: shape ``(H,)``, the
            ``e_h`` curve for the model and for each trivial baseline, same
            indexing as ``normalized_state_error``'s output (index 0 is
            ``e_1``, so ``e_{h_gate} = curve[h_gate - 1]``).
        h_gate: PROTOCOL's ``H_gate`` (project value: 25). Every curve must
            have length ``>= h_gate``; a shorter curve is a REFUSAL --
            never silently reduced via ``min(H, h_gate)`` (PROTOCOL.md
            section 12.3 forbids exactly this substitution).
        gamma: PROTOCOL's ``gamma`` (project value: 5).
        tau_1: PROTOCOL's ``tau_1`` (``3 * e_recon`` per stratum, measured
            at milestone D2).

    Returns:
        A ``PositiveControlResult``.

    Asymmetry, deliberate: ``e_H`` and ``e_1`` (the NUMERATORS of the two
    ratios) may legitimately be ``0`` -- a perfect model gives ``m_low ==
    0.0`` and ``m_high == 0.0``, both correct passes, and neither is
    refused. Only the DENOMINATORS are constrained: ``gamma``, ``tau_1``,
    and ``min(e_pers_H, e_lin_H)`` must each be finite and ``> 0``, since
    each divides something. Refusing a zero numerator would be the ironic
    failure mode here -- rejecting the exact result this gate exists to
    reward.

    Raises:
        ValueError: ``gamma`` is non-finite or ``<= 0``; ``tau_1`` is
            non-finite or ``<= 0``; ``h_gate`` is not a positive integer;
            any curve is not rank-1, contains a non-finite value, or has
            fewer than ``h_gate`` entries (checked in this order so a bad
            curve is never misreported as a division problem); or
            ``min(e_pers_H, e_lin_H)`` is not finite and ``> 0`` (a
            baseline of exactly zero means the baseline predicted
            perfectly, which makes the lower-bound ratio meaningless
            rather than infinite -- a refusal, not a silent ``inf``).
    """
    gamma = float(gamma)
    if not (math.isfinite(gamma) and gamma > 0.0):
        raise ValueError(f"gamma must be finite and > 0, got {gamma}")

    tau_1 = float(tau_1)
    if not (math.isfinite(tau_1) and tau_1 > 0.0):
        raise ValueError(f"tau_1 must be finite and > 0, got {tau_1}")

    if isinstance(h_gate, bool) or not isinstance(h_gate, (int, np.integer)) or h_gate <= 0:
        raise ValueError(f"h_gate must be a positive integer, got {h_gate!r}")

    model = as_float64_array(model_curve)
    pers = as_float64_array(persistence_curve)
    lin = as_float64_array(linear_curve)

    for name, curve in (
        ("model_curve", model),
        ("persistence_curve", pers),
        ("linear_curve", lin),
    ):
        if curve.ndim != 1:
            raise ValueError(f"{name} must have shape (H,); got {curve.shape}")
        if not np.isfinite(curve).all():
            raise ValueError(f"{name} contains non-finite values (NaN/inf)")
        if curve.shape[0] < h_gate:
            raise ValueError(
                f"{name} has length {curve.shape[0]}, shorter than "
                f"h_gate={h_gate}; a short curve is refused outright, never "
                "silently reduced via min(H, h_gate) (PROTOCOL.md section 12.3)"
            )

    e_1 = float(model[0])
    e_H = float(model[h_gate - 1])
    e_H_pers = float(pers[h_gate - 1])
    e_H_lin = float(lin[h_gate - 1])

    baseline = min(e_H_pers, e_H_lin)
    if not (math.isfinite(baseline) and baseline > 0.0):
        raise ValueError(
            f"min(e_pers_H, e_lin_H) = {baseline} at h_gate={h_gate} is not "
            "> 0; a baseline that is exactly zero predicted perfectly, which "
            "makes the lower-bound ratio meaningless rather than infinite -- "
            "this is a refusal, not a silent inf"
        )
    lower_bound = e_H < baseline / gamma
    upper_bound = e_1 < tau_1

    # PROTOCOL.md section 12.4's own ratios, computed directly from its
    # formula (not derived from lower_bound/upper_bound above) -- kept as
    # an independent computation on purpose, so the two never silently
    # drift, and so equivalence between them is something this module can
    # be tested against rather than assumed.
    m_low = e_H * gamma / baseline
    m_high = e_1 / tau_1

    verdict = "learns" if (lower_bound and upper_bound) else "does_not_learn"
    if verdict == "does_not_learn":
        failing_ratios = [m for ok, m in ((lower_bound, m_low), (upper_bound, m_high)) if not ok]
        near_miss = 1.0 < max(failing_ratios) <= 1.5
    else:
        near_miss = False

    return PositiveControlResult(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        m_low=m_low,
        m_high=m_high,
        verdict=verdict,
        near_miss=near_miss,
    )
