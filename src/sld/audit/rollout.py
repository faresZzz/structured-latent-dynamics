"""Rollout-level offline audit.

Growing home for anything that scores a free-running rollout against ground
truth (PROTOCOL.md sections 12.1-12.2): today only ``vpt``; a later phase
adds ``normalized_state_error`` and the persistence/linear baselines
(PROTOCOL.md section 12.3) into this SAME file rather than a new one --
do not anticipate those here.

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
