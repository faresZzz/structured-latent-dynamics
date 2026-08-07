"""Single source of truth for the phase-D positive-control gate's constants
and PROTOCOL.md section 15.1's ``tau_1`` rule.

This module **is** the D0 validator -- the never-delivered checkbox of
milestone D0 (``phaseD-apprentissage-PROGRESS.md``'s own "Validateur" line;
read there for history, but the values below are the authority, not that
doc). ``H_GATE``/``GAMMA``/``TAU_VPT``/``GAMMA_1`` exist nowhere else in this
codebase: every caller imports them from here instead of repeating the
literal, so a real number can no longer drift silently between PROTOCOL.md's
prose and the code that gates a run.

``gate.py`` deliberately does NOT import these globals: ``evaluate_free_rollout``
and ``evaluate_positive_control`` keep explicit ``h_gate``/``gamma``/``tau_1``/
``tau`` arguments, because that module is a primitive whose own tests must be
free to vary each threshold independently of any project-wide default. This
module is the caller-side default, not a replacement for that contract.

Noisy refusals: every function here raises ``ValueError``, naming the
offending quantity, on a non-finite or non-positive input -- never a silent
NaN or negative value propagating into a gate verdict.
"""

from __future__ import annotations

import math

# PROTOCOL.md section 12.3 (H_gate, gamma), 12.2 (tau), 15.1 (gamma_1) --
# committed constants, never redefined at a call site.
H_GATE: int = 25  # PROTOCOL.md §12.3
GAMMA: float = 5.0  # §12.3
TAU_VPT: float = 0.5  # §12.2
GAMMA_1: float = 2.0  # §15.1


def _require_finite_positive(value: float, name: str) -> float:
    value = float(value)
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be finite and > 0, got {value}")
    return value


def tau_1(e_1_pers: float) -> float:
    """PROTOCOL.md section 15.1's upper-bound rule -- supersedes 12.3's
    ``3 * e_recon`` for D3/D4 onward: ``tau_1 = e_1_pers / GAMMA_1``.

    ``e_1_pers`` is the persistence baseline's one-step error, measured
    per-stratum at D2 on the val split with zero trained predictors --
    already the quantity ``FrozenRepresentationArtifact.manifest["e_1_pers"]``
    carries, so this rule adds no new measurement, only a new use of one
    already made.

    Raises:
        ValueError: ``e_1_pers`` is non-finite or ``<= 0``.
    """
    e_1_pers = _require_finite_positive(e_1_pers, "e_1_pers")
    return e_1_pers / GAMMA_1


def tau1_validity(e_recon: float, e_1_pers: float) -> dict:
    """PROTOCOL.md section 15.1's two validity conditions, checked at D2,
    published, never adjusted afterward:

    - ``attainable``: ``tau_1 > e_recon`` -- the direction 12.3's own sanity
      condition never checked: a ceiling below the measurement chain's own
      reconstruction floor is unattainable regardless of predictor quality.
    - ``discriminant``: ``tau_1 < e_1_pers`` -- true by construction as soon
      as ``GAMMA_1 > 1``; kept as an explicit, independently-computed check
      rather than assumed from the algebra.

    Both comparisons are strict; both are independent of each other.

    Raises:
        ValueError: ``e_recon`` or ``e_1_pers`` is non-finite or ``<= 0``.
    """
    e_recon = _require_finite_positive(e_recon, "e_recon")
    t1 = tau_1(e_1_pers)
    return {"attainable": t1 > e_recon, "discriminant": t1 < e_1_pers}
