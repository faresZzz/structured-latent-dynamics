"""Reusable reparameterization stress test (REGLES.md section 4: "Invariance
obligatoire -- aucun diagnostic n'entre dans une figure ou une conclusion
sans avoir passe le stress test de reparametrisation (`audit/invariance.py`).
Un diagnostic qui echoue est repare ou retire").

This module is deliberately generic: it knows nothing about Jacobians,
gauges, or readouts. It only knows how to build a fixed, named battery of
finite invertible diagonal reparameterizations ``D`` and compare a caller-
supplied scalar measurement across them. Any diagnostic that DECLARES
invariance under ``q' = D q`` for finite invertible diagonal ``D`` -- the
gauge domain PROTOCOL.md section 10.3 declares for the normalized Jacobian
margin, and the one section 5 point 4 already declares for the fig1 P1a
readout -- can be checked with the same tool by supplying a ``measure``
closure that applies ``D`` however that diagnostic's own math requires.

Two complementary checks:

- ``check_diagonal_gauge_invariance``: the POSITIVE claim -- the measured
  value is (numerically) unchanged across every tested ``D``, including
  signed and extreme-scale ones (``1e-20``, ``1e+20``).
- ``check_shear_breaks_invariance``: the DECLARED LIMIT, made explicit
  rather than left as a silent gap -- a diagnostic invariant under diagonal
  reparameterization is generally NOT invariant under a shear or a rotation,
  and PROTOCOL section 10.3 says so explicitly. This function lets a test
  confirm the quantity actually DOES move under a shear, which is the
  constructive counter-test that a passing diagonal check alone cannot
  provide (REGLES section 4's own worked example: a diagnostic that would
  ALSO pass on the un-normalized quantity proves nothing).
"""

from __future__ import annotations

import math
from typing import Callable, NamedTuple

import numpy as np


class GaugeCheckResult(NamedTuple):
    """One named diagonal gauge's outcome against the reference (identity
    gauge) measurement.

    Attributes:
        label: which named diagonal this is (see ``diagonal_gauges``).
        value: the measurement under this diagonal.
        reference: the measurement under the identity diagonal.
        relative_diff: ``|value - reference| / |reference|`` (or the raw
            absolute difference when ``reference == 0``, since a relative
            difference against zero is undefined, never a divide-by-zero).
        invariant: ``relative_diff <= rtol``.
    """

    label: str
    value: float
    reference: float
    relative_diff: float
    invariant: bool


def diagonal_gauges(d: int, *, seed: int = 0) -> dict[str, np.ndarray]:
    """A fixed, named battery of test diagonals for the declared gauge
    domain (PROTOCOL.md section 10.3): signs, extreme scales (``1e-20``,
    ``1e+20``), a mix of both extremes in the SAME diagonal, and two random
    draws -- one positive, one signed-and-extreme. Every entry is finite,
    non-zero, and finite-invertible (``1/entry`` finite too).

    Args:
        d: dimension of the diagonal.
        seed: RNG seed for the random entries -- fixed by default so a
            failing check is reproducible.

    Returns:
        ``{label: (d,) float64 array}``, always including ``"identity"``
        (used as the reference by ``check_diagonal_gauge_invariance``).
    """
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}")
    rng = np.random.default_rng(seed)
    return {
        "identity": np.ones(d, dtype=np.float64),
        "signed": (rng.integers(0, 2, size=d) * 2 - 1).astype(np.float64)
        * (0.5 + rng.random(d)),
        "extreme_large": np.full(d, 1e20, dtype=np.float64),
        "extreme_small": np.full(d, 1e-20, dtype=np.float64),
        "mixed_extreme": np.where(np.arange(d) % 2 == 0, 1e20, 1e-20).astype(np.float64),
        "random_positive": 0.1 + 10.0 * rng.random(d),
        "random_signed_extreme": rng.choice([-1e15, 1e15], size=d).astype(np.float64),
    }


def check_diagonal_gauge_invariance(
    measure: Callable[[np.ndarray], float],
    d: int,
    *,
    seed: int = 0,
    rtol: float = 1e-6,
    gauges: dict[str, np.ndarray] | None = None,
) -> list[GaugeCheckResult]:
    """Evaluate ``measure`` under every diagonal of ``diagonal_gauges(d,
    seed=seed)`` (or a caller-supplied ``gauges`` battery) and compare each
    to the ``"identity"`` reference.

    Args:
        measure: ``diag -> float``, closing over whatever the specific
            diagnostic needs (a step function, a Jacobian, a gauge object)
            to apply the reparameterization ``q' = diag . q`` and return
            the scalar under test. Must be finite for the identity gauge
            (checked below) -- a non-finite reference makes "unchanged"
            meaningless, loi du fini (REGLES.md section 5).
        d: dimension of the diagonal ``measure`` expects.
        seed, gauges: see ``diagonal_gauges``; ``gauges`` overrides the
            default battery entirely when given (still must include
            ``"identity"``).
        rtol: relative tolerance for "unchanged" (default ``1e-6`` --
            float64 dense linear algebra headroom; the declared invariance
            is EXACT in exact arithmetic, PROTOCOL section 10.3, so this is
            a floating-point tolerance, not a physical one).

    Returns:
        One ``GaugeCheckResult`` per gauge in the battery, in the same
        order (Python 3.7+ dict insertion order), including
        ``"identity"`` itself (trivially invariant, kept in the report
        rather than filtered so the report is self-documenting).

    Raises:
        ValueError: ``gauges`` (if given) omits ``"identity"``, or the
            identity-gauge measurement is non-finite.
    """
    battery = gauges if gauges is not None else diagonal_gauges(d, seed=seed)
    if "identity" not in battery:
        raise ValueError('gauges must include an "identity" entry as the reference')
    reference = float(measure(battery["identity"]))
    if not math.isfinite(reference):
        raise ValueError(
            f"reference (identity gauge) measurement is non-finite ({reference!r}); "
            "cannot check invariance against an undefined reference"
        )
    results = []
    for label, diag in battery.items():
        value = float(measure(diag))
        if reference != 0.0:
            rel = abs(value - reference) / abs(reference)
        else:
            rel = abs(value - reference)
        results.append(
            GaugeCheckResult(
                label=label, value=value, reference=reference, relative_diff=rel, invariant=rel <= rtol
            )
        )
    return results


def assert_diagonal_gauge_invariant(
    measure: Callable[[np.ndarray], float],
    d: int,
    *,
    seed: int = 0,
    rtol: float = 1e-6,
    gauges: dict[str, np.ndarray] | None = None,
) -> list[GaugeCheckResult]:
    """``check_diagonal_gauge_invariance``, raising ``AssertionError`` naming
    every gauge that failed instead of returning a report to inspect.
    Returns the full (all-passing) result list on success, so a caller can
    still log it."""
    results = check_diagonal_gauge_invariance(measure, d, seed=seed, rtol=rtol, gauges=gauges)
    failed = [r for r in results if not r.invariant]
    if failed:
        detail = "; ".join(
            f"{r.label}: value={r.value:.6g} reference={r.reference:.6g} "
            f"relative_diff={r.relative_diff:.3g}"
            for r in failed
        )
        raise AssertionError(f"diagonal gauge invariance FAILED for: {detail}")
    return results


def check_shear_breaks_invariance(
    measure_reference: Callable[[], float],
    measure_shear: Callable[[], float],
    *,
    rtol: float = 1e-6,
) -> GaugeCheckResult:
    """Declared-limit check (PROTOCOL.md section 10.3): the normalized
    quantity is invariant under diagonal reparameterization but NOT under a
    shear or rotation, and the protocol says so explicitly rather than
    leaving it untested. Confirms the quantity genuinely MOVES under a
    shear, the constructive counter-test that a passing diagonal-only check
    cannot provide on its own (REGLES.md section 4).

    Args:
        measure_reference: ``() -> float``, the quantity with no
            reparameterization applied.
        measure_shear: ``() -> float``, the SAME quantity with a shear (or
            rotation) applied instead of a diagonal.
        rtol: as ``check_diagonal_gauge_invariance``.

    Returns:
        A ``GaugeCheckResult`` labeled ``"shear"`` whose ``invariant``
        field is ``True`` if the quantity happened to stay put (an
        UNDECLARED extra invariance the caller should treat as suspicious,
        not a pass) and ``False`` if it moved (confirms the documented
        limitation -- the expected, declared outcome).
    """
    reference = float(measure_reference())
    if not math.isfinite(reference):
        raise ValueError(f"reference measurement is non-finite ({reference!r})")
    value = float(measure_shear())
    rel = abs(value - reference) / abs(reference) if reference != 0.0 else abs(value - reference)
    return GaugeCheckResult(
        label="shear", value=value, reference=reference, relative_diff=rel, invariant=rel <= rtol
    )
