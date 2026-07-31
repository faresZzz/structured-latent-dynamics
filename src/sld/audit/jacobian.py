"""Quantitative Jacobian objects for the offline audit.

Audit doctrine (REGLES section 5): this module computes the objects the
audit *measures* -- as opposed to ``spectral.py``, which is what the audit
*displays*. Everything here is CPU float64, dense, and deterministic; it is
the counterpart of ``objectives.jac_reg.TotalJacobianPenalty`` (the
STOCHASTIC Hutchinson estimator used during training). Never reuse that
estimator here, and never call these functions from training code -- they
are for checkpoint audits only.

Two numerical rules apply everywhere below:

1. Every Jacobian here is the total Jacobian of the AUGMENTED state
   ``q = (h, m)`` (``predictors.base`` semantics), never the frozen-context
   conditional pair returned by ``Predictor.local_operator``. The full
   augmented Jacobian is the only object that sees the ``h``-loop and the
   control-through-scheduling paths.
2. Raw products of Jacobians are NEVER formed beyond a handful of steps --
   they overflow/underflow in float64 well before 20 multiplications for
   any mildly expanding or contracting system. Every multi-step quantity
   here goes through ``jacobian_products``, which propagates an
   orthonormal tangent basis via QR at every step and accumulates
   ``log|diag R_t|`` instead of the raw product.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, NamedTuple

import numpy as np
import torch
from torch import Tensor

from sld.audit._common import as_float64_array, require_cpu_float64_tensor
from sld.audit.spectral import operator_norm

# Wave 5 round 6 amendment (PROTOCOL.md section 10.3): the declared-metric
# gauge transform is owned by objectives.jac_reg, not reimplemented here --
# TotalJacobianPenalty applies the IDENTICAL S_out^-1 . (.) . S_in transform
# elementwise to JVP probes at training time; this module applies it to the
# dense excluded Jacobian for the offline audit. The dependency runs THIS
# WAY (audit -> objectives) and not the reverse: REGLES.md section 2 places
# objectives/ in the "modele appris (encoder, predictor, loss)" layer and
# audit/ in the downstream "mesure un modele appris sans le modifier" layer
# -- the same directionality REGLES section 10 already uses for
# calibration/ depending on audit/ and training/, never the inverse.
from sld.objectives.jac_reg import Gauge, normalize_jacobian

# A row of d(m_next)/d(m) counts as an identity passthrough when it is a
# canonical basis vector to this absolute tolerance. A genuine verbatim copy
# (torch.cat of the old cache -- the ACTUAL transformer's KV-cache update) is
# EXACT under autograd: the VJP of a cat/slice copy is a pure gather with no
# floating-point arithmetic, so it differentiates to EXACTLY 1.0/0.0 at any
# d. This tolerance instead guards a hypothetical family that routes the same
# copy through an identity matmul, which accumulates ~d * eps float64
# rounding. At the largest d this audit has actually measured -- the
# registered transformer's augmented input dimension of 7,924 (PROTOCOL
# section 5 calibration, t=99) -- that worst-case noise floor is
# 7924 * 2.22e-16 ~= 1.8e-12, ABOVE the previous 1e-12 default (corrected
# here; the earlier "d <= 10^3" justification is what the real transformer
# falsifies). 1e-6 sits ~6 decades above that worst-case noise floor and ~6
# decades below the smallest structurally different coefficient a real,
# trained non-copy map could produce by coincidence (an entry landing within
# 1e-6 of exactly 1.0 while every OTHER entry in the same row lands within
# 1e-6 of exactly 0.0 is not a configuration a trained network produces by
# chance), so the classification has no grey zone even at this dimension.
_IDENTITY_ROW_ATOL = 1e-6

# identity_atol must stay inside this range: above it, a genuinely learned
# (non-copy) coefficient risks being misclassified as an identity passthrough
# and silently deleted -- the adversarial reviewer's poison-pill probe,
# identity_atol=0.1, would misclassify a 0.95*m derivative as a verbatim
# copy. At or below zero, or non-finite, the classification is undefined.
#
# The bound is sandwiched between two measured quantities, not chosen freely:
# - FLOOR: the rounding noise expected for the largest d this audit has
#   actually registered -- the transformer's augmented input dimension of
#   7,924 (PROTOCOL section 5 calibration, t=99) -- is O(d * eps) ~=
#   7924 * 2.22e-16 ~= 1.8e-12. The bound must stay above that.
# - CEILING: the smallest "learned" (non-copy) deviation this audit must
#   never delete. ``m_next = cat([0.9995 * m, new])`` differs from a
#   verbatim copy by 5e-4 per entry -- a plausible trained coefficient, not
#   a copy -- so the bound must stay below that.
# 1e-4 satisfies both, honestly quantified: log10(1e-4 / 1.8e-12) ~= 7.7, so
# the bound sits ~7.7 decades above the noise floor -- comfortable margin on
# that side. log10(5e-4 / 1e-4) ~= 0.7, i.e. only a factor of 5 below the
# smallest learned deviation the bound must not delete -- tight, but on the
# right side (the previous 1e-3 bound failed this: 5e-4 < 1e-3 would have
# deleted it). The default, ``_IDENTITY_ROW_ATOL = 1e-6``, sits far more
# conservatively on the ceiling side: log10(5e-4 / 1e-6) ~= 2.7 decades of
# margin below that same deviation.
_IDENTITY_ATOL_MAX = 1e-4


def total_jacobian(step_fn: Callable[[Tensor, Tensor], Tensor], q: Tensor, u: Tensor) -> np.ndarray:
    """Dense total Jacobian ``J^q = d(q_next)/dq`` of the augmented state.

    This is the audit's ``J^q`` of the augmented state ``q = (h, m)`` -- the
    object named in ``objectives.jac_reg`` and in ``predictors.base``'s
    doctrine, computed exactly and deterministically via
    ``torch.func.jacrev`` (never the stochastic Hutchinson estimator, and
    never the frozen-context pair from ``local_operator``).

    Args:
        step_fn: pure function ``(q, u) -> q_next`` with ``q``: ``(d_q,)``,
            ``u``: ``(d_u,)``, called with no side effects.
        q: augmented state, ``(d_q,)``, CPU float64.
        u: control/input, ``(d_u,)``, CPU float64.

    Returns:
        ``J`` of shape ``(d_q, d_q)``, float64 numpy, ``J[i, j] =
        d(q_next)_i / d(q)_j`` evaluated at ``(q, u)``.

    Raises:
        ValueError: if ``q``/``u`` are not 1-D CPU float64 tensors, or if
            ``step_fn(q, u)`` does not return a tensor of the same shape
            as ``q``.
    """
    require_cpu_float64_tensor(q, "q")
    require_cpu_float64_tensor(u, "u")
    if not torch.isfinite(q).all():
        raise ValueError("q contains non-finite values")
    if not torch.isfinite(u).all():
        raise ValueError("u contains non-finite values")
    if q.dim() != 1:
        raise ValueError(f"q must be 1-D (d_q,), got shape {tuple(q.shape)}")
    if u.dim() != 1:
        raise ValueError(f"u must be 1-D (d_u,), got shape {tuple(u.shape)}")

    q_next = step_fn(q, u)
    require_cpu_float64_tensor(q_next, "step_fn(q, u) output")
    if not torch.isfinite(q_next).all():
        raise ValueError("step_fn(q, u) output contains non-finite values")
    if q_next.shape != q.shape:
        got = tuple(q_next.shape)
        raise ValueError(
            f"step_fn(q, u) must return a tensor of the same shape as q "
            f"(the augmented state), got {got} vs {tuple(q.shape)}"
        )

    J = as_float64_array(torch.func.jacrev(step_fn, argnums=0)(q, u))
    if not np.isfinite(J).all():
        raise ValueError("J contains non-finite values; derivative of invalid function")
    return J


class ExcludedAugmentedJacobian(NamedTuple):
    """What ``augmented_jacobian_excluding_identity`` reports.

    Attributes:
        jacobian: dense excluded Jacobian, ``(d_out - n_identity_rows,
            d_in)`` float64. Input coordinates are ``vec(h)`` then
            ``vec(m)`` (row-major over each tensor AS GIVEN -- this module
            never interprets ``m``'s axes); output coordinates are
            ``vec(h_next)`` then the NON-copied coordinates of
            ``vec(m_next)``, in their original order.
        sigma_max: largest singular value of ``jacobian`` -- the quantity
            the calibration takes the median of (PROTOCOL section 5 step 2).
        frobenius_sq: ``||jacobian||_F^2``. Reported because it is the exact
            counterpart of what ``objectives.jac_reg`` estimates at training
            time; the calibration band is on ``sigma_max``, and the gap
            between the two is honesty declaration (a) of PROTOCOL section 5.
        n_identity_rows: number of excluded rows. Equals ``m.numel()``
            whenever the exclusion applied, ``0`` when it did not.
        d_in: ``h.numel() + m.numel()``, the augmented input dimension.
        excluded_row_indices: absolute row indices (into the FULL,
            un-excluded ``vec(h_next) ++ vec(m_next)`` output, before
            deletion) that were removed to produce ``jacobian``. Exposed
            (Wave 5 round 6 amendment, PROTOCOL section 10.3) so a caller
            applying the declared gauge normalization can align its own
            full-length ``S_out`` reference-scale vector to ``jacobian``'s
            actual (post-deletion) rows via ``np.delete(s_out_full,
            excluded_row_indices)`` -- never by recomputing the identity-row
            detection a second time.
    """

    jacobian: np.ndarray
    sigma_max: float
    frobenius_sq: float
    n_identity_rows: int
    d_in: int
    excluded_row_indices: np.ndarray


def augmented_jacobian_excluding_identity(
    step_fn: Callable[[Tensor, Tensor | None, Tensor], tuple[Tensor, Tensor | None]],
    h: Tensor,
    m: Tensor | None,
    u: Tensor,
    *,
    identity_atol: float = _IDENTITY_ROW_ATOL,
) -> ExcludedAugmentedJacobian:
    """Dense ``J^q`` of a growing augmented state, minus the identity passthrough.

    The measurement counterpart of the training-time exclusion declared in
    ``objectives.jac_reg.TotalJacobianPenalty.on_state_steps``: same object,
    computed exactly (dense ``torch.func.jacrev``, CPU float64,
    deterministic) instead of by Hutchinson probes. This is what the
    calibration of PROTOCOL section 5 measures.

    Why this function exists. ``total_jacobian`` requires ``q_next`` to have
    the SAME shape as ``q`` -- a square map is what ``jacobian_products``
    and ``fr_lyapunov_exponents`` need downstream. A growing-memory family
    (the transformer, whose KV cache gains one entry per step) violates that
    from ``t >= 1`` onward, so ``total_jacobian`` raises for it rather than
    returning a rectangular object its callers would misuse. This function
    is the separate rectangular entry point; ``total_jacobian``'s contract
    is unchanged.

    WHAT IS EXCLUDED, AND WHY. Under pure append, ``m_next`` reproduces
    every entry of ``m`` verbatim, so ``d(m_next)/d(m)`` contains an exact
    identity block. It forces ``||J^q||_F^2 >= m.numel()`` and
    ``sigma_max(J^q) >= 1`` -- structurally, for any parameters -- which
    would put the declared margin band ``[0.90, 1.00]`` out of reach for
    that family by construction. The block is excluded, and the exclusion is
    declared rather than hidden: it is a copy, not a learned map, its
    derivative with respect to the parameters is identically zero, and no
    penalty weight could ever move it. Removing it removes a constant, not a
    constraint.

    HOW, and why ROW DELETION rather than a singular-value correction. The
    identity rows are found NUMERICALLY -- rows of the ``m_next`` block that
    are canonical basis vectors pointing into the ``m`` input block, to
    ``identity_atol`` -- never from knowledge of ``m``'s layout, which stays
    family-private (``predictors.base``; Liskov: the audit must not learn a
    family's cache layout). Those rows are then deleted. Writing ``R`` for
    that row set, and using that its rows are the basis vectors ``e_j`` with
    ``j`` ranging over the ``m``-input block exactly once (a partial
    permutation, hence norm-preserving), for every tangent ``v = (v_h,
    v_m)``::

        ||J v||^2 = ||J_excl v||^2 + ||v_m||^2

    equivalently, in Gram form and exactly::

        J_excl^T J_excl = J^T J - diag(0_{n_h}, I_{n_m})

    which is the identity the training estimator uses at probe level (it
    subtracts ``||v_m||^2``); the two therefore penalize and measure the
    SAME object. It also shows why a singular-value-level shortcut is NOT
    available: the two Gram matrices differ by a PSD rank-``n_m``
    projection, so Weyl gives only ``sigma_max(J_excl)^2 >=
    sigma_max(J)^2 - 1``, with equality iff the top right-singular vector of
    ``J_excl`` lies entirely inside the ``m`` block. Measured on the real
    transformer at ``t = 1``: ``sigma_max(J) = 2.0303``,
    ``sigma_max(J_excl) = 2.0196``, while ``sqrt(sigma_max(J)^2 - 1) =
    1.767``. The excluded operator norm is not recoverable from the full
    spectrum -- the rows must be identified and removed.

    VERIFIED, NOT ASSUMED. When the state grows, the identity rows must form
    a bijection onto ALL of ``m``'s coordinates; anything else means the
    family does not append purely and the identity above is false for it, so
    this raises instead of silently reporting a wrong number.

    Args:
        step_fn: pure map ``(h, m, u) -> (h_next, m_next)``, ``m``/``m_next``
            a tensor or ``None``. Same signature as the ``step_fn`` of
            ``objectives.jac_reg.on_state_steps``, so a caller can feed the
            identical closure to the penalty and to this audit.
        h: latent state, any shape, CPU float64. The batch axis is the
            caller's business: pass ONE sample for a per-state margin (the
            batched map's Jacobian is block-diagonal, and mixing samples
            into a single ``sigma_max`` would report the batch max, not a
            per-state quantity).
        m: threaded memory, any shape (opaque here) or ``None`` for a fresh
            state, CPU float64.
        u: control, any shape, CPU float64. Frozen: the differentiated
            object is ``J^q``, not a mixed Jacobian (same doctrine as
            ``total_jacobian``).
        identity_atol: absolute tolerance classifying a row as a canonical
            basis vector (see ``_IDENTITY_ROW_ATOL`` for why the default has
            no grey zone). Must be finite and in ``(0, _IDENTITY_ATOL_MAX]``
            -- validated up front, since a looser value (e.g. ``0.1``) would
            silently reclassify a genuine learned coefficient as a verbatim
            copy and delete it.

    Returns:
        ``ExcludedAugmentedJacobian``. When ``m is None`` (fresh state) or
        ``m_next.numel() == m.numel()`` (fixed-size recurrent state such as
        ``min_ssm``, where ``m_next`` is a transformation of ``m`` and not a
        copy of it), NOTHING is excluded and the full dense Jacobian is
        returned with ``n_identity_rows == 0``.

    Raises:
        ValueError: ``identity_atol`` non-finite or outside ``(0,
            _IDENTITY_ATOL_MAX]``; non-CPU/non-float64/non-finite inputs or
            outputs; ``step_fn`` not returning a 2-tuple; a SHRINKING memory
            (outside the declared taxonomy); for a growing memory, identity
            rows that do not form a bijection onto ``m``'s coordinates
            (falsifies the pure-append hypothesis for that family); or
            ``sigma_max``/``frobenius_sq`` overflowing float64 despite a
            finite ``J`` (M8: a loud, named error, never a silent ``inf``).
    """
    if not (math.isfinite(identity_atol) and 0.0 < identity_atol <= _IDENTITY_ATOL_MAX):
        raise ValueError(
            f"identity_atol must be finite and in (0, {_IDENTITY_ATOL_MAX!r}], got "
            f"{identity_atol!r} -- a looser tolerance risks misclassifying a "
            "genuine learned coefficient as a verbatim identity copy and "
            "silently deleting it"
        )
    require_cpu_float64_tensor(h, "h")
    require_cpu_float64_tensor(u, "u")
    if not torch.isfinite(h).all():
        raise ValueError("h contains non-finite values")
    if not torch.isfinite(u).all():
        raise ValueError("u contains non-finite values")
    if m is not None:
        require_cpu_float64_tensor(m, "m")
        if not torch.isfinite(m).all():
            raise ValueError("m contains non-finite values")

    n_h = h.numel()
    n_m = 0 if m is None else m.numel()

    with torch.no_grad():
        primal = step_fn(h, m, u)
    if not (isinstance(primal, tuple) and len(primal) == 2):
        raise ValueError(
            "step_fn must return a 2-tuple (h_next, m_next); got "
            f"{type(primal).__name__} of length "
            f"{len(primal) if isinstance(primal, tuple) else 'n/a'}"
        )
    h_next, m_next = primal
    require_cpu_float64_tensor(h_next, "step_fn h_next output")
    if not torch.isfinite(h_next).all():
        raise ValueError("step_fn h_next output contains non-finite values")
    n_h_next = h_next.numel()
    if m_next is None:
        n_m_next = 0
    else:
        require_cpu_float64_tensor(m_next, "step_fn m_next output")
        if not torch.isfinite(m_next).all():
            raise ValueError("step_fn m_next output contains non-finite values")
        n_m_next = m_next.numel()

    if n_m > 0 and n_m_next < n_m:
        raise ValueError(
            f"m_next has {n_m_next} elements, fewer than m's {n_m}; a shrinking "
            "recurrent memory is outside the declared taxonomy (fixed-size state "
            "or pure append) and the identity-passthrough exclusion is undefined "
            "for it"
        )

    h_shape, m_shape = h.shape, None if m is None else m.shape

    # One flat coordinate system for the whole augmented state: vec(h) then
    # vec(m), each row-major over the tensor AS GIVEN. Flattening is the only
    # thing done to m -- its axes are never interpreted (family-private).
    def flat_map(q_flat: Tensor) -> Tensor:
        h_arg = q_flat[:n_h].reshape(h_shape)
        m_arg = None if m_shape is None else q_flat[n_h:].reshape(m_shape)
        out_h, out_m = step_fn(h_arg, m_arg, u)
        parts = [out_h.reshape(-1)]
        if out_m is not None:
            parts.append(out_m.reshape(-1))
        return torch.cat(parts)

    q_flat = h.reshape(-1) if m is None else torch.cat([h.reshape(-1), m.reshape(-1)])
    J = as_float64_array(torch.func.jacrev(flat_map)(q_flat))
    if not np.isfinite(J).all():
        raise ValueError("J contains non-finite values; derivative of invalid function")

    d_in = n_h + n_m
    excluded_rows: np.ndarray = np.empty(0, dtype=np.intp)
    if n_m > 0 and n_m_next > n_m:
        # Layout-agnostic detection: a row of the m_next block realizes the
        # passthrough iff it is a canonical basis vector pointing into the
        # m-input block. Nothing here knows WHICH slots were appended.
        block = J[n_h_next:, :]
        near_one = np.abs(block - 1.0) <= identity_atol
        n_nonzero = (np.abs(block) > identity_atol).sum(axis=1)
        is_identity = (near_one.sum(axis=1) == 1) & (n_nonzero == 1)
        col = np.argmax(near_one, axis=1)
        local_rows = np.nonzero(is_identity & (col >= n_h))[0]
        matched = np.unique(col[local_rows])
        expected = np.arange(n_h, n_h + n_m)
        if local_rows.size != n_m or not np.array_equal(matched, expected):
            raise ValueError(
                f"pure-append hypothesis violated: found {local_rows.size} identity "
                f"passthrough rows covering {matched.size} distinct m coordinates, "
                f"expected {n_m} rows in bijection with m's {n_m} coordinates "
                f"(m: {n_m} -> {n_m_next} elements). This family does NOT copy its "
                "old memory entries verbatim into m_next, so the declared "
                "identity-passthrough exclusion does not apply to it (REGLES "
                "section 5: a violated hypothesis is a bug to chase upstream, "
                "never a number to report anyway)"
            )
        excluded_rows = local_rows + n_h_next

    J_excl = np.delete(J, excluded_rows, axis=0) if excluded_rows.size else J

    # M8 (adversarial review round 2): a FINITE J can still overflow float64
    # in its DERIVED norms -- e.g. step_fn(h) = 1e200*h is finite, sigma_max
    # (1e200) is a legitimate finite report, but naive sum(J**2) hits 1e400
    # and silently returns inf. Divide by max|J| BEFORE squaring (REGLES
    # section 5: same discipline as fr_drift_curve's "divide before
    # subtracting") so (J_excl/scale) has |entries| <= 1 and its
    # sum-of-squares is always representable; take the sqrt BEFORE squaring
    # back up so the only remaining place overflow can occur is the FINAL
    # squaring of a value on the same order as `scale` itself (already
    # confirmed finite by np.linalg.svd's own internal scaling). If even
    # THAT overflows, the true frobenius_sq genuinely exceeds float64 range,
    # and an explicit named error is the correct report -- never a silent inf.
    if J_excl.size == 0:
        sigma_max = 0.0
        frobenius_sq = 0.0
    else:
        sigma_max = float(np.linalg.svd(J_excl, compute_uv=False)[0])
        if not math.isfinite(sigma_max):
            raise ValueError(
                f"sigma_max overflowed float64 ({sigma_max!r}) despite a finite "
                "J_excl; the spectral norm itself is not representable here"
            )
        scale = float(np.abs(J_excl).max())
        if scale == 0.0:
            frobenius_sq = 0.0
        else:
            norm_sq_scaled = float(((J_excl / scale) ** 2).sum())  # in [0, size], always finite
            frob = scale * math.sqrt(norm_sq_scaled)  # same order as scale -- finite
            # guard BEFORE squaring: Python float ** raises a bare
            # OverflowError past float64 max instead of returning inf, so
            # the finite-postcheck below would never run (M8: the error must
            # be the loud, NAMED one). frob*frob (multiplication) yields inf
            # instead of raising, letting the named check fire.
            frobenius_sq = frob * frob
            if not math.isfinite(frobenius_sq):
                raise ValueError(
                    f"frobenius_sq overflowed float64: ||J_excl||_F ~= {frob:.6g} "
                    "exceeds sqrt(float64 max) ~= 1.34e154 -- sigma_max "
                    f"({sigma_max:.6g}) is still a legitimate, representable report; "
                    "only the squared Frobenius norm is out of range here (REGLES "
                    "section 5: an unrepresentable value is a loud error, never a "
                    "silent inf)"
                )
            # Symmetric guard on the OTHER side (m9, adversarial review round
            # 3): a positive frob this small still squares to exactly 0.0 --
            # float64's smallest positive (subnormal) value is ~4.94e-324, so
            # any frob < sqrt(float64 min normal) ~= 1.49e-154 underflows
            # silently to zero on squaring even though frob itself, and
            # sigma_max, remain exact finite reports. Never let that silent
            # zero pass as a legitimate frobenius_sq.
            if frob > 0.0 and frobenius_sq == 0.0:
                raise ValueError(
                    f"frobenius_sq underflowed float64: ||J_excl||_F ~= {frob:.6g} "
                    "< sqrt(float64 min normal) ~= 1.49e-154 -- the squared norm is "
                    f"not representable; sigma_max ({sigma_max:.6g}) remains an "
                    "exact, legitimate report (REGLES section 5: an unrepresentable "
                    "value is a loud error, never a silent zero)"
                )

    return ExcludedAugmentedJacobian(
        jacobian=J_excl,
        sigma_max=sigma_max,
        frobenius_sq=frobenius_sq,
        n_identity_rows=int(excluded_rows.size),
        d_in=d_in,
        excluded_row_indices=excluded_rows,
    )


def augmented_sigma_max_normalized(
    step_fn: Callable[[Tensor, Tensor | None, Tensor], tuple[Tensor, Tensor | None]],
    h: Tensor,
    m: Tensor | None,
    u: Tensor,
    gauge: Gauge,
    *,
    identity_atol: float = _IDENTITY_ROW_ATOL,
) -> float:
    """PROTOCOL.md section 10.3's declared metric:
    ``sigma_max(S_out^-1 . J_t^{q,excl} . S_in)``.

    The SAME excluded object ``augmented_jacobian_excluding_identity``
    computes raw, now normalized by the frozen reference scale ``gauge``
    via ``objectives.jac_reg.normalize_jacobian`` -- the single shared
    definition of the gauge transform (its module docstring proves this is
    the SAME normalized operator ``TotalJacobianPenalty`` targets at
    training time via elementwise JVP scaling, never a dense Jacobian
    there).

    Why identity-row DETECTION still runs on the RAW Jacobian, never a
    normalized one: ``h`` and ``m`` can differ in reference scale by many
    orders of magnitude -- min_ssm's ``cond(Sigma) ~= 1.5e7`` is the whole
    motivation for this amendment -- so normalizing FIRST would rescale
    the ``~1e-12`` float64 rounding noise ``_IDENTITY_ROW_ATOL`` is tuned
    against by that same huge factor, risking a false "not a copy"
    classification on a genuine verbatim-copy row. Reusing the RAW
    exclusion here is still exactly correct for the NORMALIZED object
    because of ``Gauge.s_m_positions``'s declared property: a persisting
    cache coordinate's ``S_out`` equals its ``S_in`` EXACTLY (the same
    frozen per-position slice, reused), so the normalized Jacobian's copy
    rows are STILL an exact identity block onto the same input columns --
    the row set excluded on the raw matrix is exactly the row set that
    would be excluded on the normalized one.

    Args:
        step_fn, h, m, u, identity_atol: as
            ``augmented_jacobian_excluding_identity``.
        gauge: the frozen reference scale (``experiments.selection``
            estimates it -- PROTOCOL section 10.3's "Reference scales").

    Returns:
        ``sigma_max`` of the gauge-normalized excluded Jacobian.

    Raises:
        MetricIndeterminate (``objectives.jac_reg``): any coordinate
            ``gauge`` needs is zero, non-finite, or (growing memory)
            beyond the frozen reference rollout's coverage -- the caller
            reports "metric indeterminate" verbatim, no epsilon
            substituted (PROTOCOL section 10.3).
        ValueError: same conditions as
            ``augmented_jacobian_excluding_identity``.
    """
    with torch.no_grad():
        _, m_next = step_fn(h, m, u)
    result = augmented_jacobian_excluding_identity(step_fn, h, m, u, identity_atol=identity_atol)

    s_in_h = as_float64_array(gauge.h_scale()).reshape(-1)
    s_in_m = as_float64_array(gauge.m_scale(m)).reshape(-1) if m is not None else np.empty(0)
    s_in = np.concatenate([s_in_h, s_in_m])

    s_out_h = as_float64_array(gauge.h_scale()).reshape(-1)
    s_out_m = as_float64_array(gauge.m_scale(m_next)).reshape(-1) if m_next is not None else np.empty(0)
    s_out_full = np.concatenate([s_out_h, s_out_m])
    s_out = np.delete(s_out_full, result.excluded_row_indices)

    M = normalize_jacobian(result.jacobian, s_in, s_out)
    return operator_norm(M)


def jacobian_products(jacobians: Iterable[np.ndarray | Tensor]) -> tuple[np.ndarray, np.ndarray]:
    """QR-stabilized left product ``Phi_T = J_T ... J_1`` of a Jacobian chain.

    Never forms the raw product (REGLES section 5: guaranteed
    overflow/underflow past ~20 steps). Instead propagates an orthonormal
    tangent basis ``Q_0 = I`` and, at each step ``t``, first scales by
    ``m_t = max|J_t|`` then factors
    ``Z = (J_t / m_t) @ Q_{t-1} = Q_t R_t`` via ``np.linalg.qr``. The sign of
    each column of ``Q_t`` (and the paired row of ``R_t``) is canonicalized so
    that ``diag(R_t) >= 0`` -- QR is only unique up to such sign flips, and
    without this the returned ``Q_final`` would not be deterministic across
    otherwise-identical calls.

    Args:
        jacobians: sequence/iterable of square ``(d, d)`` matrices (numpy
            or torch, in application order ``J_1, ..., J_T``), converted
            via ``as_float64_array``.

    Returns:
        ``(Q_final, log_diag_r)``: ``Q_final`` is ``(d, d)`` orthonormal;
        ``log_diag_r`` is ``(T, d)`` with
        ``log_diag_r[t] = log(diag(R_t)) + log(m_t)``.

    Raises:
        ValueError: if ``jacobians`` is empty, any matrix is not square,
            matrices have inconsistent size, any matrix contains NaN/inf,
            or any step is numerically rank-deficient. After max-scaling, a
            step is rejected when ``sigma_min < d * eps * sigma_max`` (the
            ``numpy.matrix_rank`` convention); QR diagonals are not
            rank-revealing. This also rejects full-rank dynamics beyond ``1 / eps`` (for example,
            ``diag(1e200, 1e-200)`` underflows to zero after scaling): float64
            cannot represent that step's tangent geometry. This is a
            documented instrument limit, not a bug.
    """
    arrays = [as_float64_array(J) for J in jacobians]
    if len(arrays) == 0:
        raise ValueError("jacobians must be a non-empty sequence")

    d = arrays[0].shape
    if len(d) != 2 or d[0] != d[1]:
        raise ValueError(f"each Jacobian must be square (d, d), got shape {d}")
    d0 = d[0]
    for i, J in enumerate(arrays):
        if J.shape != (d0, d0):
            raise ValueError(
                f"jacobians[{i}] has shape {J.shape}, inconsistent with {(d0, d0)}"
            )
        if not np.isfinite(J).all():
            raise ValueError(
                f"jacobians[{i}] contains non-finite values; a NaN/inf Jacobian "
                "is a bug to chase upstream (diverged rollout, bad step_fn), "
                "never an audit input"
            )

    T = len(arrays)
    Q = np.eye(d0, dtype=np.float64)
    log_diag_r = np.empty((T, d0), dtype=np.float64)
    for t, J in enumerate(arrays):
        m_t = np.abs(J).max()
        if m_t == 0.0:
            raise ValueError("J_t is exactly zero; singular steps are not supported")
        s = np.linalg.svd(J / m_t, compute_uv=False)
        if s[-1] < d0 * np.finfo(np.float64).eps * s[0]:
            raise ValueError(
                "numerically rank-deficient at float64 (policy: sigma_min < "
                "d*eps*sigma_max of the max-scaled step, numpy matrix_rank "
                "convention; QR diagonals are not rank-revealing)"
            )
        Z = (J / m_t) @ Q
        Qt, Rt = np.linalg.qr(Z)
        diag_r = np.diag(Rt)
        diag = diag_r
        signs = np.where(diag < 0.0, -1.0, 1.0)
        Q = Qt * signs  # flip columns of Q ...
        Rt = Rt * signs[:, None]  # ... and the paired rows of R (product unchanged: s^2 = 1)
        diag_r = np.diag(Rt)
        if (diag_r == 0.0).any():
            raise ValueError(
                "J has zero singular value; rank-deficient chains require SVD, "
                "not supported in Wave 4a"
            )
        log_diag_r[t] = np.log(diag_r) + np.log(m_t)

    return Q, log_diag_r


def fr_lyapunov_exponents(
    step_fn: Callable[[Tensor, Tensor], Tensor],
    q0: Tensor,
    u_seq: Tensor,
    *,
    n_exp: int | None = None,
    discard: int = 0,
) -> np.ndarray:
    """Discrete-time Benettin free-running Lyapunov exponents.

    Reference: Benettin, Galgani, Giorgilli & Strelcyn (1980), "Lyapunov
    Characteristic Exponents for smooth dynamical systems and for
    hamiltonian systems; a method for computing all of them. Part 1:
    Theory", Meccanica 15(1), 9-20, https://doi.org/10.1007/BF02128236.

    Rolls ``step_fn`` forward from ``q0`` under ``u_seq`` (free-running:
    the rollout never sees ground truth), collects the total Jacobian
    ``J_t`` at every visited state via ``total_jacobian``, and propagates
    the tangent basis through the *same* QR routine as ``jacobian_products``
    -- one code path for both, since ``T x d^2`` stays small in audit
    settings. Exponents are the mean stabilized QR log factors
    ``log_diag_r[t]``, sorted descending.

    Units: **per-step** (discrete-time). If the underlying system has a
    physical timestep ``dt``, divide the returned exponents by ``dt`` to
    get continuous-time rates -- this function does not do that for you.

    Args:
        step_fn: pure function ``(q, u) -> q_next``, as in ``total_jacobian``.
        q0: initial augmented state, ``(d_q,)``, CPU float64.
        u_seq: control sequence, ``(T, d_u)``, CPU float64.
        n_exp: number of exponents to return. ``None`` returns the full
            spectrum (``d_q``). Values ``< d_q`` would require propagating
            a rectangular tangent basis (a genuinely different QR
            recursion); rather than silently return a wrong subset, this
            raises ``NotImplementedError`` until that path exists.
        discard: number of leading steps to drop as transient before
            averaging. For a time-invariant ``step_fn``, the stabilized QR
            log factors converge geometrically to the true exponents, so
            ``discard`` burns off the transient from the arbitrary initial
            tangent basis ``Q_0 = I``.

    Returns:
        ``(n_exp,)`` float64 array, sorted descending.

    Raises:
        ValueError: bad shapes/dtypes, or ``discard`` not in ``[0, T)``, or
            ``n_exp`` outside ``(0, d_q]``.
        NotImplementedError: ``0 < n_exp < d_q`` (see above).
    """
    require_cpu_float64_tensor(q0, "q0")
    require_cpu_float64_tensor(u_seq, "u_seq")
    if not torch.isfinite(q0).all():
        raise ValueError("q0 contains non-finite values; Lyapunov exponents require a valid initial state")
    if not torch.isfinite(u_seq).all():
        raise ValueError("u_seq contains non-finite values; Lyapunov exponents require valid controls")
    if q0.dim() != 1:
        raise ValueError(f"q0 must be 1-D (d_q,), got shape {tuple(q0.shape)}")
    if u_seq.dim() != 2:
        raise ValueError(f"u_seq must be (T, d_u), got shape {tuple(u_seq.shape)}")

    d_q = q0.shape[0]
    T = u_seq.shape[0]
    if not (0 <= discard < T):
        raise ValueError(f"discard must satisfy 0 <= discard < T={T}, got {discard}")

    if n_exp is None:
        n_exp = d_q
    elif n_exp <= 0 or n_exp > d_q:
        raise ValueError(f"n_exp must be in (0, d_q={d_q}], got {n_exp}")
    elif n_exp < d_q:
        raise NotImplementedError(
            "n_exp < d_q requires propagating a rectangular tangent basis "
            "(a different QR recursion than the full-spectrum one); not "
            "implemented -- pass n_exp=None (or n_exp=d_q) for the full "
            "spectrum instead of risking a silently wrong subset"
        )

    q = q0
    jacobians = []
    for t in range(T):
        u_t = u_seq[t]
        jacobians.append(total_jacobian(step_fn, q, u_t))
        with torch.no_grad():
            q = step_fn(q, u_t)
        if not torch.isfinite(q).all():
            raise ValueError(
                f"rollout diverged to non-finite state at step {t}; Lyapunov exponents "
                "of an invalid rollout are meaningless"
            )

    _, log_diag_r = jacobian_products(jacobians)
    exponents = log_diag_r[discard:].mean(axis=0)
    return np.sort(exponents)[::-1]


def conditional_jacobian(J_q: np.ndarray, d_h: int) -> np.ndarray:
    """The ``(1, 1)`` block of the augmented Jacobian: ``J_q[:d_h, :d_h]``.

    Doctrine (REGLES section 5): this conditional Jacobian -- the response
    of ``h`` to ``h`` with the auxiliary state ``m`` held frozen -- never concludes alone.
    It ignores every coupling that runs through ``m``
    (selective-SSM hidden state, transformer KV cache, scheduling); a
    contracting conditional block next to an expanding total Jacobian is
    exactly the failure mode this audit exists to catch. Only the PAIR
    ``(total_jacobian, conditional_jacobian)`` read together is a valid
    audit statement -- never report this block in isolation.

    Args:
        J_q: total augmented Jacobian, ``(d_q, d_q)``.
        d_h: dimension of the ``h`` block (``0 < d_h <= d_q``).

    Returns:
        ``(d_h, d_h)`` float64 array, a copy of the leading block.

    Raises:
        ValueError: ``J_q`` not square, or ``d_h`` out of range.
    """
    J_q = as_float64_array(J_q)  # audit outputs are float64, whatever came in
    if not np.isfinite(J_q).all():
        raise ValueError("J_q contains non-finite values")
    if J_q.ndim != 2 or J_q.shape[0] != J_q.shape[1]:
        raise ValueError(f"J_q must be square (d_q, d_q), got shape {J_q.shape}")
    d_q = J_q.shape[0]
    if not (0 < d_h <= d_q):
        raise ValueError(f"d_h must satisfy 0 < d_h <= d_q={d_q}, got {d_h}")
    return J_q[:d_h, :d_h].copy()


def lipschitz_pi(
    pi_fn: Callable[[Tensor], Tensor], h_samples: Tensor, *, quantile: float | None = None
) -> float:
    """Empirical Lipschitz estimate of the read-out ``pi`` over sampled states.

    For each row of ``h_samples``, computes the dense Jacobian of ``pi_fn``
    via ``torch.func.jacrev`` and its spectral norm (largest singular value,
    via ``np.linalg.svd``). Returns the max over the SUPPLIED samples, or a
    quantile of them. Neither is a certified global Lipschitz bound: a
    finite sample set cannot see a derivative spike between samples, so the
    max is an empirical LOWER estimate of the true constant (worst case
    observed, not worst case possible), and the quantile is a robust summary
    of the same finite sample. Honest reporting says "over N rollout
    states", never "the Lipschitz constant".

    Args:
        pi_fn: read-out map, called row-by-row as ``pi_fn(h_samples[i])``.
        h_samples: ``(N, d_h)`` states sampled along rollouts, CPU float64.
        quantile: if given, return this quantile (in ``(0, 1]``) of the
            per-sample spectral norms instead of the max.

    Returns:
        Scalar float: max (or quantile) spectral norm over ``h_samples``.

    Raises:
        ValueError: ``h_samples`` not ``(N, d_h)`` CPU float64 with
            ``N >= 1``, or ``quantile`` outside ``(0, 1]``.
    """
    require_cpu_float64_tensor(h_samples, "h_samples")
    if not torch.isfinite(h_samples).all():
        raise ValueError("h_samples contains non-finite values")
    if h_samples.dim() != 2:
        raise ValueError(f"h_samples must be (N, d_h), got shape {tuple(h_samples.shape)}")
    N = h_samples.shape[0]
    if N < 1:
        raise ValueError(f"h_samples must have N >= 1, got N={N}")
    if quantile is not None and not (0.0 < quantile <= 1.0):
        raise ValueError(f"quantile must be in (0, 1], got {quantile}")

    jac_fn = torch.func.jacrev(pi_fn)
    norms = np.empty(N, dtype=np.float64)
    for i in range(N):
        out = pi_fn(h_samples[i])
        require_cpu_float64_tensor(out, "pi_fn output")
        if not torch.isfinite(out).all():
            raise ValueError("pi_fn output contains non-finite values")
        J = as_float64_array(jac_fn(h_samples[i]))
        if not np.isfinite(J).all():
            raise ValueError("pi_fn Jacobian contains non-finite values; derivative of invalid function")
        norms[i] = np.linalg.svd(J, compute_uv=False)[0]
    if not np.isfinite(norms).all():
        raise ValueError("pi_fn Jacobian spectral norms contain non-finite values")

    if quantile is None:
        return float(norms.max())
    return float(np.quantile(norms, quantile))


def fr_drift_curve(pred: np.ndarray | Tensor, target: np.ndarray | Tensor) -> np.ndarray:
    """Normalized free-running drift vs. horizon (future Fig. 1 metric).

    ``err_t = mean_B ||pred_t - target_t||_2``, normalized by
    ``scale = sqrt(mean_{T,B} ||target||^2)`` so the curve is
    scale-invariant and reads as "fraction of typical target magnitude".
    Both quantities are computed after pre-dividing by ``max|target|``, so
    the scale invariance holds NUMERICALLY too: equivalent rescalings of
    ``(pred, target)`` produce the same curve, with no silent overflow of
    the squares at extreme magnitudes (e.g. ``|target| ~ 1e155``) and no
    underflow-driven false "identically zero" at tiny ones. A remaining
    overflow is only possible when ``pred`` and ``target`` live at
    astronomically different scales -- and then it fails loudly below.

    Args:
        pred: ``(T, B, d)``, numpy or torch (converted via
            ``as_float64_array``).
        target: ``(T, B, d)``, same shape as ``pred``.

    Returns:
        ``(T,)`` float64 array.

    Raises:
        ValueError: shapes differ, not rank-3, empty, non-finite inputs,
            ``target`` identically zero (the normalization is undefined),
            or float64 overflow in the computation -- a NaN/inf is a bug to
            chase, never an audit result (REGLES section 5).
    """
    pred_a = as_float64_array(pred)
    target_a = as_float64_array(target)
    if pred_a.shape != target_a.shape:
        raise ValueError(
            f"pred and target must have the same shape, got {pred_a.shape} vs {target_a.shape}"
        )
    if pred_a.ndim != 3:
        raise ValueError(f"pred/target must be (T, B, d), got ndim={pred_a.ndim}")
    if pred_a.size == 0:
        raise ValueError(f"pred/target must be non-empty, got shape {pred_a.shape}")
    if not (np.isfinite(pred_a).all() and np.isfinite(target_a).all()):
        raise ValueError(
            "pred/target contain non-finite values; NaN/inf is a bug to "
            "chase upstream, never an audit result"
        )

    # Divide each term by max|target| before subtracting so the subtraction
    # itself stays finite at extreme but equivalent physical scales.
    m = float(np.abs(target_a).max())
    if m == 0.0:
        raise ValueError("target is identically zero; normalized drift is undefined (scale == 0)")
    target_n = target_a / m
    diff_n = pred_a / m - target_a / m

    err_t = np.sqrt((diff_n**2).sum(axis=-1)).mean(axis=1)
    scale = np.sqrt((target_n**2).sum(axis=-1).mean())
    curve = err_t / scale
    if not np.isfinite(curve).all():
        raise ValueError(
            "drift curve overflowed float64: pred and target live at "
            "astronomically different scales; rescale before auditing"
        )
    return curve
