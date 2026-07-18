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

from typing import Callable, Iterable

import numpy as np
import torch
from torch import Tensor

from sld.audit._common import as_float64_array, require_cpu_float64_tensor


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
