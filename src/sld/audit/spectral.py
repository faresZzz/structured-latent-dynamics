"""Display-only spectral objects for the audit: EDMD baseline, pseudospectra,
and state-dependent "local spectra" fields.

REGLES doctrine (sections 3, 5) enforced here:

- A displayed spectrum is NEVER a stability object. Every quantitative
  stability claim is built elsewhere, on the total Jacobian of the full
  augmented state (Wave 4); this module only computes what gets plotted.
- For a state-dependent transition (the ``selective`` regime of
  ``koopman_ssm.py`` -- an LPV ablation, NOT a Koopman operator, REGLES
  section 3 -- or a generic family with no structured operator at all)
  there is no single fixed operator, so this module never says "the
  spectrum" for those cases -- ``local_spectra_field`` returns a FIELD of
  local spectra, one per probed ``(h, u, state)`` point.
- "Koopman" is reserved for the fixed-operator regimes (i)/(ii) of
  ``koopman_ssm.py`` ("constant"/"bilinear"); ``edmd_fit`` is the
  linear-dictionary EDMD baseline appropriate to exactly those regimes
  (Williams, Kevrekidis & Rowley 2015, https://arxiv.org/abs/1408.4408).
- The audit never special-cases a family: every entry point here goes
  through the ``Predictor`` protocol (``step``, ``local_operator``,
  ``PredictorState``) from ``predictors/base.py`` -- no ``isinstance`` on
  a family anywhere in this file.
- Pseudospectrum grids run a dense SVD per grid point: a HARD ``n <= 64``
  gate (REGLES section 5) -- this is display-only tooling, not meant to
  scale to large operators.
- Enforcement is two-tier (see ``_common``): MODEL-facing entry points
  (``operator_spectrum``, ``local_spectra_field``) validate CPU float64
  loudly -- including the operator returned by ``local_operator`` itself --
  while the pure array utilities (``edmd_fit``, ``operator_norm``) widen
  real inputs to float64 and refuse complex ones. The audit runs offline on
  checkpoints, never on MPS, never in float32 compute (REGLES section 5).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn
from torch.func import jacrev

from sld.audit._common import (
    as_float64_array,
    require_cpu_float64_module,
    require_cpu_float64_tensor,
)
from sld.predictors.base import PredictorState


def _sorted_eigvals(A: np.ndarray) -> np.ndarray:
    """Eigenvalues of ``A``, sorted deterministically (imag, then real)."""
    ev = np.linalg.eigvals(A).astype(np.complex128)
    if not np.isfinite(ev).all():
        raise ValueError("eigenvalues overflow float64; rescale the operator")
    order = np.lexsort((ev.real, ev.imag))
    return ev[order]


def edmd_fit(X: np.ndarray, Y: np.ndarray, *, cond_max: float = 1e8) -> np.ndarray:
    """Least-squares linear-dictionary EDMD fit: ``x_{k+1} ~= A x_k``.

    ``X``, ``Y``: ``(N, d)`` snapshot pairs, row ``k`` holding ``(x_k,
    x_{k+1})``. Solves ``min_A ||X @ A.T - Y||_F`` via
    ``np.linalg.lstsq(X, Y, rcond=None)`` and transposes back to the
    convention ``x_{k+1} ~= A @ x_k``. This is the plain least-squares
    dictionary estimator (the EDMD baseline for the fixed-operator Koopman
    regimes (i)/(ii) -- REGLES section 3 vocabulary).

    CONTRACT -- autonomous / fixed-input data only: this signature has no
    input channel, so it fits the propagator of the dynamics AS SAMPLED. On
    data gathered under a varying control ``u_t`` the result is a confounded
    average of ``A + (input terms)`` and its spectrum is not the drift
    spectrum of regime (i)/(ii). Callers must restrict the snapshot pairs to
    ``u = 0`` (or one fixed ``u``) segments; a controlled/bilinear dictionary
    is a different estimator and deliberately not this function.

    Guards against a well-known gauge trap (REGLES section 7 gold-test
    doctrine; Wave 3 spectral-identifiability lesson): a one-step fit can hit
    near-zero residual on the training pairs while still not pinning the true
    dynamics if ``X`` is (numerically) rank-deficient, or more generally if
    only checked one step ahead. ``np.linalg.lstsq`` silently returns the
    MIN-NORM solution on unresolved directions -- i.e. it regularizes
    without telling you -- so near-singularity must be refused loudly: the
    fit is rejected when ``cond(X) > cond_max`` (or the effective rank is
    below ``d``). Any spectral claim built on the returned ``A`` MUST also
    be validated by a multi-step free rollout against held-out data, never
    by the one-step residual alone.

    Raises:
        ValueError: if ``X``/``Y`` shapes disagree, contain non-finite
            values, if ``N < d`` snapshot pairs, or if ``X`` is numerically
            non-identifiable (effective rank < ``d`` or condition number
            above ``cond_max``).
    """
    X = as_float64_array(X)
    Y = as_float64_array(Y)
    if X.ndim != 2 or Y.shape != X.shape:
        raise ValueError(
            f"X, Y must both be (N, d) with matching shape, got {X.shape} vs {Y.shape}"
        )
    if not (np.isfinite(X).all() and np.isfinite(Y).all()):
        raise ValueError("X/Y contain non-finite values; NaN/inf is a bug, not data")
    if not (np.isfinite(cond_max) and cond_max >= 1.0):
        raise ValueError(
            f"cond_max must be a finite number >= 1, got {cond_max}; a NaN/inf "
            "threshold would silently disable the identifiability guard -- to "
            "consciously loosen it, pass a large FINITE value (e.g. 1e15)"
        )
    N, d = X.shape
    if N < d:
        raise ValueError(f"need N >= d snapshot pairs, got N={N}, d={d}")
    coef, _, rank, s = np.linalg.lstsq(X, Y, rcond=None)  # X @ coef ~= Y, coef == A.T
    cond = float(s[0] / s[-1]) if s[-1] > 0.0 else np.inf
    if rank < d or cond > cond_max:
        raise ValueError(
            f"X is numerically non-identifiable (effective rank {rank} < d={d} "
            f"or cond(X)={cond:.3e} > cond_max={cond_max:.1e}): lstsq would "
            "silently min-norm-regularize unresolved directions and the "
            "returned spectrum would be an artifact of that regularization"
        )
    return coef.T


def operator_norm(A: Tensor | np.ndarray) -> float:
    """Largest singular value of ``A`` (``np.ndarray`` or ``torch.Tensor``, 2-D).

    ``A`` need not be square. Used both to read off ``||A||`` and, applied
    to ``A^k`` by the caller, to trace transient growth curves -- REGLES
    section 5 forbids raw matrix powers beyond ~20 steps on models learned
    from data, but this function itself is agnostic to how ``A`` was built.
    """
    A = as_float64_array(A)
    if A.ndim != 2:
        raise ValueError(f"A must be a 2-D matrix, got shape {A.shape}")
    if not np.isfinite(A).all():
        raise ValueError("A contains non-finite values")
    s = np.linalg.svd(A, compute_uv=False)
    if not np.isfinite(s).all():
        raise ValueError("operator norm overflow float64; rescale the operator")
    return float(s[0])


def pseudospectrum_grid(A: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Epsilon-pseudospectrum proxy: ``sigma_min(A - lambda I)`` over ``grid``.

    ``grid``: complex array of arbitrary shape. Returns a real float64 array
    of the same shape, one smallest singular value per grid point (computed
    by a plain per-point ``np.linalg.svd``, no vectorization trick).

    Raises:
        ValueError: if ``A`` is not square, or if ``A.shape[0] > 64`` (SVD
            grid only for n <= 64, REGLES section 5 -- this routine is
            display-only tooling and does not scale to large operators).
    """
    # Unlike the real-valued EDMD/operator-norm inputs, a displayed operator
    # may itself be complex.  ``as_float64_array`` would discard its imaginary
    # part, so retain it explicitly while still normalizing real arrays to
    # float64 precision.
    A = np.asarray(A, dtype=np.complex128)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got shape {A.shape}")
    if not np.isfinite(A).all():
        raise ValueError("A contains non-finite values")
    n = A.shape[0]
    if n > 64:
        raise ValueError(f"SVD grid only for n <= 64, REGLES section 5; got n={n}")

    grid = np.asarray(grid, dtype=np.complex128)
    if not np.isfinite(grid).all():
        raise ValueError("grid contains non-finite values")
    eye = np.eye(n, dtype=np.complex128)
    flat = grid.reshape(-1)
    out = np.empty(flat.shape[0], dtype=np.float64)
    for i, lam in enumerate(flat):
        with np.errstate(over="ignore", invalid="ignore"):
            shifted = A - lam * eye
        if not np.isfinite(shifted).all():
            raise ValueError("pseudospectrum value overflow float64; rescale the operator or grid")
        s = np.linalg.svd(shifted, compute_uv=False)
        if not np.isfinite(s).all():
            raise ValueError("pseudospectrum value overflow float64; rescale the operator or grid")
        out[i] = s[-1]
    if not np.isfinite(out).all():
        raise ValueError("pseudospectrum value overflow float64; rescale the operator or grid")
    return out.reshape(grid.shape)


def operator_spectrum(
    predictor: nn.Module, h: Tensor, u: Tensor, *, atol: float = 1e-8
) -> np.ndarray:
    """Eigenvalues of the ONE shared local operator across ``n_probes`` probes.

    ``h``: ``(n_probes, d_h)`` with ``n_probes >= 2`` DISTINCT probe points;
    ``u``: ``(d_u,)``, a single constant input broadcast over the probe
    batch. Calls ``predictor.local_operator(h, u_batch, PredictorState())``
    and requires the returned ``A`` to be identical across every probe,
    ENTRYWISE within ``atol * (1 + |A[0]|)`` -- a per-entry relative
    comparison, so order-one variation in a small-magnitude block is never
    masked by a large-magnitude block elsewhere in the same operator.

    This is a SCREENING guard, necessary but never sufficient: a finite
    probe set cannot certify global input/state-independence. Dependence
    that is invisible at the probe slice -- through the auxiliary state
    ``m`` (empty at the fresh probing state), through inputs other than the
    supplied ``u``, or localized away from the probes -- passes undetected.
    The claim "this family's operator is fixed" belongs to the run
    configuration (regime (i)/(ii)), never to this probe; the guard only
    catches displays that are provably NOT a single spectrum (the code-level
    form of the vocabulary discipline, REGLES section 3).

    Raises:
        ValueError: if ``local_operator`` returns ``None`` (no structured
            operator; use ``local_spectra_field``), if the probe rows are
            all identical (a vacuous constancy check), or if the returned
            ``A`` varies across probes (a state-dependent operator -- this
            is a field of local spectra, not "the spectrum"; use
            ``local_spectra_field``).
    """
    if not (np.isfinite(atol) and atol >= 0.0):
        raise ValueError("atol must be finite and >= 0; NaN disables screening, inf accepts everything")
    require_cpu_float64_module(predictor)
    require_cpu_float64_tensor(h, "h")
    require_cpu_float64_tensor(u, "u")
    if not torch.isfinite(h).all():
        raise ValueError("h contains non-finite values")
    if not torch.isfinite(u).all():
        raise ValueError("u contains non-finite values")
    if h.dim() != 2 or h.shape[0] < 2:
        raise ValueError(f"h must be (n_probes, d_h) with n_probes >= 2, got {tuple(h.shape)}")
    if u.dim() != 1:
        raise ValueError(f"u must be (d_u,), a single constant input, got {tuple(u.shape)}")
    if torch.equal(h, h[0].expand_as(h)):
        raise ValueError(
            "all probe rows are identical -- the constancy check would be "
            "vacuous; supply distinct probe points"
        )

    n_probes = h.shape[0]
    u_batch = u.unsqueeze(0).expand(n_probes, -1)
    op = predictor.local_operator(h, u_batch, PredictorState())
    if op is None:
        raise ValueError("no structured operator; use local_spectra_field")
    A, _ = op
    # A parameterless predictor slips past the module validator, so the
    # returned operator itself is validated too (no float32 screening).
    require_cpu_float64_tensor(A, "local_operator A")
    if A.dim() != 3:
        raise ValueError(f"A must have shape (n_probes, d, d); got {A.shape}")
    if A.shape != (n_probes, A.shape[-1], A.shape[-1]):
        raise ValueError(
            f"A must have shape (n_probes, d, d); got {A.shape}. "
            "Broadcast of (1,d,d) renders constancy screen vacuous."
        )
    if not torch.isfinite(A).all():
        raise ValueError("A contains non-finite values (NaN/Inf); screening will fail silently")
    A0 = A[0]
    tol = atol * (1.0 + A0.abs())
    if bool(((A - A0).abs() > tol).any()):
        raise ValueError(
            "state-dependent operator — this is a field of local spectra, not "
            "the spectrum; use local_spectra_field"
        )
    return _sorted_eigvals(as_float64_array(A0))


def _field_eigvals_at(predictor: nn.Module, h: Tensor, u: Tensor, state: PredictorState) -> np.ndarray:
    """Eigenvalues at one sampled time, one row per batch element -- (B, d_r)."""
    batch = h.shape[0]
    op = predictor.local_operator(h, u, state)
    if op is not None:
        A, _ = op
        require_cpu_float64_tensor(A, "local_operator A")
        if A.dim() != 3:
            raise ValueError(f"A must have shape (batch, d_r, d_r); got {A.shape}")
        if A.shape[0] != batch or A.shape[-2] != A.shape[-1]:
            raise ValueError(f"A must have shape (batch, d_r, d_r); got {A.shape}")
        if not torch.isfinite(A).all():
            raise ValueError("A contains non-finite values (NaN/Inf); screening will fail silently")
        A_np = as_float64_array(A)
        return np.stack([_sorted_eigvals(A_np[b]) for b in range(batch)], axis=0)

    # No structured operator: conditional Jacobian d h_{t+1} / d h_t at a
    # frozen context, by autograd.  The auxiliary state is detached WHOLE:
    # the audit cannot know which axis of ``state.m`` is the batch axis
    # (that is family-specific layout knowledge -- e.g. the transformer KV
    # cache carries batch on axis 2), so instead of slicing ``m`` per
    # element we differentiate the full batched step and read the diagonal
    # ``(b, :, b, :)`` blocks.  Cross-element blocks are computed and
    # discarded -- an accepted cost for a family-agnostic offline instrument.
    frozen = PredictorState() if state.m is None else PredictorState(m=state.m.detach())
    u_frozen = u.detach()
    h_probe, _ = predictor.step(h.detach(), u_frozen, frozen)
    require_cpu_float64_tensor(h_probe, "step output")
    if h_probe.shape != h.shape:
        raise ValueError(
            f"step output must preserve h shape, got {tuple(h_probe.shape)} vs {tuple(h.shape)}"
        )
    if not torch.isfinite(h_probe).all():
        raise ValueError("step output contains non-finite values")

    def f(h_in: Tensor) -> Tensor:
        h_next, _ = predictor.step(h_in, u_frozen, frozen)
        return h_next

    J_full = as_float64_array(jacrev(f)(h.detach()))  # (B, d_h, B, d_h)
    if not np.isfinite(J_full).all():
        raise ValueError("J contains non-finite values; derivative of invalid function")
    rows = [_sorted_eigvals(J_full[b, :, b, :]) for b in range(batch)]
    return np.stack(rows, axis=0)


def local_spectra_field(
    predictor: nn.Module, h0: Tensor, u_seq: Tensor, *, sample_every: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Field of local spectra — display only, never a stability object.

    Threads the predictor's own state exactly like ``PredictorModule.rollout``
    (fresh ``PredictorState()`` at ``t=0``). At every sampled time ``t = 0,
    sample_every, 2*sample_every, ...`` (``t < T``), reads the local operator
    at ``(h_t, u_t, state_t)``:

    - if ``local_operator`` returns ``(A, B)``, eigenvalues of ``A`` per
      batch element;
    - if ``None`` (generic families), the conditional Jacobian
      ``d h_{t+1} / d h_t`` at a frozen context, by ``torch.func.jacrev``,
      looped over the batch.

    Args:
        h0: ``(B, d_h)`` initial state.
        u_seq: ``(T, B, d_u)`` input sequence.
        sample_every: sampling stride in steps, ``>= 1``.

    Returns:
        ``(times, eigvals)``: ``times`` is ``(n_s,)`` int64, ``eigvals`` is
        ``(n_s, B, d_r)`` complex128, each row sorted deterministically
        (imag, then real -- same convention as ``operator_spectrum`` and the
        Wave 3 gold tests).
    """
    require_cpu_float64_module(predictor)
    require_cpu_float64_tensor(h0, "h0")
    require_cpu_float64_tensor(u_seq, "u_seq")
    if not torch.isfinite(h0).all():
        raise ValueError("h0 contains non-finite values")
    if not torch.isfinite(u_seq).all():
        raise ValueError("u_seq contains non-finite values")
    if sample_every < 1:
        raise ValueError(f"sample_every must be >= 1, got {sample_every}")
    if h0.dim() != 2:
        raise ValueError(f"h0 must be (B, d_h), got {tuple(h0.shape)}")
    if u_seq.dim() != 3 or u_seq.shape[1] != h0.shape[0]:
        raise ValueError(
            f"u_seq must be (T, B, d_u) with B == h0.shape[0], "
            f"got {tuple(u_seq.shape)} vs h0 {tuple(h0.shape)}"
        )

    T = u_seq.shape[0]
    sample_times = list(range(0, T, sample_every))

    h, state = h0, PredictorState()
    fields = []
    pending = iter(sample_times)
    next_sample = next(pending, None)
    for t in range(T):
        if t == next_sample:
            fields.append(_field_eigvals_at(predictor, h, u_seq[t], state))
            next_sample = next(pending, None)
        h_shape = h.shape
        h, state = predictor.step(h, u_seq[t], state)
        require_cpu_float64_tensor(h, "step output h")
        if h.shape != h_shape:
            raise ValueError(
                f"step output h must preserve shape at step {t}, got {tuple(h.shape)} vs {tuple(h_shape)}"
            )
        if not torch.isfinite(h).all():
            raise ValueError(f"step output h contains non-finite values at step {t}")
        if isinstance(state.m, Tensor):
            require_cpu_float64_tensor(state.m, "step output state.m")
            if not torch.isfinite(state.m).all():
                raise ValueError(f"step output state.m contains non-finite values at step {t}")
        # Detach the threaded rollout at every step: nothing downstream
        # differentiates through the trajectory itself (each sampled reading
        # is local), and without this the autograd graph of the whole
        # rollout would be retained -- unbounded memory on long
        # checkpoint audits.
        h = h.detach()
        if state.m is not None:
            state = PredictorState(m=state.m.detach())

    times = np.asarray(sample_times, dtype=np.int64)
    eigvals = np.stack(fields, axis=0)
    return times, eigvals
