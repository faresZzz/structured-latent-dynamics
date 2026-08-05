"""Analytic gold: the three structured arms ARE the exact ZOH coupled oscillators.

The strongest positive control this repo has. Under the declared conditions
(``m1 = m2 = m``, ``k1 = k2 = k``, diagonalizable, UNDERDAMPED ``Omega_pm^2 >
alpha^2``, lifted dimension ``2r >= 4``, neutral context ``W_theta = 0``), the
two-mass damped chain is EXACTLY in the model class of ``StructuredPredictor``:
there is a linear ``psi`` and its exact inverse ``pi`` for which the structured
cell reproduces the zero-order-hold discretization of the physics to machine
precision, for a hundred free-running steps. Nothing is learned here (phase-C
rule C7); the algebra either holds or it does not.

INDEPENDENCE OF THE REFERENCE (phase-C rule C3, REGLES section 7.1). Phase A
settled that a gold may not be validated against data produced by the same
closed form -- that is the object against itself. The oracle here is a THIRD
derivation, independent of both:

- of the object under test: the physical ``(A_x, B_x)`` is assembled by hand
  from ``m, k, k_c, c``, never from ``CoupledOscillators.A_mat``; its
  discretization comes from the Van Loan augmented matrix assembled below and
  evaluated by ``torch.matrix_exp`` (scaling-and-squaring on a general 8x8),
  never from ``discretize_zoh`` / ``rotation_scaling_zoh``, whose algorithm is
  the closed form in the ``(I, J)`` coefficient basis;
- of the reference phase A already used: no RK4 stability polynomial appears
  anywhere. The gold is exact-ZOH end to end, never RK4 data.

DERIVATION (this is what the test encodes; the plan gives the free part, the
input coupling is derived here). With ``m1 = m2 = m`` and ``k1 = k2 = k``, a
force ``u`` on mass 1 only:

    m q1'' = -k q1 - k_c (q1 - q2) - c q1' + u
    m q2'' = -k q2 - k_c (q2 - q1) - c q2'

Sum and difference in ``q_pm = (q1 +/- q2)/sqrt(2)`` decouple the chain (the
coupling terms cancel in the sum and double in the difference), and BOTH modes
inherit the same input, halved by the normalization:

    q_pm'' + (c/m) q_pm' + Omega_pm^2 q_pm = u / (m sqrt(2))
    Omega_+^2 = k/m,  Omega_-^2 = (k + 2 k_c)/m,  alpha = -c/(2m)

With ``omega_pm = sqrt(Omega_pm^2 - alpha^2)`` (REAL exactly under the
underdamping hypothesis) and the modal lift ``h_{pm,1} = q_pm``,
``h_{pm,2} = (alpha q_pm - q_pm') / omega_pm``, write ``f = u / (m sqrt(2))``:

    h_{pm,1}' = q_pm' = alpha h_{pm,1} - omega_pm h_{pm,2}
    h_{pm,2}' = (alpha q_pm' - q_pm'') / omega_pm
              = (-alpha q_pm' + Omega_pm^2 q_pm - f) / omega_pm      [c/m = -2 alpha]
              = omega_pm h_{pm,1} + alpha h_{pm,2} - f / omega_pm

so the input enters the SECOND component of each block only, with a negative
sign and a ``1/omega_pm`` weight:

    h_pm' = [[alpha, -omega_pm], [omega_pm, alpha]] h_pm
            + [0, -1/(sqrt(2) m omega_pm)]^T u

Block order is not free: ``block_diagonal`` puts block ``i`` on rows
``2i:2i+2``, so ``psi`` must emit ``h = (h_{+,1}, h_{+,2}, h_{-,1}, h_{-,2})``
and ``theta_0`` must be ``(alpha, alpha, omega_+, omega_-, vec B_h)``.

TOLERANCES. All float64. One step and the eigenvalues are two matrix
exponentials and one change of basis apart from each other, i.e. a handful of
ulp amplified by cond(S) = 1.73 (measured): ``1e-10`` leaves ~6 decades of
headroom. The rollout composes that a hundred times; ``alpha < 0`` makes every
step a contraction (``|eig(A_bar_x)| = exp(Delta alpha) = 0.9933 < 1``,
measured), so the error accumulates linearly rather than exponentially, and
``1e-8`` leaves ~7 decades. Both bounds come from the phase-C plan; the
measured errors are 3.3e-16 (step), 3.9e-15 (H=100) and 4.5e-16 (eigenvalues),
identical on the three arms.

WHAT THIS GOLD DOES NOT SHOW, stated so it is not over-read. At the neutral
context the three arms produce the SAME operator by construction
(``W_theta = 0``), and indeed the three measured errors agree to the last bit.
So this file does not separate the backbones -- it proves, three times, that
each arm's facade really routes through psi, the head, the exact ZOH and pi,
with no residual path around them, and that the shared cell contains the true
system exactly. Whether a LIVE context reaches the operator without leaking
``u_t`` is wave C2's question, not this one.

VOCABULARY (REGLES section 3). ``A_bar_t`` is produced by a context head, so
the transition is LPV / cocycle even here, where the context happens to be
neutral. Nothing in this file is a Koopman operator, and ``A_bar h + B_bar u``
is AFFINE, not bilinear.
"""

from __future__ import annotations

import itertools
import math

import pytest
import torch
from torch import Tensor

from sld.predictors.base import PredictorState
from sld.predictors.structured import (
    MLPKPredictor,
    SSMKPredictor,
    StructuredPredictor,
    TransformerKPredictor,
)
from sld.systems.coupled_oscillators import CoupledOscillators

DTYPE = torch.float64

# Declared physical point. m1 = m2, k1 = k2 (the plan's symmetric conditions);
# c is small enough that BOTH modes are underdamped with a wide margin
# (checked, not assumed, by test_declared_regime_is_underdamped).
M, K, K_C, C = 1.3, 2.7, 0.55, 0.35
DELTA = 0.05  # ZOH step
R, D_Z, D_U = 2, 4, 1
# d_u = 1 is the physics, not a convenience: the force acts on mass 1 only, so
# B_x has exactly one column. Inventing a second channel would test an input
# map the reference system does not have.
WIDTH = 8  # divisible by the transformer's n_heads=2; irrelevant since W_theta=0
HORIZON = 100

TOL_STEP = 1e-10
TOL_ROLLOUT = 1e-8
TOL_EIG = 1e-10

ARMS = [MLPKPredictor, TransformerKPredictor, SSMKPredictor]
ARM_IDS = ["mlp_k", "transformer_k", "ssm_k"]


# ---------------------------------------------------------------------------
# the physics, by hand -- NOT CoupledOscillators
# ---------------------------------------------------------------------------
def _physical(m: float, k: float, k_c: float, c: float) -> tuple[Tensor, Tensor]:
    """``(A_x, B_x)`` on ``z = (q1, q2, q1', q2')``, assembled from the EoM."""
    A_x = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-(k + k_c) / m, k_c / m, -c / m, 0.0],
            [k_c / m, -(k + k_c) / m, 0.0, -c / m],
        ],
        dtype=DTYPE,
    )
    B_x = torch.tensor([[0.0], [0.0], [1.0 / m], [0.0]], dtype=DTYPE)
    return A_x, B_x


def _van_loan(A: Tensor, B: Tensor, delta: float) -> tuple[Tensor, Tensor]:
    """Exact ZOH by the Van Loan augmented matrix (1978).

    ``expm([[delta A, delta I], [0, 0]])`` has top-left block ``exp(delta A)``
    and top-right block ``Phi = int_0^delta exp(sA) ds``; hence
    ``B_bar = Phi B``. ``torch.matrix_exp`` is scaling-and-squaring on the full
    ``2n x 2n`` matrix -- a different algorithm from the ``(I, J)`` closed form
    under test, and it never touches ``src/``.
    """
    n = A.shape[-1]
    aug = torch.zeros(2 * n, 2 * n, dtype=A.dtype)
    aug[:n, :n] = delta * A
    aug[:n, n:] = delta * torch.eye(n, dtype=A.dtype)
    expm = torch.matrix_exp(aug)
    return expm[:n, :n], expm[:n, n:] @ B


def _modal(m: float, k: float, k_c: float, c: float) -> tuple[float, float, float]:
    """``(alpha, omega_+, omega_-)``; REFUSES a non-underdamped regime.

    ``omega_pm = sqrt(Omega_pm^2 - alpha^2)`` is real only while
    ``Omega_pm^2 > alpha^2``. Outside that, the mode has two DISTINCT REAL
    eigenvalues, which no rotation-scaling block can hold (its pair
    ``alpha +/- i omega`` is real only at ``omega = 0``, where the two
    eigenvalues coincide). Signalling here is the point: the caller must not be
    able to reach a complex or silently-clamped ``omega``.
    """
    alpha = -c / (2.0 * m)
    out = [alpha]
    for omega_sq in (k / m, (k + 2.0 * k_c) / m):
        if omega_sq <= alpha**2:
            raise ValueError(
                f"not underdamped: Omega^2={omega_sq!r} <= alpha^2={alpha**2!r}; "
                "the modal lift needs a real omega"
            )
        out.append(math.sqrt(omega_sq - alpha**2))
    return out[0], out[1], out[2]


def _modal_maps(
    alpha: float, w_p: float, w_m: float, m: float = M
) -> tuple[Tensor, Tensor, Tensor]:
    """``(S, S_inv, B_h)``: the change of basis, its hand-built inverse, ``B`` in H.

    ``S`` reads off ``q_pm = (q1 +/- q2)/sqrt(2)`` and
    ``h_{pm,2} = (alpha q_pm - q_pm')/omega_pm``; ``S_inv`` reads off the
    inverted relations ``q1 = (q_+ + q_-)/sqrt(2)`` and
    ``q_pm' = alpha h_{pm,1} - omega_pm h_{pm,2}``. Both are written out, so
    ``S_inv @ S == I`` is a real check of the derivation and not of
    ``torch.linalg.inv``.

    ``m`` is a genuine parameter (defaults to the module's ``M`` only for the
    call sites that never vary it): ``B_h`` depends on mass through
    ``1/(sqrt(2) m omega_pm)``, and reading the global ``M`` here instead
    would silently mismatch any future call built at a different mass.
    """
    s2 = math.sqrt(2.0)
    S = torch.tensor(
        [
            [1 / s2, 1 / s2, 0.0, 0.0],
            [alpha / (s2 * w_p), alpha / (s2 * w_p), -1 / (s2 * w_p), -1 / (s2 * w_p)],
            [1 / s2, -1 / s2, 0.0, 0.0],
            [alpha / (s2 * w_m), -alpha / (s2 * w_m), -1 / (s2 * w_m), 1 / (s2 * w_m)],
        ],
        dtype=DTYPE,
    )
    S_inv = torch.tensor(
        [
            [1 / s2, 0.0, 1 / s2, 0.0],
            [1 / s2, 0.0, -1 / s2, 0.0],
            [alpha / s2, -w_p / s2, alpha / s2, -w_m / s2],
            [alpha / s2, -w_p / s2, -alpha / s2, w_m / s2],
        ],
        dtype=DTYPE,
    )
    B_h = torch.tensor(
        [[0.0], [-1 / (s2 * m * w_p)], [0.0], [-1 / (s2 * m * w_m)]], dtype=DTYPE
    )
    return S, S_inv, B_h


def _theta_0(alpha: float, w_p: float, w_m: float, B_h: Tensor) -> Tensor:
    """Head bias in the frozen layout ``[alpha (r), omega (r), vec B (2r*d_u)]``."""
    return torch.cat(
        [torch.tensor([alpha, alpha, w_p, w_m], dtype=DTYPE), B_h.reshape(-1)]
    )


def _build(factory, r: int = R, delta: float = DELTA) -> StructuredPredictor:
    return factory(
        D_Z, D_U, delta, r=r, width=WIDTH, psi_hidden=(), pi_hidden=(), dtype=DTYPE
    )


def _configure(model: StructuredPredictor, S: Tensor, S_inv: Tensor, theta_0: Tensor):
    """Plant the exact analytic solution: ``psi = S``, ``pi = S^-1``, bias = theta_0.

    ``W_theta`` is NOT touched: PROTOCOL 13.6 makes it exactly zero at
    construction, so the neutral context is the model as built.
    """
    with torch.no_grad():
        model.psi[0].weight.copy_(S)
        model.psi[0].bias.zero_()
        model.pi[0].weight.copy_(S_inv)
        model.pi[0].bias.zero_()
        model.head.bias.copy_(theta_0)
    return model


def _gold_model(factory) -> StructuredPredictor:
    alpha, w_p, w_m = _modal(M, K, K_C, C)
    S, S_inv, B_h = _modal_maps(alpha, w_p, w_m, M)
    return _configure(_build(factory), S, S_inv, _theta_0(alpha, w_p, w_m, B_h))


def _reference_rollout(z0: Tensor, u_seq: Tensor) -> Tensor:
    """``x_1 .. x_T`` of the exact-ZOH physics, ``(T, B, 4)``."""
    A_bar, B_bar = _van_loan(*_physical(M, K, K_C, C), DELTA)
    x, outs = z0, []
    for t in range(u_seq.shape[0]):
        x = x @ A_bar.T + u_seq[t] @ B_bar.T
        outs.append(x)
    return torch.stack(outs, dim=0)


def _inputs(batch: int, horizon: int, seed: int = 0) -> tuple[Tensor, Tensor]:
    g = torch.Generator().manual_seed(seed)
    z0 = torch.rand(batch, D_Z, generator=g, dtype=DTYPE) * 2.0 - 1.0
    u_seq = torch.rand(horizon, batch, D_U, generator=g, dtype=DTYPE) * 2.0 - 1.0
    return z0, u_seq


def _eig_max_error(a: Tensor, b: Tensor) -> float:
    """Max error under the BEST pairing of two eigenvalue multisets.

    Brute force over the 4! assignments rather than ``sort_complex``: the two
    conjugate pairs share the same real part ``alpha`` to the last bit here, so
    a real-part-first ordering flips on rounding noise
    (``tests/systems/test_coupled_oscillators.py:63``).

    ponytail: brute force over ``n!`` assignments -- exact, and 24 of them at
    ``n = 2r = 4``. Ceiling: it stops being free past ``r ~ 4`` (8! = 40320);
    swap in a Hungarian assignment then, not before.
    """
    xs, ys = a.tolist(), b.tolist()
    return min(
        max(abs(x - ys[p[i]]) for i, x in enumerate(xs))
        for p in itertools.permutations(range(len(ys)))
    )


# ---------------------------------------------------------------------------
# the montage itself, checked before it is used
# ---------------------------------------------------------------------------
def test_hand_built_system_is_the_repo_system():
    """Montage control, deliberately separate: "my hand system IS that system".

    NOT circular. ``A_x`` is the SYSTEM; the object under test is the
    representation the predictor builds of it after discretization, and no part
    of the gold chain below reads ``CoupledOscillators``. This assertion only
    rules out the boring failure where the gold is exact about the wrong
    physics.
    """
    A_x, B_x = _physical(M, K, K_C, C)
    system = CoupledOscillators(m1=M, m2=M, k1=K, k_c=K_C, k2=K, c=C)
    torch.testing.assert_close(A_x, torch.as_tensor(system.A_mat), atol=1e-15, rtol=0)
    torch.testing.assert_close(B_x, torch.as_tensor(system.B_mat), atol=1e-15, rtol=0)


def test_van_loan_helper_matches_the_scalar_closed_form():
    """Positive on the ORACLE: 1x1 case, where ZOH is elementary.

    ``A = [a]``, ``B = [1]`` gives ``A_bar = exp(a delta)`` and
    ``B_bar = (exp(a delta) - 1)/a`` by direct integration. Catches a
    transposed or misplaced block in the augmented assembly -- the one way this
    oracle could be silently wrong.
    """
    a, delta = -0.7, 0.05
    A_bar, B_bar = _van_loan(
        torch.tensor([[a]], dtype=DTYPE), torch.ones(1, 1, dtype=DTYPE), delta
    )
    assert abs(A_bar.item() - math.exp(a * delta)) < 1e-15
    assert abs(B_bar.item() - (math.exp(a * delta) - 1.0) / a) < 1e-15


def test_declared_regime_is_underdamped():
    """The hypothesis of the whole file, measured rather than asserted in prose."""
    alpha = -C / (2.0 * M)
    assert K / M > alpha**2
    assert (K + 2.0 * K_C) / M > alpha**2
    # and the modal frequencies are real and distinct, so the two blocks are
    # genuinely different (a gold on two identical blocks would be weaker)
    _, w_p, w_m = _modal(M, K, K_C, C)
    assert abs(w_p - w_m) > 0.1


def test_modal_change_of_basis_block_diagonalises_the_generator():
    """``S A_x S^-1`` IS ``blkdiag([[a,-w],[w,a]])`` -- the plan's free-part formula."""
    alpha, w_p, w_m = _modal(M, K, K_C, C)
    S, S_inv, _ = _modal_maps(alpha, w_p, w_m, M)
    torch.testing.assert_close(S_inv @ S, torch.eye(4, dtype=DTYPE), atol=1e-15, rtol=0)

    expected = torch.zeros(4, 4, dtype=DTYPE)
    expected[0:2, 0:2] = torch.tensor([[alpha, -w_p], [w_p, alpha]], dtype=DTYPE)
    expected[2:4, 2:4] = torch.tensor([[alpha, -w_m], [w_m, alpha]], dtype=DTYPE)
    A_x, _ = _physical(M, K, K_C, C)
    torch.testing.assert_close(S @ A_x @ S_inv, expected, atol=1e-14, rtol=0)


def test_modal_input_matches_the_hand_derivation():
    """``B_h`` derived from the EoM equals ``S B_x`` -- the input coupling, checked.

    The plan gives the FREE modal form only; ``B_h = [0, -1/(sqrt(2) m
    omega_pm)]^T`` per block is derived in the module docstring, and this is
    the check that the derivation is right.
    """
    alpha, w_p, w_m = _modal(M, K, K_C, C)
    S, _, B_h = _modal_maps(alpha, w_p, w_m, M)
    _, B_x = _physical(M, K, K_C, C)
    torch.testing.assert_close(S @ B_x, B_h, atol=1e-15, rtol=0)


def test_modal_input_tracks_the_mass_argument_not_the_module_global():
    """Closes the trap: ``_modal_maps`` used to read the module's ``M`` for
    ``B_h`` regardless of the ``m`` its caller cared about. No existing test
    varies mass, so a future negative built at ``m != M`` would have gotten a
    silently wrong ``B_h`` and passed. Build at ``m_other = 2.1 != M`` and
    check ``B_h`` is ``S @ B_x`` computed AT THAT MASS, not at ``M``.
    """
    m_other = 2.1
    assert m_other != M
    alpha, w_p, w_m = _modal(m_other, K, K_C, C)
    S, _, B_h = _modal_maps(alpha, w_p, w_m, m_other)
    _, B_x_other = _physical(m_other, K, K_C, C)
    torch.testing.assert_close(S @ B_x_other, B_h, atol=1e-15, rtol=0)
    # and it must NOT equal the B_h you'd get by (wrongly) using the global M
    _, B_x_M = _physical(M, K, K_C, C)
    assert (S @ B_x_M - B_h).abs().max().item() > 1e-3


@pytest.mark.parametrize("factory", ARMS, ids=ARM_IDS)
def test_head_weight_is_exactly_zero_at_construction(factory):
    """PROTOCOL 13.6: the neutral context is the model AS BUILT, not patched.

    If this ever fails, every positive below is about a hand-rigged variant
    instead of about the three arms, and the gold must be re-read.
    """
    assert bool((_build(factory).head.weight == 0).all())


# ---------------------------------------------------------------------------
# the gold, three arms
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("factory", ARMS, ids=ARM_IDS)
def test_one_step_matches_exact_zoh(factory):
    """One number through psi, the head, A_bar, B_bar, u and pi at once."""
    model = _gold_model(factory)
    z0, u_seq = _inputs(batch=3, horizon=1)
    h_next, _ = model.step(model.lift(z0), u_seq[0], PredictorState())
    err = (model.project(h_next) - _reference_rollout(z0, u_seq)[0]).abs().max().item()
    assert err < TOL_STEP, f"one-step error {err:.3e}"


@pytest.mark.parametrize("factory", ARMS, ids=ARM_IDS)
def test_free_rollout_matches_exact_zoh(factory):
    """``H = 100`` free-running: 100 compositions of the structured cell."""
    model = _gold_model(factory)
    z0, u_seq = _inputs(batch=3, horizon=HORIZON)
    z_hat = model.project(model.rollout(model.lift(z0), u_seq))
    err = (z_hat - _reference_rollout(z0, u_seq)).abs().max().item()
    assert err < TOL_ROLLOUT, f"rollout error {err:.3e}"


@pytest.mark.parametrize("factory", ARMS, ids=ARM_IDS)
def test_modal_eigenvalues_match_exact_zoh(factory):
    """Spectrum of the cell's ``A_bar`` vs the spectrum of the Van Loan ``A_bar_x``.

    Similar matrices (``A_bar_h = S A_bar_x S^-1``), so the multisets coincide
    exactly; paired by proximity, never sorted.
    """
    model = _gold_model(factory)
    z0, u_seq = _inputs(batch=1, horizon=1)
    A_bar, _ = model.local_operator(model.lift(z0), u_seq[0], PredictorState())
    A_bar_x, _ = _van_loan(*_physical(M, K, K_C, C), DELTA)
    err = _eig_max_error(
        torch.linalg.eigvals(A_bar[0]), torch.linalg.eigvals(A_bar_x)
    )
    assert err < TOL_EIG, f"eigenvalue error {err:.3e}"


# ---------------------------------------------------------------------------
# negatives -- 4/4 must be red (REGLES section 7.1)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("factory", ARMS, ids=ARM_IDS)
def test_negative_flipped_rotation_sign(factory):
    """``omega_+ -> -omega_+`` on ONE mode: the gold must collapse.

    The flip conjugates that block's rotation (and changes ``Phi``, hence
    ``B_bar``), which a test blind to orientation would miss entirely.
    """
    alpha, w_p, w_m = _modal(M, K, K_C, C)
    S, S_inv, B_h = _modal_maps(alpha, w_p, w_m, M)
    model = _configure(_build(factory), S, S_inv, _theta_0(alpha, -w_p, w_m, B_h))
    z0, u_seq = _inputs(batch=3, horizon=1)
    h_next, _ = model.step(model.lift(z0), u_seq[0], PredictorState())
    err = (model.project(h_next) - _reference_rollout(z0, u_seq)[0]).abs().max().item()
    assert err > 1e6 * TOL_STEP, f"flipped rotation went unnoticed: error {err:.3e}"


@pytest.mark.parametrize("factory", ARMS, ids=ARM_IDS)
def test_negative_wrong_coupling_constant(factory):
    """The model's oracle is built for ``k_c + 0.05``, the physics keeps ``k_c``.

    Only the ANTISYMMETRIC mode sees ``k_c``, so this also proves the gold is
    not blind to the second block.
    """
    alpha, w_p, w_m = _modal(M, K, K_C + 0.05, C)
    S, S_inv, B_h = _modal_maps(alpha, w_p, w_m, M)
    model = _configure(_build(factory), S, S_inv, _theta_0(alpha, w_p, w_m, B_h))
    z0, u_seq = _inputs(batch=3, horizon=1)
    h_next, _ = model.step(model.lift(z0), u_seq[0], PredictorState())
    err = (model.project(h_next) - _reference_rollout(z0, u_seq)[0]).abs().max().item()
    assert err > 1e6 * TOL_STEP, f"wrong k_c went unnoticed: error {err:.3e}"


@pytest.mark.parametrize("factory", ARMS, ids=ARM_IDS)
def test_negative_lift_too_short(factory):
    """``r = 1`` (``2r = 2 < 4``): the 4D system CANNOT be represented, at all.

    Not "this particular configuration fails" -- with linear ``psi: R^4 -> R^2``
    and ``pi: R^2 -> R^4`` the whole one-step map factors through 2 dimensions,
    so its linear part has rank <= 2 whatever the head does. Eckart-Young then
    bounds the error of ANY ``r = 1`` model from below by ``sigma_3(A_bar_x)``,
    measured here at ~1.0 -- eight decades above the rollout tolerance. The
    best-effort model (configured for the symmetric mode, ``pi`` its
    pseudo-inverse) is built too, so the bound is exhibited and not just
    invoked.
    """
    alpha, w_p, w_m = _modal(M, K, K_C, C)
    S, _, B_h = _modal_maps(alpha, w_p, w_m, M)
    S_short = S[:2]  # keep the symmetric mode; pi is its pseudo-inverse
    model = _build(factory, r=1)
    with torch.no_grad():
        model.psi[0].weight.copy_(S_short)
        model.psi[0].bias.zero_()
        model.pi[0].weight.copy_(torch.linalg.pinv(S_short))
        model.pi[0].bias.zero_()
        model.head.bias.copy_(
            torch.cat([torch.tensor([alpha, w_p], dtype=DTYPE), B_h[:2].reshape(-1)])
        )

    A_bar_x, _ = _van_loan(*_physical(M, K, K_C, C), DELTA)
    # the model's one-step LINEAR part, read off the basis at u = 0
    basis = torch.eye(4, dtype=DTYPE)
    zero_u = torch.zeros(4, D_U, dtype=DTYPE)
    h_next, _ = model.step(model.lift(basis), zero_u, PredictorState())
    linear_part = model.project(h_next).T  # columns are the images of e_i

    sigma_3 = torch.linalg.svdvals(A_bar_x)[2].item()
    assert sigma_3 / 4.0 > TOL_ROLLOUT, "rank bound must exceed the gold tolerance"
    # Eckart-Young: no rank-2 matrix gets closer than sigma_3 in spectral norm
    gap = torch.linalg.matrix_norm(linear_part - A_bar_x, ord=2).item()
    assert gap >= sigma_3 - 1e-12, f"rank-2 bound violated: {gap:.3e} < {sigma_3:.3e}"
    assert torch.linalg.matrix_rank(linear_part).item() <= 2

    z0, u_seq = _inputs(batch=3, horizon=1)
    h_next, _ = model.step(model.lift(z0), u_seq[0], PredictorState())
    err = (model.project(h_next) - _reference_rollout(z0, u_seq)[0]).abs().max().item()
    assert err > 1e6 * TOL_STEP, f"a 2D lift reproduced a 4D system: error {err:.3e}"


def test_negative_overdamped_regime_is_refused():
    """``Omega_pm^2 <= alpha^2``: real physical parameters, not a hand-set omega.

    ``c = 5.0`` at ``m = 1.3``, ``k = 2.7`` is past the critical damping of BOTH
    modes (``c_crit = 2 sqrt(k m) = 3.75`` for the symmetric one), so the modal
    lift has no real ``omega`` and ``_modal`` must refuse rather than return a
    NaN. The structural reason is exhibited underneath: the discrete spectrum
    becomes REAL and its four values are DISTINCT, while a rotation-scaling
    block forces its two eigenvalues to be conjugates -- equal as soon as they
    are real. No ``(alpha_i, omega_i)`` can match that multiset.
    """
    c_over = 5.0
    alpha = -c_over / (2.0 * M)
    assert K / M <= alpha**2 and (K + 2.0 * K_C) / M <= alpha**2
    with pytest.raises(ValueError, match="not underdamped"):
        _modal(M, K, K_C, c_over)

    A_bar_x, _ = _van_loan(*_physical(M, K, K_C, c_over), DELTA)
    eig = torch.linalg.eigvals(A_bar_x)
    assert eig.imag.abs().max().item() < 1e-12  # no oscillation left
    real = sorted(eig.real.tolist())
    gaps = [b - a for a, b in zip(real[:-1], real[1:], strict=True)]
    # every consecutive gap is wide: whichever two eigenvalues a block is asked
    # to carry, it would have to make them equal
    assert min(gaps) > 1e-3, f"eigenvalues not distinct enough: {gaps}"
