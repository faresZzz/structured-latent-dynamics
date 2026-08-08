"""D3's gold: PROTOCOL.md section 19 (amendment 10), rewritten -- section
19.2's bounds are now taken in STATE coordinates on the effective one-step
map, at ``r = 2`` (a CHOICE, phase-C's own proven configuration -- no longer
a precondition, since ``J_x`` is ``d x d`` for any ``2r``).

Four bounds, on held-out, in GENUINELY FREE rollout: state error at
``H = 100`` (section 19.1, scored in STATE coordinates -- identity
encoder/readout, so ``e_recon = 0``), the relative error of ``J_x``'s
eigenvalues under OPTIMAL pairing, the relative error of ``J_u``, and the
improvement over the two baselines (that fourth bound is
``sld.experiments.gate``'s own job and is not re-exercised here).

This file tests ``sld.experiments.oscillator_gold``: the pure primitives
(``discrete_reference``, ``eig_relative_error``, ``action_relative_error``,
``local_jacobians``, ``oscillator_bounds``) behind the SPECTRAL and ACTION
bounds, plus the state bound wired through the real, unmodified
``sld.experiments.gate.evaluate_free_rollout`` (never reimplemented here).
Modeled on ``tests/predictors/test_structured_oscillator_gold.py``
(phase-C's own gold): same R2 discipline (REGLES section 7.1) -- every
reference used to judge the object under test is built by a DIFFERENT
algorithm than the one under test, never the same closed form looking at
itself.

WHAT CHANGED, AND WHY (kept as history, not just conclusion -- REGLES
hierarchy: correction > honesty > readability > speed). An earlier version
of ``oscillator_bounds`` compared ``B_bar`` -- read directly out of
``local_operator``'s block basis -- against ``Phi B``. Building the
planted-model positive control for THAT version required an elaborate
construction (solving, per rotation-scaling block, for the ONE member of
the basis-change family that also fixed the action vector exactly) because
the comparison was basis-DEPENDENT: any OTHER valid change of basis (e.g.
phase-C's own modal transform) scored O(1), not ~0, despite exactly
reproducing the physics. That finding, and a second one (the block-basis
spectrum is blind to a rotation-sign flip: eigenvalue SETS of a conjugate
pair are invariant under ``omega -> -omega``), are what motivated section
19.2's rewrite: both bounds are now taken on ``J_x``/``J_u``, the Jacobians
of the STATE-coordinate map ``x_hat = pi(A_bar psi(x) + B_bar u)`` (see
``oscillator_gold.py``'s own module docstring for the full argument). One
consequence worth stating plainly: under the new formulation, ANY valid
block-diagonalizing basis reproduces the exact physics through ``J_x``/
``J_u`` (verified below, ``test_planted_model_...``) -- the elaborate
"action-consistent" construction is no longer needed, and this file uses
phase-C's own plain modal transform throughout.

A THIRD finding, from re-measuring the rotation-sign-flip negative under
the new formulation (as requested): the state-coordinate rewrite does NOT
repair the spectral bound's blindness to that specific mutation. ``J_x``'s
eigenvalues are a similarity invariant of ``A_bar``, and ``omega -> -omega``
leaves a conjugate pair's eigenvalue SET unchanged regardless of which
similarity class ``A_bar`` sits in -- state coordinates change nothing
about that fact. Measured below
(``test_negative_flipped_rotation_sign_...``): ``eig_relative_error(J_x,
...)`` stays at ~1e-16, both before and after the coordinate change. The
ACTION bound catches the mutation robustly (``J_u`` is not spectrum-only).
This is reported as measured, not smoothed over.

VOCABULARY (REGLES section 3). ``psi``/``pi`` are MLPs in a real run
(``psi_hidden``/``pi_hidden`` default to ``(64,)``, not config keys); this
file's own planted models use a single LINEAR layer (``psi_hidden=()``) to
keep the positive control exact and hand-checkable, which is NOT D3's real
regime -- ``test_production_path_short_...`` below exercises the real,
nonlinear default instead. What ``oscillator_bounds`` reports across probe
states is a FIELD of local Jacobians, never "the operator" or "the
spectrum". Nothing computed anywhere in this file is a Koopman operator or
a stability certificate.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from sld.audit.rollout import normalized_state_error
from sld.experiments.gate import evaluate_free_rollout
from sld.experiments.oscillator_gold import (
    action_relative_error,
    discrete_reference,
    eig_relative_error,
    local_jacobians,
    oscillator_bounds,
)
from sld.predictors.base import PredictorState
from sld.predictors.structured import (
    MLPKPredictor,
    SSMKPredictor,
    StructuredPredictor,
    TransformerKPredictor,
)
from sld.systems.coupled_oscillators import CoupledOscillators

DTYPE = torch.float64

# Same symmetric, underdamped point phase-C uses (m1=m2, k1=k2 -- required
# for the closed-form modal split below); reused deliberately rather than
# re-derived, to keep this file's own risk surface small.
M, K, K_C, C = 1.3, 2.7, 0.55, 0.35
DELTA = 0.05
R, D_Z, D_U = 2, 4, 1
WIDTH = 8
HORIZON = 100

TOL_STATE = 1e-9
TOL_SPECTRAL = 1e-9
TOL_ACTION = 1e-9

ARMS = [MLPKPredictor, TransformerKPredictor, SSMKPredictor]
ARM_IDS = ["mlp_k", "transformer_k", "ssm_k"]


# ---------------------------------------------------------------------------
# physics, by hand -- NOT CoupledOscillators (the object under test's
# reference must not be assembled from the object it is validating)
# ---------------------------------------------------------------------------
def _physical(m: float, k: float, k_c: float, c: float) -> tuple[Tensor, Tensor]:
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


def _modal(m: float, k: float, k_c: float, c: float) -> tuple[float, float, float]:
    """``(alpha, omega_+, omega_-)`` -- same closed form as phase-C's own
    ``_modal`` (test_structured_oscillator_gold.py), reused rather than
    imported: independent test files, independent risk."""
    alpha = -c / (2.0 * m)
    out = [alpha]
    for omega_sq in (k / m, (k + 2.0 * k_c) / m):
        if omega_sq <= alpha**2:
            raise ValueError(f"not underdamped: Omega^2={omega_sq!r} <= alpha^2={alpha**2!r}")
        out.append(math.sqrt(omega_sq - alpha**2))
    return out[0], out[1], out[2]


def _modal_maps(alpha: float, w_p: float, w_m: float, m: float) -> tuple[Tensor, Tensor]:
    """``(S, S_inv)`` -- the symmetric/antisymmetric modal transform, same
    formula as phase-C's own ``_modal_maps``. ANY valid block-diagonalizing
    transform works equally well for the STATE-coordinate bounds below
    (module docstring); this is just one, picked because it is already
    proven correct elsewhere in this repo."""
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
    return S, S_inv


def _theta_0(alpha: float, w_p: float, w_m: float, B_h: Tensor) -> Tensor:
    return torch.cat([torch.tensor([alpha, alpha, w_p, w_m], dtype=DTYPE), B_h.reshape(-1)])


def _build(factory, r: int = R, delta: float = DELTA) -> StructuredPredictor:
    # ``.to(dtype=DTYPE)`` on top of the constructor's own ``dtype=`` kwarg:
    # CausalTransformerContextBackbone's sinusoidal positional-encoding
    # buffer (transformer.py's ``_build_sinusoidal_pe``) is built as float32
    # unconditionally, regardless of the predictor's own declared dtype --
    # a pre-existing quirk of that module (out of this file's scope: the
    # packet's file boundary forbids touching transformer.py), worked
    # around here the same way ``evaluate_run_free_rollout`` already reaches
    # CPU float64 for ANY checkpoint (gate.py: ``copy.deepcopy(...).double()``).
    return factory(
        D_Z, D_U, delta, r=r, width=WIDTH, psi_hidden=(), pi_hidden=(), dtype=DTYPE
    ).to(dtype=DTYPE)


def _configure(model: StructuredPredictor, S: Tensor, S_inv: Tensor, theta_0: Tensor):
    with torch.no_grad():
        model.psi[0].weight.copy_(S)
        model.psi[0].bias.zero_()
        model.pi[0].weight.copy_(S_inv)
        model.pi[0].bias.zero_()
        model.head.bias.copy_(theta_0)
    return model


def _gold_model(
    factory, *, m: float = M, k: float = K, k_c: float = K_C, c: float = C
) -> StructuredPredictor:
    alpha, w_p, w_m = _modal(m, k, k_c, c)
    S, S_inv = _modal_maps(alpha, w_p, w_m, m)
    _, B_x = _physical(m, k, k_c, c)
    B_h = S @ B_x
    return _configure(_build(factory), S, S_inv, _theta_0(alpha, w_p, w_m, B_h))


def _inputs(batch: int, seed: int) -> tuple[Tensor, Tensor]:
    """``(x0, u0)``: STATE-space probes and actions (D3 identity
    representation, section 19.1 -- ``x == z``)."""
    g = torch.Generator().manual_seed(seed)
    x0 = torch.rand(batch, D_Z, generator=g, dtype=DTYPE) * 2.0 - 1.0
    u0 = torch.rand(batch, D_U, generator=g, dtype=DTYPE) * 2.0 - 1.0
    return x0, u0


def _lti_eval_split(n_episodes: int, T: int, seed: int) -> dict:
    """A hand-built EXACT-ZOH dataset in state coordinates (``x == z_true``,
    matching D3's identity encoder/readout, section 19.1): generated from
    the SAME independent ``discrete_reference`` oracle, not from
    ``system.integrate`` (RK4) -- an RK4-integrated split would fold its own
    O(dt^5) truncation error into a claim about the MODEL's fidelity, which
    is exactly the confound phase-C's own state-space rollout test avoids by
    using its ``_van_loan``-derived reference for data too (test_gate.py's
    ``_toy_lti_split`` does the same: the exact discrete update generates
    the data, so a mismatch downstream is attributable to the model under
    test)."""
    A_x, B_x = _physical(M, K, K_C, C)
    A_d, B_d = discrete_reference(A_x, B_x, DELTA)
    g = torch.Generator().manual_seed(seed)
    z_true = torch.empty(n_episodes, T + 1, D_Z, dtype=DTYPE)
    u = torch.rand(n_episodes, T, D_U, generator=g, dtype=DTYPE) * 2.0 - 1.0
    z_true[:, 0, :] = torch.rand(n_episodes, D_Z, generator=g, dtype=DTYPE) * 2.0 - 1.0
    for t in range(T):
        z_true[:, t + 1, :] = z_true[:, t, :] @ A_d.T + u[:, t, :] @ B_d.T
    z_true_np = z_true.numpy()
    u_np = u.numpy()
    return {"x": z_true_np.copy(), "u": u_np, "z_true": z_true_np}


def _identity_readout(d: int) -> np.ndarray:
    W = np.zeros((d + 1, d), dtype=np.float64)
    W[:d, :] = np.eye(d)
    return W


class _IdentityEncoder(nn.Module):
    """``obs == state`` exactly -- D3's own state-coordinates control
    (section 19.1): ``d_z = d_x``, no parameters."""

    def forward(self, x: Tensor) -> Tensor:
        return x


def _state_bound_e_h(model: StructuredPredictor, *, horizon: int, seed: int) -> np.ndarray:
    """The state bound's ``e_h`` curve, via the REAL, unmodified
    ``evaluate_free_rollout`` -- never reimplemented here (module docstring)."""
    split = _lti_eval_split(n_episodes=3, T=horizon + 4, seed=seed)
    result = evaluate_free_rollout(
        _IdentityEncoder(),
        model,
        split,
        W=_identity_readout(D_Z),
        sigma_train=np.ones(D_Z),
        horizon=horizon,
        tau=0.5,
        h_gate=horizon,
        gamma=5.0,
        tau_1=1.0,
    )
    return result.e_h


def _system(m: float = M, k: float = K, k_c: float = K_C, c: float = C) -> CoupledOscillators:
    return CoupledOscillators(m1=m, m2=m, k1=k, k_c=k_c, k2=k, c=c)


# ---------------------------------------------------------------------------
# the montage, checked before it is used (mirrors phase-C's own
# test_hand_built_system_is_the_repo_system / test_modal_change_of_basis...)
# ---------------------------------------------------------------------------
def test_hand_built_system_is_the_repo_system():
    A_x, B_x = _physical(M, K, K_C, C)
    system = _system()
    torch.testing.assert_close(A_x, torch.as_tensor(system.A_mat), atol=1e-15, rtol=0)
    torch.testing.assert_close(B_x, torch.as_tensor(system.B_mat), atol=1e-15, rtol=0)


def test_modal_transform_block_diagonalises_the_generator():
    """The montage control this file's gold depends on -- same check as
    phase-C's own ``test_modal_change_of_basis_block_diagonalises_the_generator``."""
    alpha, w_p, w_m = _modal(M, K, K_C, C)
    S, S_inv = _modal_maps(alpha, w_p, w_m, M)
    torch.testing.assert_close(S_inv @ S, torch.eye(4, dtype=DTYPE), atol=1e-15, rtol=0)

    expected = torch.zeros(4, 4, dtype=DTYPE)
    expected[0:2, 0:2] = torch.tensor([[alpha, -w_p], [w_p, alpha]], dtype=DTYPE)
    expected[2:4, 2:4] = torch.tensor([[alpha, -w_m], [w_m, alpha]], dtype=DTYPE)
    A_x, _ = _physical(M, K, K_C, C)
    torch.testing.assert_close(S @ A_x @ S_inv, expected, atol=1e-14, rtol=0)


@pytest.mark.parametrize("factory", ARMS, ids=ARM_IDS)
def test_head_weight_is_exactly_zero_at_construction(factory):
    assert bool((_build(factory).head.weight == 0).all())


# ---------------------------------------------------------------------------
# Positive 1: the discrete_reference oracle, cross-checked by a THIRD,
# independent path (a truncated Taylor series -- neither Van Loan's
# scaling-and-squaring matrix_exp nor rotation_scaling_zoh's closed form)
# ---------------------------------------------------------------------------
def _taylor_zoh(A: Tensor, B: Tensor, delta: float, n_terms: int = 60) -> tuple[Tensor, Tensor]:
    """``A_d = sum_k (delta*A)^k / k!``, ``Phi = sum_k delta*(delta*A)^k / (k+1)!``
    -- the defining power series of the matrix exponential and its ZOH
    integral, truncated. Algorithmically independent of ``torch.matrix_exp``
    (scaling-and-squaring) and of ``rotation_scaling_zoh``'s closed-form
    ``(I, J)`` coefficients: this file's own oracle for the oracle.
    """
    n = A.shape[-1]
    dA = delta * A
    power = torch.eye(n, dtype=A.dtype)  # (delta*A)^k, k starts at 0
    A_d = torch.zeros(n, n, dtype=A.dtype)
    Phi = torch.zeros(n, n, dtype=A.dtype)
    for kk in range(n_terms):
        A_d = A_d + power / float(math.factorial(kk))
        Phi = Phi + power * (delta / float(math.factorial(kk + 1)))
        power = power @ dA
    return A_d, Phi @ B


def test_discrete_reference_cross_checked_by_truncated_taylor_series():
    A_x, B_x = _physical(M, K, K_C, C)
    A_d, B_d = discrete_reference(A_x, B_x, DELTA)
    A_taylor, B_taylor = _taylor_zoh(A_x, B_x, DELTA, n_terms=60)
    err_A = (A_d - A_taylor).abs().max().item()
    err_B = (B_d - B_taylor).abs().max().item()
    print(f"[oscillator_gold] Van Loan vs truncated Taylor: err_A={err_A:.3e} err_B={err_B:.3e}")
    assert err_A < 1e-13
    assert err_B < 1e-13


def test_van_loan_helper_matches_the_scalar_closed_form():
    """Positive on the oracle itself: 1x1, where ZOH is elementary."""
    a, delta = -0.7, 0.05
    A_d, B_d = discrete_reference(
        torch.tensor([[a]], dtype=DTYPE), torch.ones(1, 1, dtype=DTYPE), delta
    )
    assert abs(A_d.item() - math.exp(a * delta)) < 1e-15
    assert abs(B_d.item() - (math.exp(a * delta) - 1.0) / a) < 1e-15


# ---------------------------------------------------------------------------
# Positive 1.5: the Jacobian itself, cross-checked by a SECOND, independent
# path -- central finite differences (R2; the coordinator's own requirement,
# in the style of phase A's ZOH-gradient recoupment)
# ---------------------------------------------------------------------------
def _central_difference_jacobian(f, x: Tensor, step: float = 1e-6) -> Tensor:
    """``(m, n)`` central-difference Jacobian of ``f: R^n -> R^m`` at ``x``.
    Independent of ``torch.autograd`` end to end: no graph, no backward
    pass, just forward evaluations of ``f`` and a finite ratio.
    """
    cols = []
    for j in range(x.shape[0]):
        e = torch.zeros_like(x)
        e[j] = step
        cols.append((f(x + e) - f(x - e)) / (2.0 * step))
    return torch.stack(cols, dim=1)


@pytest.mark.parametrize("factory", ARMS, ids=ARM_IDS)
def test_local_jacobians_cross_checked_by_central_finite_differences(factory):
    """``local_jacobians``' autograd output, re-derived by an ALGORITHM that
    never touches ``torch.autograd`` -- an autograd Jacobian validated only
    against itself is exactly what R2 forbids."""
    model = _gold_model(factory)
    x0, u0 = _inputs(batch=1, seed=60)
    x0v, u0v = x0[0], u0[0]
    J_x, J_u = local_jacobians(model, x0v, u0v)

    h0 = model.lift(x0v.unsqueeze(0))
    A_bar, B_bar = model.local_operator(h0, u0v.unsqueeze(0), PredictorState())
    A_bar, B_bar = A_bar[0].detach(), B_bar[0].detach()

    def f(x_: Tensor, u_: Tensor) -> Tensor:
        h_ = model.lift(x_.unsqueeze(0))[0]
        h_next = A_bar @ h_ + B_bar @ u_
        return model.project(h_next.unsqueeze(0))[0]

    J_x_fd = _central_difference_jacobian(lambda x_: f(x_, u0v), x0v, step=1e-6)
    J_u_fd = _central_difference_jacobian(lambda u_: f(x0v, u_), u0v, step=1e-6)

    err_x = (J_x - J_x_fd).abs().max().item()
    err_u = (J_u - J_u_fd).abs().max().item()
    print(f"[oscillator_gold] autograd vs central-diff: err_x={err_x:.3e} err_u={err_u:.3e}")
    assert err_x < 1e-8
    assert err_u < 1e-8


# ---------------------------------------------------------------------------
# Positive 2: the planted model, all three bounds near machine precision
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("factory", ARMS, ids=ARM_IDS)
def test_planted_model_satisfies_all_three_bounds_near_machine_precision(factory):
    """The strongest positive control this file has: a model whose
    ``(psi, pi, theta_0)`` are planted analytically (never trained) to
    reproduce the physics exactly, in phase-C's OWN plain modal basis --
    under the state-coordinate formulation, ANY valid block-diagonalizing
    basis works (module docstring); no special construction is needed
    anymore. If this were at 1e-3 rather than ~1e-15, the evaluator -- not
    the physics -- would be suspect."""
    model = _gold_model(factory)
    system = _system()

    e_h = _state_bound_e_h(model, horizon=HORIZON, seed=10)
    max_state_err = float(np.max(np.abs(e_h)))

    x0, u0 = _inputs(batch=5, seed=11)
    bounds = oscillator_bounds(model, system, delta=DELTA, probe_states=x0, probe_actions=u0)

    print(
        f"[oscillator_gold] planted {factory.__name__}: "
        f"state(H={HORIZON})={max_state_err:.3e} "
        f"spectral={bounds['eig_relative_error']:.3e} "
        f"action={bounds['action_relative_error']:.3e}"
    )
    assert max_state_err < TOL_STATE, f"state bound: {max_state_err:.3e}"
    assert bounds["eig_relative_error"] < TOL_SPECTRAL, f"spectral bound: {bounds['eig_relative_error']:.3e}"
    assert bounds["action_relative_error"] < TOL_ACTION, f"action bound: {bounds['action_relative_error']:.3e}"


def test_planted_model_bounds_are_well_posed_independent_of_r():
    """Section 19.2's own point 5: ``J_x``/``J_u`` are ``d x d``/``d x d_u``
    whatever ``2r`` is. Built at ``r = 4`` (``2r = 8 > d = 4``, exactly the
    configuration the ORIGINAL block-basis bound had no truth value at,
    section 19.2's "defect"): ``psi`` embeds ``x`` into the first 4 of 8
    lifted coordinates via the modal ``S``, leaving the other 4 identically
    zero; ``pi`` reads only those first 4 back. The extra block's own
    ``(alpha, omega)`` are arbitrary (picked deliberately far from the
    physical ones) -- they must not matter, since ``pi`` never reads them.
    """
    alpha, w_p, w_m = _modal(M, K, K_C, C)
    S, S_inv = _modal_maps(alpha, w_p, w_m, M)
    _, B_x = _physical(M, K, K_C, C)
    B_h = S @ B_x

    r = 4
    d_h = 2 * r
    model = _build(MLPKPredictor, r=r)
    psi_w = torch.zeros(d_h, D_Z, dtype=DTYPE)
    psi_w[0:4, :] = S
    pi_w = torch.zeros(D_Z, d_h, dtype=DTYPE)
    pi_w[:, 0:4] = S_inv
    alphas = torch.tensor([alpha, alpha, 0.7, -0.3], dtype=DTYPE)  # extra block: arbitrary
    omegas = torch.tensor([w_p, w_m, 1.1, 0.4], dtype=DTYPE)
    vec_B = torch.zeros(d_h, D_U, dtype=DTYPE)
    vec_B[0:4, :] = B_h
    theta_0 = torch.cat([alphas, omegas, vec_B.reshape(-1)])
    with torch.no_grad():
        model.psi[0].weight.copy_(psi_w)
        model.psi[0].bias.zero_()
        model.pi[0].weight.copy_(pi_w)
        model.pi[0].bias.zero_()
        model.head.bias.copy_(theta_0)

    system = _system()
    x0, u0 = _inputs(batch=5, seed=61)
    bounds = oscillator_bounds(model, system, delta=DELTA, probe_states=x0, probe_actions=u0)
    print(
        f"[oscillator_gold] r=4 (2r=8>d=4): spectral={bounds['eig_relative_error']:.3e} "
        f"action={bounds['action_relative_error']:.3e}"
    )
    assert bounds["eig_relative_error"] < TOL_SPECTRAL
    assert bounds["action_relative_error"] < TOL_ACTION


# ---------------------------------------------------------------------------
# Positive 2.5: basis-transform invariance -- the direct proof that the
# repair works (coordinator's requirement 4)
# ---------------------------------------------------------------------------
def test_basis_transform_invariance_of_jx_ju_while_raw_operator_moves():
    """Section 19.2's own repair argument, made falsifiable: for ANY
    invertible ``T``, ``(psi, A_bar, B_bar, pi)`` and
    ``(T psi, T A_bar T^-1, T B_bar, pi T^-1)`` predict IDENTICALLY, so
    ``J_x``/``J_u`` are invariant under it -- while the raw operator pair
    moves.

    ``T`` here is an ANISOTROPIC per-coordinate diagonal scaling,
    deliberately OUTSIDE the family that commutes with a rotation-scaling
    block (``a*I + b*J``): every element of THAT family leaves ``A_bar``
    EXACTLY unchanged by construction (it commutes with the target block by
    definition -- verified separately below), which would understate this
    test's claim ("the raw operator moves"). A ``T`` outside that family
    cannot be realized by a second real ``StructuredPredictor`` (
    ``rotation_scaling_zoh`` only ever emits genuine block form), so both
    sides are evaluated directly from the closed-form affine map, reusing
    ``model.lift``/``model.project`` (the real ``psi``/``pi``) wherever
    possible.
    """
    model = _gold_model(MLPKPredictor)
    x0, u0 = _inputs(batch=1, seed=70)
    x0v, u0v = x0[0], u0[0]

    J_x_old, J_u_old = local_jacobians(model, x0v, u0v)

    h0 = model.lift(x0v.unsqueeze(0))
    A_bar, B_bar = model.local_operator(h0, u0v.unsqueeze(0), PredictorState())
    A_bar, B_bar = A_bar[0].detach(), B_bar[0].detach()

    T = torch.diag(torch.tensor([1.4, 0.6, 1.2, 0.8], dtype=DTYPE))
    T_inv = torch.linalg.inv(T)
    A_new = T @ A_bar @ T_inv
    B_new = T @ B_bar
    move_A = (A_new - A_bar).abs().max().item()
    move_B = (B_new - B_bar).abs().max().item()
    print(f"[oscillator_gold] basis transform: A_bar moves by {move_A:.3e}, B_bar by {move_B:.3e}")
    assert move_A > 1e-3, "positive control: T must actually move A_bar"
    assert move_B > 1e-3, "positive control: T must actually move B_bar"

    def f_new(x_: Tensor, u_: Tensor) -> Tensor:
        h_ = T @ model.lift(x_.unsqueeze(0))[0]  # psi_new(x) = T @ psi(x)
        h_next_new = A_new @ h_ + B_new @ u_
        return model.project((T_inv @ h_next_new).unsqueeze(0))[0]  # pi_new = pi . T^-1

    J_x_new = torch.autograd.functional.jacobian(lambda x_: f_new(x_, u0v), x0v)
    J_u_new = torch.autograd.functional.jacobian(lambda u_: f_new(x0v, u_), u0v)

    err_x = (J_x_new - J_x_old).abs().max().item()
    err_u = (J_u_new - J_u_old).abs().max().item()
    print(f"[oscillator_gold] basis transform: J_x invariance err={err_x:.3e} J_u invariance err={err_u:.3e}")
    assert err_x < 1e-13
    assert err_u < 1e-13


def test_block_commuting_transform_leaves_a_bar_exactly_unchanged():
    """The complementary fact the test above relies on: a transform WITHIN
    the family that commutes with a rotation-scaling block (a real
    per-block scalar, the simplest member) leaves ``A_bar`` EXACTLY
    unchanged (not merely close) -- this is why that family was NOT used
    above to demonstrate "the raw operator moves": it structurally can't,
    for ``A_bar``. ``B_bar`` still moves under it, which is itself exactly
    this file's original (pre-repair) finding about the block-basis action
    bound (module docstring)."""
    model = _gold_model(MLPKPredictor)
    x0, u0 = _inputs(batch=1, seed=71)
    h0 = model.lift(x0[0].unsqueeze(0))
    A_bar, B_bar = model.local_operator(h0, u0[0].unsqueeze(0), PredictorState())
    A_bar, B_bar = A_bar[0].detach(), B_bar[0].detach()

    T = torch.diag(torch.tensor([1.4, 1.4, 0.6, 0.6], dtype=DTYPE))  # real scalar PER BLOCK
    T_inv = torch.linalg.inv(T)
    A_new = T @ A_bar @ T_inv
    B_new = T @ B_bar
    err_A = (A_new - A_bar).abs().max().item()
    move_B = (B_new - B_bar).abs().max().item()
    print(f"[oscillator_gold] block-commuting T: A_bar unchanged to {err_A:.3e}, B_bar moves by {move_B:.3e}")
    assert err_A < 1e-13
    assert move_B > 1e-3


# ---------------------------------------------------------------------------
# Positive 3: production-path SMOKE TEST -- NOT a proof of learning
# ---------------------------------------------------------------------------
def test_production_path_short_train_checkpoint_reload_evaluate_produces_finite_numbers(tmp_path):
    """SMOKE TEST ONLY. A ridiculous budget (20 optimizer updates over 8
    training episodes of length 15) trains NOTHING -- this test does not,
    and must not be read to, demonstrate learning. It proves the WIRING:
    ``Trainer.fit`` -> checkpoint -> ``load_run`` -> ``evaluate_free_rollout``
    -> ``oscillator_bounds`` survives a real run end to end and returns
    finite numbers, on the real ``mlp_k`` family at ``r = 2`` against the
    real ``CoupledOscillators`` system, with ``psi``/``pi`` at their REAL
    default (nonlinear, ``hidden=(64,)``) -- never that those numbers are
    small.
    """
    from sld.data.generate import build_dataset
    from sld.data.regimes import OURegime
    from sld.encoders.pretrain import (
        load_frozen_representation,
        pretrain_representation,
        save_frozen_representation,
    )
    from sld.experiments.loading import load_run
    from sld.training.trainer import Trainer

    system = _system()
    data = build_dataset(
        system=system, regime=OURegime(theta_ou=1.0), obs=None, n_episodes=10, T=15, dt=DELTA, seed=0
    )
    art = pretrain_representation(
        data, system="CoupledOscillators", regime="OURegime", d_z=D_Z,
        encoder_hidden=(8,), seed=1, epochs=2, batch_size=8,
    )
    rep_path = tmp_path / "rep.pt"
    save_frozen_representation(art, rep_path)
    art = load_frozen_representation(rep_path)

    config = {
        "exp_name": "oscillator_gold_smoke",
        "seed": 0,
        "family": "mlp_k",
        "d_x": D_Z, "d_z": D_Z, "d_u": D_U,
        "delta": DELTA, "r": R,
        "encoder_hidden": [8],
        "target_params": 500,
        "epochs": 20,
        "batch_episodes": 2,
        "lr": 5e-2,
        "h_fr": 3,
        "results_root": str(tmp_path / "results"),
        "stratum_system": "CoupledOscillators",
        "stratum_regime": "OURegime",
    }
    trainer = Trainer(config, representation=art)
    result = trainer.fit(data)
    run_dir = Path(result["paths"]["ckpt"]).parent

    run = load_run(run_dir)
    predictor64 = copy.deepcopy(run["predictor"]).double().eval()
    encoder64 = copy.deepcopy(run["encoder"]).double().eval()

    x0, u0 = _inputs(batch=4, seed=12)
    bounds = oscillator_bounds(predictor64, system, delta=DELTA, probe_states=x0, probe_actions=u0)
    fr = evaluate_free_rollout(
        encoder64, predictor64, data["test"],
        W=art.readout.W, sigma_train=art.sigma_train,
        horizon=3, tau=0.5, h_gate=3, gamma=5.0, tau_1=1.0,
    )

    print(
        f"[oscillator_gold] production-path smoke: spectral={bounds['eig_relative_error']:.3e} "
        f"action={bounds['action_relative_error']:.3e} e_h={fr.e_h}"
    )
    assert math.isfinite(bounds["eig_relative_error"])
    assert math.isfinite(bounds["action_relative_error"])
    assert np.isfinite(fr.e_h).all()


# ---------------------------------------------------------------------------
# Negatives -- each must bite for the RIGHT reason (REGLES section 7.1)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("factory", ARMS, ids=ARM_IDS)
def test_negative_flipped_rotation_sign_blows_up_the_action_bound(factory):
    """``omega_+ -> -omega_+`` on ONE mode, ``B_h`` left as-is (still built
    for ``+omega_+``), re-measured in the NEW state-coordinate formulation
    as requested.

    Re-measured finding, reported as measured rather than assumed: the
    SPECTRAL bound is STILL structurally blind to this mutation, even in
    state coordinates. ``J_x``'s eigenvalues are a similarity invariant of
    ``A_bar``, and ``omega -> -omega`` leaves a conjugate pair's eigenvalue
    SET unchanged regardless of similarity class -- moving to state
    coordinates does not touch this fact (``oscillator_gold.py``'s own
    module docstring, point 2). Measured: ``eig_relative_error(J_x, ...)``
    stays at ~1e-16, essentially IDENTICAL to the positive control -- pinned
    below rather than left as a surprise. The ACTION bound is what catches
    it (``J_u`` is not spectrum-only): the block's rotation direction now
    disagrees with the continuous input map ``B_h`` (still built for the
    ORIGINAL sign).
    """
    alpha, w_p, w_m = _modal(M, K, K_C, C)
    S, S_inv = _modal_maps(alpha, w_p, w_m, M)
    _, B_x = _physical(M, K, K_C, C)
    B_h = S @ B_x
    model = _configure(_build(factory), S, S_inv, _theta_0(alpha, -w_p, w_m, B_h))
    system = _system()
    x0, u0 = _inputs(batch=3, seed=20)
    bounds = oscillator_bounds(model, system, delta=DELTA, probe_states=x0, probe_actions=u0)
    print(
        f"[oscillator_gold] flipped rotation (state coords): "
        f"spectral={bounds['eig_relative_error']:.3e} (expected to STAY near machine precision) "
        f"action={bounds['action_relative_error']:.3e}"
    )
    assert bounds["eig_relative_error"] < TOL_SPECTRAL, (
        "pinned finding: the spectral bound must stay blind to a pure rotation-sign flip "
        "even in state coordinates -- if this fails, the sign-invariance argument in "
        "oscillator_gold.py's module docstring needs re-examination"
    )
    assert bounds["action_relative_error"] > 1e-3, "flipped rotation went unnoticed by the action bound"


@pytest.mark.parametrize("factory", ARMS, ids=ARM_IDS)
def test_negative_wrong_coupling_constant_violates_the_spectral_bound(factory):
    """The model is planted for ``k_c + 0.2``; the physics keeps ``k_c``.
    Only the antisymmetric mode sees ``k_c``, so this is not blind to the
    second block."""
    alpha, w_p, w_m = _modal(M, K, K_C + 0.2, C)
    S, S_inv = _modal_maps(alpha, w_p, w_m, M)
    _, B_x = _physical(M, K, K_C + 0.2, C)
    B_h = S @ B_x
    model = _configure(_build(factory), S, S_inv, _theta_0(alpha, w_p, w_m, B_h))
    system = _system()  # TRUE k_c
    x0, u0 = _inputs(batch=3, seed=21)
    bounds = oscillator_bounds(model, system, delta=DELTA, probe_states=x0, probe_actions=u0)
    print(
        f"[oscillator_gold] wrong k_c: spectral={bounds['eig_relative_error']:.3e} "
        f"action={bounds['action_relative_error']:.3e}"
    )
    assert bounds["eig_relative_error"] > 1e-3 or bounds["action_relative_error"] > 1e-3, (
        "wrong coupling constant went unnoticed"
    )


def test_negative_sort_based_pairing_mispairs_conjugate_blocks_sharing_a_real_part():
    """The eliminatory case ``eig_relative_error``'s own docstring names,
    demonstrated deterministically (not left to incidental rounding). Two
    conjugate eigenvalue pairs share the SAME real part ``alpha`` (both
    discrete modes of a symmetric coupled-oscillator system do, to the
    physics) but differ in ``omega``. A real-part-first sort's tie-break is
    then decided by whatever infinitesimal deviation the two matrices
    happen to carry -- injected here explicitly so the failure is
    guaranteed, not merely likely.
    """

    def _rot(a: float, w: float) -> Tensor:
        return torch.tensor([[a, -w], [w, a]], dtype=DTYPE)

    def _blkdiag(a: Tensor, b: Tensor) -> Tensor:
        out = torch.zeros(4, 4, dtype=DTYPE)
        out[0:2, 0:2] = a
        out[2:4, 2:4] = b
        return out

    alpha, w1, w2 = -0.2, 0.9, 1.6
    eps = 1e-10
    A_reference = _blkdiag(_rot(alpha, w1), _rot(alpha, w2))
    # SAME two eigenvalue pairs, block order swapped, each block's real part
    # perturbed by a tiny +-eps -- the "shared real part, asymmetric noise"
    # pattern genuine rounding produces between two independent eigval calls.
    A_learned = _blkdiag(_rot(alpha + eps, w2), _rot(alpha - eps, w1))

    brute = eig_relative_error(A_learned, A_reference)
    assert brute < 1e-8, f"brute force must find the correct (swapped) pairing: {brute:.3e}"

    def _sorted_pairing_error(a_mat: Tensor, b_mat: Tensor) -> float:
        xs = sorted(torch.linalg.eigvals(a_mat).tolist(), key=lambda z: (z.real, z.imag))
        ys = sorted(torch.linalg.eigvals(b_mat).tolist(), key=lambda z: (z.real, z.imag))
        return max(abs(x - y) / abs(y) for x, y in zip(xs, ys, strict=True))

    naive_sorted = _sorted_pairing_error(A_learned, A_reference)
    print(f"[oscillator_gold] pairing: brute={brute:.3e} sorted={naive_sorted:.3e}")
    assert naive_sorted > 0.1, f"a real-part-first sort must mispair here: got {naive_sorted:.3e}"
    assert naive_sorted > 1e6 * brute


def test_negative_teacher_forced_evaluation_disagrees_with_free_rollout():
    """D3's mandatory negative: scoring the state bound with teacher forcing
    instead of a genuinely free rollout gives a DIFFERENT, misleadingly
    small number. Built independently of ``evaluate_free_rollout`` (a hand
    loop over ``model.step``, fed the TRUE state at every step -- never the
    model's own prediction), on a model whose planted ``alpha`` sign is
    flipped POSITIVE (unstable) so the two curves' disagreement is
    unambiguous rather than a matter of degree.
    """
    alpha, w_p, w_m = _modal(M, K, K_C, C)
    S, S_inv = _modal_maps(alpha, w_p, w_m, M)
    _, B_x = _physical(M, K, K_C, C)
    B_h = S @ B_x
    unstable_theta_0 = _theta_0(-alpha, w_p, w_m, B_h)  # alpha < 0 (damped) -> -alpha > 0 (growing)
    model = _configure(_build(MLPKPredictor), S, S_inv, unstable_theta_0)

    horizon = 20
    split = _lti_eval_split(n_episodes=3, T=horizon + 20, seed=22)
    result = evaluate_free_rollout(
        _IdentityEncoder(), model, split, W=_identity_readout(D_Z), sigma_train=np.ones(D_Z),
        horizon=horizon, tau=0.5, h_gate=horizon, gamma=5.0, tau_1=1.0,
    )

    z_true = split["z_true"]
    u = split["u"]
    state = PredictorState()
    preds = []
    for t in range(horizon):
        h_true_t = model.lift(torch.as_tensor(z_true[:, 1 + t, :], dtype=DTYPE))
        h_next, state = model.step(h_true_t, torch.as_tensor(u[:, 1 + t, :], dtype=DTYPE), state)
        preds.append(model.project(h_next))
    tf_pred_x = np.transpose(torch.stack(preds, dim=0).detach().numpy(), (1, 0, 2))
    target = z_true[:, 2 : 2 + horizon, :]
    tf_e_h = normalized_state_error(tf_pred_x, target, np.ones(D_Z))

    print(f"[oscillator_gold] teacher-forced e_h[-1]={tf_e_h[-1]:.3e} free-running e_h[-1]={result.e_h[-1]:.3e}")
    assert not np.allclose(tf_e_h, result.e_h, atol=1e-6)
    assert result.e_h[-1] > tf_e_h[-1] * 5, "free running must be the exponentially worse one here"


# ---------------------------------------------------------------------------
# Mutation campaign: mutate the reasoning THIS file relies on, in memory,
# and confirm the corresponding assertion reddens (REGLES section 7.1)
# ---------------------------------------------------------------------------
def test_mutant_van_loan_replaced_by_euler_would_fail_the_cross_check():
    """Mutant: ``discrete_reference`` implemented as first-order Euler
    (``A_d = I + delta*A``, ``B_d = delta*B``) instead of the exact Van Loan
    exponential. Shows the mutant diverges from the correct oracle at a
    level the Taylor cross-check (and hence the planted-model positive
    control) would catch."""
    A_x, B_x = _physical(M, K, K_C, C)
    A_correct, B_correct = discrete_reference(A_x, B_x, DELTA)
    A_euler = torch.eye(4, dtype=DTYPE) + DELTA * A_x
    B_euler = DELTA * B_x
    err_A = (A_correct - A_euler).abs().max().item()
    err_B = (B_correct - B_euler).abs().max().item()
    print(f"[mutant] Van Loan vs Euler: err_A={err_A:.3e} err_B={err_B:.3e}")
    assert err_A > 1e-3, "positive control: the Euler mutant must actually diverge from Van Loan"
    assert err_B > 5e-4, "positive control: the Euler mutant must actually diverge from Van Loan"


def test_mutant_relative_error_turned_absolute_would_silently_pass_a_small_magnitude_mismatch():
    """Mutant: ``eig_relative_error`` divides by nothing (returns the
    absolute error). At small eigenvalue magnitude, a fixed absolute
    deviation can be tiny in absolute terms while enormous relative to the
    reference -- exactly the regime a genuinely damped oscillator's slower
    mode can sit in. The mutant would (wrongly) pass a 1e-3 gate here; the
    real, relative function correctly fails it.
    """

    def _rot(a: float, w: float) -> Tensor:
        return torch.tensor([[a, -w], [w, a]], dtype=DTYPE)

    A_reference = _rot(-0.001, 0.002)
    A_learned = _rot(-0.0005, 0.0025)  # absolute deviation 5e-4 on each entry
    abs_err = (A_learned - A_reference).abs().max().item()
    rel_err = eig_relative_error(A_learned, A_reference)
    print(f"[mutant] absolute={abs_err:.3e} relative={rel_err:.3e}")
    assert abs_err < 1e-3, "positive control: the absolute-error mutant must (wrongly) pass"
    assert rel_err > 1e-3, "the real relative-error function must catch it"


# Mutation campaign item 3 ("pairing -> sort") is not re-tested here: it IS
# test_negative_sort_based_pairing_mispairs_conjugate_blocks_sharing_a_real_part
# above -- that test already mutates the pairing rule (brute force -> sort)
# in memory and shows the sorted version reddens where brute force does not.


def test_mutant_jacobian_replaced_by_frozen_block_basis_reading_would_hide_the_basis_confound():
    """Mutant: ``local_jacobians`` deleted, ``oscillator_bounds`` reads
    ``B_bar`` straight out of ``local_operator`` (this module's OWN earlier,
    repealed design). Reproduces the exact finding that forced the rewrite:
    a model that exactly reproduces the physics (this file's planted gold)
    would score O(1) on the mutant's action bound while scoring ~0 on the
    real, state-coordinate one -- the mutant is not merely imprecise, it
    would flag a perfectly correct model as a failure.
    """
    model = _gold_model(MLPKPredictor)
    x0, u0 = _inputs(batch=1, seed=72)
    x0v, u0v = x0[0], u0[0]

    h0 = model.lift(x0v.unsqueeze(0))
    A_bar, B_bar = model.local_operator(h0, u0v.unsqueeze(0), PredictorState())
    A_bar, B_bar = A_bar[0].detach(), B_bar[0].detach()

    A_x, B_x = _physical(M, K, K_C, C)
    A_ref, B_ref = discrete_reference(A_x, B_x, DELTA)

    mutant_action_err = action_relative_error(B_bar, B_ref)  # the repealed design
    real_action_err = action_relative_error(local_jacobians(model, x0v, u0v)[1], B_ref)
    print(f"[mutant] block-basis reading action_err={mutant_action_err:.3e} vs real J_u action_err={real_action_err:.3e}")
    assert mutant_action_err > 1e-3, "positive control: the repealed design must (wrongly) flag this exact model"
    assert real_action_err < 1e-9, "the real state-coordinate bound must correctly pass it"


def test_mutant_dispersion_not_computed_would_hide_context_dependence():
    """Mutant: ``oscillator_bounds`` reports the FIRST probe's error,
    repeated ``N`` times, instead of genuinely looping over probes -- its
    reported dispersion would always be exactly 0, regardless of whether the
    context actually varies with ``x``. Break the planted gold's neutral
    context (``W_theta != 0``, so the map genuinely varies across probes)
    and show the REAL dispersion this module computes is measurably
    nonzero, unlike what the repeated-first-probe mutant would report for
    the identical probe batch.
    """
    model = _gold_model(MLPKPredictor)
    g = torch.Generator().manual_seed(30)
    with torch.no_grad():
        model.head.weight.normal_(0.0, 0.5, generator=g)  # breaks W_theta = 0 neutrality
    system = _system()
    x0, u0 = _inputs(batch=6, seed=31)
    bounds = oscillator_bounds(model, system, delta=DELTA, probe_states=x0, probe_actions=u0)

    mutant_std = float(np.std([bounds["eig_relative_error_values"][0]] * len(bounds["eig_relative_error_values"])))
    print(
        f"[mutant] dispersion: real eig_std={bounds['eig_relative_error_std']:.3e} "
        f"repeated-first-probe mutant std={mutant_std:.3e}"
    )
    assert bounds["eig_relative_error_std"] > 0.0, "a non-neutral context must produce a nonzero field"
    assert mutant_std == 0.0
    assert bounds["eig_relative_error_std"] > mutant_std


# ---------------------------------------------------------------------------
# Direct unit coverage for oscillator_gold's own guards
# ---------------------------------------------------------------------------
def test_action_relative_error_refuses_a_zero_reference():
    with pytest.raises(ValueError, match="zero"):
        action_relative_error(torch.ones(2, 1, dtype=DTYPE), torch.zeros(2, 1, dtype=DTYPE))


def test_eig_relative_error_refuses_shape_mismatch():
    with pytest.raises(ValueError, match="mismatch"):
        eig_relative_error(torch.eye(4, dtype=DTYPE), torch.eye(2, dtype=DTYPE))


def test_oscillator_bounds_refuses_probe_count_mismatch():
    model = _gold_model(MLPKPredictor)
    system = _system()
    x0, u0 = _inputs(batch=3, seed=40)
    with pytest.raises(ValueError, match="leading"):
        oscillator_bounds(model, system, delta=DELTA, probe_states=x0, probe_actions=u0[:2])


def test_oscillator_bounds_refuses_a_float32_model():
    model = _build(MLPKPredictor).float()
    system = _system()
    x0 = torch.rand(2, D_Z, dtype=torch.float32)
    u0 = torch.rand(2, D_U, dtype=torch.float32)
    with pytest.raises(ValueError, match="float64"):
        oscillator_bounds(model, system, delta=DELTA, probe_states=x0, probe_actions=u0)
