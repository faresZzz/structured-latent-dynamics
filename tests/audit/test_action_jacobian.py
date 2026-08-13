"""The action Jacobian: a closed form it must reproduce, and the ways it must refuse.

REGLES 7.1. The positive check is an LTI cell whose Jacobian is known by hand,
so the reference is analytic and not a second run of the same code. The
negative checks are the ones that matter: a Jacobian that disagrees with
centred finite differences must RAISE rather than be returned, and a
rank-deficient action map must report `inf` rather than a regularised ratio.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from sld.audit.action_jacobian import (
    check_against_finite_differences,
    conditioning,
    finite_difference_action_jacobian,
    jacobian_action,
)
from sld.predictors.base import PredictorModule, PredictorState


class LTI(PredictorModule):
    """``h <- A h + B u``. Its endpoint Jacobian is known in closed form:
    column ``t`` of ``d x_H / d u`` is ``A**(H-1-t) B`` -- derived by hand from
    ``x_H = A**H h_0 + sum_t A**(H-1-t) B u_t``, not by running this class."""

    def __init__(self, A: torch.Tensor, B: torch.Tensor):
        super().__init__()
        self.A = nn.Parameter(A.clone(), requires_grad=False)
        self.B = nn.Parameter(B.clone(), requires_grad=False)

    def step(self, h, u, state: PredictorState):
        return h @ self.A.T + u @ self.B.T, state


def _lti(d_h: int = 3, d_u: int = 2, seed: int = 0) -> LTI:
    g = torch.Generator().manual_seed(seed)
    A = 0.6 * torch.randn(d_h, d_h, generator=g, dtype=torch.float64)
    B = torch.randn(d_h, d_u, generator=g, dtype=torch.float64)
    return LTI(A, B).double()


def _closed_form(model: LTI, horizon: int) -> torch.Tensor:
    """``[A**(H-1-t) B]`` laid out to match the flattened ``(T, d_u)`` order."""
    d_h, d_u = model.B.shape
    jac = torch.zeros(d_h, horizon * d_u, dtype=torch.float64)
    for t in range(horizon):
        jac[:, t * d_u : (t + 1) * d_u] = torch.linalg.matrix_power(model.A, horizon - 1 - t) @ model.B
    return jac


# --- positive: the value, against a hand reference ---------------------------


@pytest.mark.parametrize("horizon", (1, 5, 25))
def test_matches_the_closed_form_on_an_lti_cell(horizon):
    model = _lti()
    h0 = torch.randn(3, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    u = torch.randn(horizon, 2, generator=torch.Generator().manual_seed(2), dtype=torch.float64)

    jac = jacobian_action(model, h0, u)

    assert jac.shape == (3, horizon * 2)
    torch.testing.assert_close(jac, _closed_form(model, horizon), rtol=1e-12, atol=1e-12)


def test_the_jacobian_does_not_depend_on_h0_for_an_lti_cell():
    """A property of the closed form, checked rather than assumed: ``x_H``'s
    dependence on ``u`` is affine, so ``d x_H / d u`` is the same at any h0."""
    model = _lti()
    u = torch.randn(7, 2, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    a = jacobian_action(model, torch.zeros(3, dtype=torch.float64), u)
    b = jacobian_action(model, torch.full((3,), 4.0, dtype=torch.float64), u)
    torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


def test_centred_differences_reproduce_autograd_to_better_than_1e_8():
    """The independent-reference corollary: no shared code path between the
    two, so agreement is evidence rather than a tautology."""
    model = _lti()
    h0 = torch.randn(3, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    u = torch.randn(25, 2, generator=torch.Generator().manual_seed(5), dtype=torch.float64)

    rel = check_against_finite_differences(model, h0, u, jacobian_action(model, h0, u))

    assert rel < 1e-8, f"max relative error {rel:.3e}"


def test_conditioning_runs_the_check_itself():
    """A figure cannot get an unchecked Jacobian by forgetting to ask."""
    model = _lti()
    h0 = torch.randn(3, generator=torch.Generator().manual_seed(6), dtype=torch.float64)
    u = torch.randn(25, 2, generator=torch.Generator().manual_seed(7), dtype=torch.float64)

    result = conditioning(model, h0, u)

    assert result.fd_max_rel_error < 1e-8
    expected = torch.linalg.svdvals(_closed_form(model, 25))
    torch.testing.assert_close(result.singular_values, expected, rtol=1e-10, atol=1e-10)
    assert math.isclose(result.cond, (expected[0] / expected[-1]).item(), rel_tol=1e-10)


# --- negative: it must refuse, and it must not smooth anything over ----------


def test_a_wrong_jacobian_is_refused_not_returned():
    """The check that gives the positive one its meaning. A Jacobian perturbed
    by 1e-6 -- far too small to see on a plot, far larger than 1e-8 -- must
    raise, naming the measured disagreement."""
    model = _lti()
    h0 = torch.randn(3, generator=torch.Generator().manual_seed(8), dtype=torch.float64)
    u = torch.randn(10, 2, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    wrong = jacobian_action(model, h0, u).clone()
    wrong[0, 0] += 1e-6

    with pytest.raises(ValueError, match="finite differences"):
        check_against_finite_differences(model, h0, u, wrong)


def test_a_rank_deficient_action_map_reports_inf_not_a_regularised_ratio():
    """An UNREACHABLE latent direction must give `cond = inf`, not a large
    finite number that reads on a figure as "hard but possible".

    The construction has to be earned: zeroing a column of ``B`` is NOT enough,
    because the reachable subspace is span{A**k B} and a generic ``A`` mixes a
    single excited coordinate back into all the others within ``d_h`` steps --
    the first version of this test asserted a zero singular value and measured
    0.469. Decoupling is what makes a direction unreachable, so ``A`` is block
    diagonal here and ``B`` excites only the first block. Coordinate 2 then
    depends on ``h0`` alone, its row of the Jacobian is exactly zero, and
    ``s_min`` is exactly zero for a structural reason rather than by rounding.
    """
    g = torch.Generator().manual_seed(10)
    A = torch.tensor([[0.5, 0.1, 0.0], [0.2, 0.4, 0.0], [0.0, 0.0, 0.3]], dtype=torch.float64)
    B = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=torch.float64)
    model = LTI(A, B).double()
    h0 = torch.zeros(3, dtype=torch.float64)
    u = torch.randn(4, 2, generator=g, dtype=torch.float64)

    result = conditioning(model, h0, u)

    assert torch.count_nonzero(result.jacobian[2]) == 0, "coordinate 2 must be unreachable"
    assert result.singular_values[-1].item() == pytest.approx(0.0, abs=1e-14)
    assert math.isinf(result.cond)


def test_float32_input_is_refused():
    """REGLES section 5: the audit runs in float64. A float32 predictor would
    give a finite-difference error near 1e-4 and the 1e-8 gate would fire
    downstream -- refusing here names the cause instead."""
    model = _lti().float()
    h0 = torch.zeros(3, dtype=torch.float64)
    u = torch.zeros(4, 2, dtype=torch.float64)
    with pytest.raises(ValueError):
        jacobian_action(model, h0, u)


@pytest.mark.parametrize(
    ("h0", "u", "match"),
    [
        (torch.zeros(1, 3, dtype=torch.float64), torch.zeros(4, 2, dtype=torch.float64), "h0"),
        (torch.zeros(3, dtype=torch.float64), torch.zeros(4, dtype=torch.float64), "u_seq"),
    ],
)
def test_wrong_shapes_are_refused_by_name(h0, u, match):
    with pytest.raises(ValueError, match=match):
        jacobian_action(_lti(), h0, u)


def test_the_finite_difference_path_alone_also_matches_the_closed_form():
    """Guards the reference itself: if `_rollout_endpoint` were wrong, autograd
    and finite differences would agree with each other and both be wrong. The
    closed form catches that, so it is asserted against the FD path too."""
    model = _lti()
    u = torch.randn(6, 2, generator=torch.Generator().manual_seed(11), dtype=torch.float64)
    fd = finite_difference_action_jacobian(model, torch.zeros(3, dtype=torch.float64), u)
    torch.testing.assert_close(fd, _closed_form(model, 6), rtol=1e-7, atol=1e-9)
