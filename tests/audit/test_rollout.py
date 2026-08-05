"""Tests for sld.audit.rollout (PROTOCOL.md sections 12.1-12.3, REGLES.md
section 7.1's two-check rule): hand-worked positive locks, plus negatives
that each show a plausible-but-wrong convention disagreeing with the locked
value -- "would a wrong implementation fail this test?" (REGLES 7.1) is the
criterion every test below is written against.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from sld.audit.rollout import (
    evaluate_positive_control,
    fit_train_state_scale,
    linear_extrapolation_rollout,
    normalized_state_error,
    persistence_rollout,
    vpt,
)

# ---------------------------------------------------------------------------
# positives -- all hand-worked, no reference computed by the function itself
# ---------------------------------------------------------------------------


def test_vpt_hand_worked_example_locked_at_2():
    """PROTOCOL.md section 12.2's own locked example. Validity is STRICT
    '<', so e_3 == tau (0.4 == 0.4) is NOT a valid step: the maximal valid
    prefix is (e_1, e_2) only, hence VPT = 2, never 3."""
    e = np.array([0.0, 0.2, 0.4, 0.6])
    result = vpt(e, tau=0.4)
    assert result.value == 2.0
    assert result.status == "crossed"
    assert result.horizon == 4


def test_vpt_perfect_predictor_is_censored_not_a_finite_count():
    """Zero error at every step never reaches tau: the crossing time was
    never observed on this horizon, so the outcome is right-censoring, not
    a finite VPT of "10" or "11"."""
    e = np.zeros(10)
    result = vpt(e, tau=0.5)
    assert result.status == "censored"
    assert math.isinf(result.value) and result.value > 0
    assert result.horizon == 10


def test_vpt_is_zero_when_first_step_already_invalid():
    """e_1 >= tau: zero valid steps. A "first-crossing index" convention
    would report 1 here (see the negative test below) -- this is the
    convention this project rejects."""
    e = np.array([0.6, 0.1, 0.05, 0.02])
    result = vpt(e, tau=0.4)
    assert result.value == 0.0
    assert result.status == "crossed"  # a crossing WAS observed -- at step 1


def test_vpt_crossing_at_a_known_later_step():
    """Hand-built curve valid for steps 1..5 (five values < tau), invalid
    at step 6: the maximal valid prefix has length 5."""
    e = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.55, 0.6, 0.65])
    result = vpt(e, tau=0.5)
    assert result.value == 5.0
    assert result.status == "crossed"
    assert result.horizon == 8


# ---------------------------------------------------------------------------
# negatives -- each must FAIL under the wrong convention it targets
# ---------------------------------------------------------------------------


def test_negative_first_crossing_index_convention_disagrees_at_zero():
    """The wrong-but-plausible "index of first crossing" convention
    (1-based ``min{h : e_h > tau}``) reports 1 on this curve; the true VPT
    is 0. Showing the two numbers differ is the point of the negative
    check -- a test unable to distinguish them would not be testing the
    convention at all."""
    e = np.array([0.6, 0.1, 0.05])
    tau = 0.4
    first_crossing_index = next(h for h, val in enumerate(e, start=1) if val > tau)
    assert first_crossing_index == 1
    assert vpt(e, tau).value == 0.0
    assert first_crossing_index != vpt(e, tau).value


def test_negative_off_by_one_index_shift_fails_the_main_lock():
    """A one-step index shift -- treating e_h == tau as still valid, i.e.
    using strict '>' instead of the contract's '>=' for invalidity --
    changes the locked example's answer from 2 to 3. This is exactly the
    failure mode test_vpt_hand_worked_example_locked_at_2 exists to catch:
    an off-by-one implementation would disagree with the locked value."""
    e = np.array([0.0, 0.2, 0.4, 0.6])
    tau = 0.4
    off_by_one_invalid = e > tau  # WRONG convention: should be >=
    shifted = int(np.argmax(off_by_one_invalid)) if off_by_one_invalid.any() else None
    assert shifted == 3
    assert shifted != vpt(e, tau).value
    assert vpt(e, tau).value == 2.0


def test_negative_non_crossing_is_never_H_or_H_plus_1_in_disguise():
    """No exception, no NaN, and -- the substitution this contract forbids
    -- no finite stand-in equal to H or H+1 for a censored outcome."""
    H = 7
    e = np.full(H, 0.01)  # never reaches tau=0.5
    result = vpt(e, tau=0.5)
    assert result.status == "censored"
    assert result.value != H
    assert result.value != H + 1
    assert math.isinf(result.value)


def test_negative_nan_in_curve_raises_not_a_silent_number():
    """NaN in an error curve is an upstream bug, never a censored or
    finite audit outcome -- it must raise, not propagate."""
    e = np.array([0.1, 0.2, float("nan"), 0.6])
    with pytest.raises(ValueError, match="non-finite"):
        vpt(e, tau=0.4)


# ---------------------------------------------------------------------------
# batching: one value per row, never an aggregate
# ---------------------------------------------------------------------------


def test_batched_input_returns_one_value_per_row_not_aggregated():
    e = np.array(
        [
            [0.0, 0.2, 0.4, 0.6],  # -> 2, crossed
            [0.6, 0.1, 0.05, 0.02],  # -> 0, crossed
            [0.0, 0.0, 0.0, 0.0],  # -> censored
        ]
    )
    result = vpt(e, tau=0.4)
    assert result.value.shape == (3,)
    assert result.status.shape == (3,)
    assert list(result.value) == [2.0, 0.0, np.inf]
    assert list(result.status) == ["crossed", "crossed", "censored"]
    assert result.horizon == 4  # one shared horizon, not per-row


def test_single_curve_input_yields_0d_arrays():
    """The signature choice (parallel arrays sharing e's leading shape):
    a lone (H,) curve reduces to 0-d arrays, not a length-1 batch."""
    result = vpt(np.array([0.0, 0.2, 0.4, 0.6]), tau=0.4)
    assert result.value.shape == ()
    assert result.status.shape == ()


def test_accepts_torch_tensor_input():
    e = torch.tensor([0.0, 0.2, 0.4, 0.6], dtype=torch.float32)
    result = vpt(e, tau=0.4)
    assert result.value == 2.0


# ---------------------------------------------------------------------------
# boundary validation
# ---------------------------------------------------------------------------


def test_tau_negative_raises():
    with pytest.raises(ValueError, match="tau"):
        vpt(np.array([0.1, 0.2]), tau=-0.1)


def test_tau_non_finite_raises():
    with pytest.raises(ValueError, match="tau"):
        vpt(np.array([0.1, 0.2]), tau=float("inf"))
    with pytest.raises(ValueError, match="tau"):
        vpt(np.array([0.1, 0.2]), tau=float("nan"))


def test_empty_horizon_raises():
    with pytest.raises(ValueError, match="horizon"):
        vpt(np.zeros((3, 0)), tau=0.5)


def test_scalar_input_raises_no_horizon_axis():
    with pytest.raises(ValueError, match="0-d"):
        vpt(np.array(0.3), tau=0.4)


# ---------------------------------------------------------------------------
# fit_train_state_scale / normalized_state_error (PROTOCOL.md section 12.1)
# ---------------------------------------------------------------------------


def test_normalized_state_error_perfect_prediction_is_exactly_zero():
    """pred == target at every step: e_h must be 0 at every h, not merely
    small -- this is the reference this quantity is built to satisfy."""
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(50, 3)) * np.array([1.0, 5.0, 0.2])
    sigma = fit_train_state_scale(x_train)

    target = rng.normal(size=(4, 6, 3))
    pred = target.copy()
    e = normalized_state_error(pred, target, sigma)
    assert e.shape == (6,)
    np.testing.assert_allclose(e, 0.0, atol=1e-12)


def test_normalized_state_error_unit_invariance():
    """Multiplying state, prediction AND sigma_train coordinate-wise by an
    arbitrary positive unit vector must leave e_h (and the VPT computed
    from it) unchanged -- this is the check that per-coordinate
    normalization, not a global RMS, is doing the work. Deliberately
    adversarial scale: 1e-8 vs 1e6."""
    rng = np.random.default_rng(1)
    x_train = rng.normal(size=(200, 2)) + np.array([0.0, 10.0])
    sigma = fit_train_state_scale(x_train)

    target = rng.normal(size=(5, 8, 2))
    pred = target + rng.normal(scale=0.3, size=target.shape)
    e_ref = normalized_state_error(pred, target, sigma)
    vpt_ref = vpt(e_ref, tau=0.4)

    unit = np.array([1e-8, 1e6])
    e_scaled = normalized_state_error(pred * unit, target * unit, sigma * unit)
    vpt_scaled = vpt(e_scaled, tau=0.4)

    np.testing.assert_allclose(e_scaled, e_ref, atol=1e-12, rtol=1e-9)
    assert vpt_scaled.value == vpt_ref.value
    assert vpt_scaled.status == vpt_ref.status


def test_fit_train_state_scale_zero_variance_coordinate_is_refused():
    """A constant coordinate (sigma_i == 0) is a REFUSAL naming the
    offending index, never a silent epsilon substitute."""
    x_train = np.stack(
        [np.array([1.0, 2.0, 3.0]), np.full(3, 7.0)], axis=1
    )  # column 1 is constant
    with pytest.raises(ValueError, match=r"coordinate\(s\) \[1\]"):
        fit_train_state_scale(x_train)


def test_normalized_state_error_zero_sigma_is_refused_not_epsilon():
    """Even a sigma that reaches normalized_state_error already zero (e.g.
    hand-built rather than produced by fit_train_state_scale) is refused,
    never divided-through with an epsilon."""
    pred = np.zeros((2, 3, 2))
    target = np.zeros((2, 3, 2))
    sigma = np.array([1.0, 0.0])
    with pytest.raises(ValueError, match="zero entries"):
        normalized_state_error(pred, target, sigma)


def test_normalized_state_error_nonfinite_pred_is_refused():
    pred = np.zeros((1, 2, 2))
    pred[0, 0, 0] = float("nan")
    target = np.zeros((1, 2, 2))
    sigma = np.array([1.0, 1.0])
    with pytest.raises(ValueError, match="pred contains non-finite"):
        normalized_state_error(pred, target, sigma)


def test_normalized_state_error_nonfinite_target_is_refused():
    pred = np.zeros((1, 2, 2))
    target = np.zeros((1, 2, 2))
    target[0, 1, 1] = float("inf")
    sigma = np.array([1.0, 1.0])
    with pytest.raises(ValueError, match="target contains non-finite"):
        normalized_state_error(pred, target, sigma)


def test_normalized_state_error_nonfinite_sigma_is_refused():
    pred = np.zeros((1, 2, 2))
    target = np.zeros((1, 2, 2))
    sigma = np.array([1.0, float("nan")])
    with pytest.raises(ValueError, match="sigma_train contains non-finite"):
        normalized_state_error(pred, target, sigma)


def test_normalized_state_error_shape_mismatch_is_refused():
    pred = np.zeros((1, 2, 2))
    target = np.zeros((1, 2, 3))
    sigma = np.array([1.0, 1.0])
    with pytest.raises(ValueError, match="shapes disagree"):
        normalized_state_error(pred, target, sigma)


def test_normalized_state_error_is_per_step_not_aggregated_over_horizon():
    """e_h must vary independently per horizon step -- a mutant that
    aggregates the RMS over h as well as over (b, i) (i.e. sqrt(mean over
    ALL of b, h, i) broadcast back to shape (H,)) would pass every other
    test in this file (they either use H=1 or only compare two curves
    computed the SAME wrong way against each other) but must fail here,
    where h=0 and h=2 are hand-built to be exactly zero while h=1 is not."""
    pred = np.zeros((1, 3, 1))
    target = np.zeros((1, 3, 1))
    pred[0, 1, 0] = 1.0  # only the middle step (h=2, index 1) has any error
    sigma = np.array([1.0])
    e = normalized_state_error(pred, target, sigma)
    np.testing.assert_allclose(e, [0.0, 1.0, 0.0], atol=1e-12)


def test_normalized_state_error_hand_worked_value():
    """A single episode, one horizon step, hand-computed by arithmetic
    outside the function under test -- an independent reference (REGLES
    section 7.1), not the function validated against its own output."""
    pred = np.array([[[1.0, 0.0]]])  # (B=1, H=1, d=2)
    target = np.array([[[0.0, 2.0]]])
    sigma = np.array([2.0, 4.0])
    # ((1-0)/2)^2 + ((0-2)/4)^2 = 0.25 + 0.25 = 0.5; /(B*d)=/2 -> 0.25; sqrt -> 0.5
    e = normalized_state_error(pred, target, sigma)
    assert e.shape == (1,)
    np.testing.assert_allclose(e, [0.5], atol=1e-12)


# ---------------------------------------------------------------------------
# persistence_rollout / linear_extrapolation_rollout (PROTOCOL.md section 12.3)
# ---------------------------------------------------------------------------


def test_persistence_rollout_is_bit_identical_copies():
    x0 = np.array([[2.0, -1.0]])
    out = persistence_rollout(x0, horizon=3)
    assert out.shape == (1, 3, 2)
    expected = np.array([[[2.0, -1.0], [2.0, -1.0], [2.0, -1.0]]])
    np.testing.assert_array_equal(out, expected)  # bit-identical: atol=rtol=0


def test_linear_extrapolation_rollout_reproduces_a_known_linear_trajectory():
    """A trajectory built exactly as x_h = x0 + h*(x0 - x_minus_1): the
    baseline must reproduce it with error 0 (up to float64 rounding)."""
    x_minus_1 = np.array([[1.0, 5.0], [0.0, 0.0]])
    x0 = np.array([[1.5, 4.0], [1.0, -1.0]])
    horizon = 4
    velocity = x0 - x_minus_1
    hs = np.arange(1, horizon + 1)
    target = x0[:, None, :] + hs[None, :, None] * velocity[:, None, :]

    out = linear_extrapolation_rollout(x_minus_1, x0, horizon)
    np.testing.assert_allclose(out, target, atol=1e-12)


def test_linear_extrapolation_rollout_index_convention_first_step_is_not_x0():
    """The trap: the first returned step (index 0) is x0 + 1*(x0 -
    x_minus_1), NOT x0. A caller assuming output[:, 0] == x0 has the wrong
    convention -- this test fails under that wrong convention."""
    x_minus_1 = np.array([[0.0, 0.0]])
    x0 = np.array([[1.0, 2.0]])
    out = linear_extrapolation_rollout(x_minus_1, x0, horizon=3)
    expected_first_step = np.array([[2.0, 4.0]])  # x0 + 1*(x0 - x_minus_1)
    np.testing.assert_allclose(out[:, 0, :], expected_first_step, atol=1e-12)
    assert not np.allclose(out[:, 0, :], x0)  # NOT x0


def test_linear_extrapolation_rollout_off_by_one_swap_gives_different_curve():
    """Swapping x_minus_1 and x0 must produce a genuinely different curve
    -- catches the off-by-one/argument-order mutant."""
    x_minus_1 = np.array([[0.0, 0.0]])
    x0 = np.array([[1.0, 2.0]])
    correct = linear_extrapolation_rollout(x_minus_1, x0, horizon=3)
    swapped = linear_extrapolation_rollout(x0, x_minus_1, horizon=3)
    assert not np.allclose(correct, swapped)


# ---------------------------------------------------------------------------
# evaluate_positive_control (PROTOCOL.md section 12.3)
# ---------------------------------------------------------------------------


def _curve(h_gate: int, e1: float, e_hgate: float) -> np.ndarray:
    """Hand-built monotone e_h curve of length h_gate with e[0]=e1 and
    e[h_gate-1]=e_hgate, linearly interpolated in between (the interior
    shape never matters to the gate, only the two indexed values do)."""
    return np.linspace(e1, e_hgate, h_gate)


def test_evaluate_positive_control_both_bounds_true_is_learns_with_ratios_below_one():
    h_gate = 25
    gamma = 5.0
    tau_1 = 0.3
    model = _curve(h_gate, e1=0.05, e_hgate=0.2)  # e_25 = 0.2
    pers = _curve(h_gate, e1=0.13, e_hgate=2.0)  # min(pers,lin)/gamma = 2.0/5 = 0.4 > 0.2
    lin = _curve(h_gate, e1=0.13, e_hgate=2.5)
    result = evaluate_positive_control(
        model, pers, lin, h_gate=h_gate, gamma=gamma, tau_1=tau_1
    )
    assert result.verdict == "learns"
    assert result.lower_bound is True
    assert result.upper_bound is True
    # 12.4's direction: < 1 is a PASS (m_low = 0.2*5/2.0 = 0.5, m_high = 0.05/0.3 = 1/6).
    assert result.m_low == pytest.approx(0.5)
    assert result.m_high == pytest.approx(1.0 / 6.0)
    assert result.m_low < 1.0
    assert result.m_high < 1.0
    assert result.near_miss is False


def test_evaluate_positive_control_only_lower_bound_true_does_not_learn():
    """Lower bound passes (e_H comfortably below the baseline floor) but
    e_1 >= tau_1: the verdict must still be does_not_learn -- both bounds
    are required."""
    h_gate = 25
    model = _curve(h_gate, e1=0.5, e_hgate=0.1)  # e_1 = 0.5 fails tau_1
    pers = _curve(h_gate, e1=0.13, e_hgate=2.0)
    lin = _curve(h_gate, e1=0.13, e_hgate=2.5)
    result = evaluate_positive_control(model, pers, lin, h_gate=h_gate, gamma=5.0, tau_1=0.3)
    assert result.lower_bound is True
    assert result.upper_bound is False
    assert result.verdict == "does_not_learn"


def test_evaluate_positive_control_only_upper_bound_true_does_not_learn():
    """e_1 passes tau_1 but e_H fails the baseline-floor comparison: still
    does_not_learn."""
    h_gate = 25
    model = _curve(h_gate, e1=0.05, e_hgate=1.0)  # e_25 = 1.0 fails floor 0.4
    pers = _curve(h_gate, e1=0.13, e_hgate=2.0)
    lin = _curve(h_gate, e1=0.13, e_hgate=2.5)
    result = evaluate_positive_control(model, pers, lin, h_gate=h_gate, gamma=5.0, tau_1=0.3)
    assert result.lower_bound is False
    assert result.upper_bound is True
    assert result.verdict == "does_not_learn"


def test_evaluate_positive_control_exact_equality_at_upper_bound_is_strict():
    """e_1 == tau_1 exactly must give m_high == 1.0 exactly and
    upper_bound == False (strict '<', never '<=')."""
    h_gate = 25
    tau_1 = 0.3
    model = _curve(h_gate, e1=tau_1, e_hgate=0.05)
    pers = _curve(h_gate, e1=0.13, e_hgate=2.0)
    lin = _curve(h_gate, e1=0.13, e_hgate=2.5)
    result = evaluate_positive_control(model, pers, lin, h_gate=h_gate, gamma=5.0, tau_1=tau_1)
    assert result.m_high == 1.0
    assert result.upper_bound is False
    assert result.verdict == "does_not_learn"


def test_evaluate_positive_control_exact_equality_at_lower_bound_is_strict():
    """e_H exactly equal to the baseline floor must give m_low == 1.0
    exactly and lower_bound == False (strict '<', never '<=')."""
    h_gate = 25
    gamma = 5.0
    floor = 2.0 / gamma  # 0.4
    model = _curve(h_gate, e1=0.05, e_hgate=floor)
    pers = _curve(h_gate, e1=0.13, e_hgate=2.0)
    lin = _curve(h_gate, e1=0.13, e_hgate=2.5)
    result = evaluate_positive_control(model, pers, lin, h_gate=h_gate, gamma=gamma, tau_1=0.3)
    assert result.m_low == 1.0
    assert result.lower_bound is False
    assert result.verdict == "does_not_learn"


def test_evaluate_positive_control_near_miss_quantifier_is_max_of_failing_bounds_only():
    """PROTOCOL.md section 14.4: near_miss requires the MAXIMUM of the
    ratios of the bounds that actually failed to lie in (1, 1.5] -- not
    'any bound in range'. Here the lower bound misses by 1.1x (m_low =
    0.44*5/2.0 = 1.1, inside (1, 1.5] on its own) but the upper bound
    misses by 5x (m_high = 1.5/0.3 = 5.0, well outside). This is exactly
    the case the original (wrong) 'either ratio' spec would have
    mislabelled as a near miss; the corrected 'max over failing bounds'
    rule must not."""
    h_gate = 25
    gamma = 5.0
    tau_1 = 0.3
    model = _curve(h_gate, e1=1.5, e_hgate=0.44)
    pers = _curve(h_gate, e1=0.13, e_hgate=2.0)
    lin = _curve(h_gate, e1=0.13, e_hgate=2.5)
    result = evaluate_positive_control(
        model, pers, lin, h_gate=h_gate, gamma=gamma, tau_1=tau_1
    )
    assert result.lower_bound is False
    assert result.upper_bound is False
    assert result.m_low == pytest.approx(1.1)
    assert result.m_high == pytest.approx(5.0)
    assert result.verdict == "does_not_learn"
    assert result.near_miss is False


def test_evaluate_positive_control_short_curve_is_refused_not_truncated():
    """A curve shorter than h_gate is a REFUSAL, never silently reduced to
    min(H, h_gate)."""
    h_gate = 25
    model = _curve(20, e1=0.05, e_hgate=0.1)  # only 20 steps
    pers = _curve(h_gate, e1=0.13, e_hgate=2.0)
    lin = _curve(h_gate, e1=0.13, e_hgate=2.5)
    with pytest.raises(ValueError, match="shorter than"):
        evaluate_positive_control(model, pers, lin, h_gate=h_gate, gamma=5.0, tau_1=0.3)


def test_evaluate_positive_control_nonfinite_curve_is_refused():
    h_gate = 5
    model = _curve(h_gate, e1=0.05, e_hgate=0.1)
    model[2] = float("nan")
    pers = _curve(h_gate, e1=0.13, e_hgate=2.0)
    lin = _curve(h_gate, e1=0.13, e_hgate=2.5)
    with pytest.raises(ValueError, match="non-finite"):
        evaluate_positive_control(model, pers, lin, h_gate=h_gate, gamma=5.0, tau_1=0.3)


# ---------------------------------------------------------------------------
# evaluate_positive_control: denominator guards (gamma, tau_1, h_gate, baseline)
# ---------------------------------------------------------------------------


def test_evaluate_positive_control_gamma_zero_is_refused():
    h_gate = 5
    model = _curve(h_gate, e1=0.05, e_hgate=0.1)
    pers = _curve(h_gate, e1=0.13, e_hgate=2.0)
    lin = _curve(h_gate, e1=0.13, e_hgate=2.5)
    with pytest.raises(ValueError, match="gamma"):
        evaluate_positive_control(model, pers, lin, h_gate=h_gate, gamma=0.0, tau_1=0.3)


def test_evaluate_positive_control_gamma_nonfinite_is_refused():
    h_gate = 5
    model = _curve(h_gate, e1=0.05, e_hgate=0.1)
    pers = _curve(h_gate, e1=0.13, e_hgate=2.0)
    lin = _curve(h_gate, e1=0.13, e_hgate=2.5)
    with pytest.raises(ValueError, match="gamma"):
        evaluate_positive_control(
            model, pers, lin, h_gate=h_gate, gamma=float("nan"), tau_1=0.3
        )


def test_evaluate_positive_control_tau_1_zero_is_refused():
    h_gate = 5
    model = _curve(h_gate, e1=0.05, e_hgate=0.1)
    pers = _curve(h_gate, e1=0.13, e_hgate=2.0)
    lin = _curve(h_gate, e1=0.13, e_hgate=2.5)
    with pytest.raises(ValueError, match="tau_1"):
        evaluate_positive_control(model, pers, lin, h_gate=h_gate, gamma=5.0, tau_1=0.0)


def test_evaluate_positive_control_h_gate_not_positive_integer_is_refused():
    model = _curve(5, e1=0.05, e_hgate=0.1)
    pers = _curve(5, e1=0.13, e_hgate=2.0)
    lin = _curve(5, e1=0.13, e_hgate=2.5)
    with pytest.raises(ValueError, match="h_gate"):
        evaluate_positive_control(model, pers, lin, h_gate=0, gamma=5.0, tau_1=0.3)


def test_evaluate_positive_control_baseline_zero_is_refused_not_infinite():
    """A baseline of exactly zero at h_gate (both trivial baselines predict
    perfectly) is refused with a distinct message naming the baseline --
    never a silent inf ratio."""
    h_gate = 5
    model = _curve(h_gate, e1=0.05, e_hgate=0.1)
    pers = _curve(h_gate, e1=0.13, e_hgate=0.0)
    lin = _curve(h_gate, e1=0.13, e_hgate=0.0)
    with pytest.raises(ValueError, match="baseline"):
        evaluate_positive_control(model, pers, lin, h_gate=h_gate, gamma=5.0, tau_1=0.3)


def test_evaluate_positive_control_perfect_model_is_not_refused_by_denominator_guards():
    """The guard constrains DENOMINATORS only: a perfect model (e_H == 0,
    e_1 == 0) with healthy baselines must pass cleanly, no exception,
    m_low == m_high == 0.0, verdict == 'learns'. Refusing this would be the
    ironic failure mode -- rejecting the exact result the gate rewards."""
    h_gate = 5
    model = _curve(h_gate, e1=0.0, e_hgate=0.0)
    pers = _curve(h_gate, e1=0.13, e_hgate=2.0)
    lin = _curve(h_gate, e1=0.13, e_hgate=2.5)
    result = evaluate_positive_control(model, pers, lin, h_gate=h_gate, gamma=5.0, tau_1=0.3)
    assert result.verdict == "learns"
    assert result.m_low == 0.0
    assert result.m_high == 0.0
    assert result.lower_bound is True
    assert result.upper_bound is True
