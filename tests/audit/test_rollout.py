"""Tests for sld.audit.rollout.vpt (PROTOCOL.md section 12.2, REGLES.md
section 7.1's two-check rule): a hand-worked positive lock, plus four
negatives that each show a plausible-but-wrong convention disagreeing with
the locked value -- "would a wrong implementation fail this test?" (REGLES
7.1) is the criterion every test below is written against.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from sld.audit.rollout import vpt

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
