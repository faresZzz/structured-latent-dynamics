"""The representability distance: a hand reference, and what it must refuse.

PROTOCOL 26.0. The number this module produces is what explains why
`on-stable-nolift` is worse than both other arms, so it carries the two checks
REGLES 7.1 demands, and its positive reference is analytic rather than a second
run of the same code.
"""

from __future__ import annotations

import numpy as np
import pytest

from sld.systems.coupled_oscillators import CoupledOscillators
from sld.systems.representability import (
    nearest_block_rotation_scaling,
    real_modal_distance,
    representability_distance,
)


# --- positive: values known by hand -----------------------------------------


def test_an_exact_block_matrix_is_at_distance_zero():
    """The only case where the answer is knowable without any arithmetic."""
    A = np.array([[-0.3, -1.2, 0, 0], [1.2, -0.3, 0, 0], [0, 0, -0.7, -2.0], [0, 0, 2.0, -0.7]])
    dist, pairing = representability_distance(A)
    assert dist == pytest.approx(0.0, abs=1e-14)
    assert set(map(frozenset, pairing)) == {frozenset({0, 1}), frozenset({2, 3})}


def test_the_pairing_is_found_and_not_assumed():
    """The same matrix with rows/columns 1 and 2 swapped is still exactly
    representable -- under a DIFFERENT pairing. A function that only tried the
    adjacent pairing would report a large distance here."""
    A = np.array([[-0.3, -1.2, 0, 0], [1.2, -0.3, 0, 0], [0, 0, -0.7, -2.0], [0, 0, 2.0, -0.7]])
    perm = [0, 2, 1, 3]
    dist, pairing = representability_distance(A[np.ix_(perm, perm)])
    assert dist == pytest.approx(0.0, abs=1e-14)
    assert set(map(frozenset, pairing)) == {frozenset({0, 2}), frozenset({1, 3})}


def test_a_single_planted_offset_is_recovered_exactly():
    """Perturb one off-block entry by `eps`; the distance must be exactly
    `|eps|`, because that entry is the only thing the class cannot carry."""
    A = np.zeros((4, 4))
    A[0, 1], A[1, 0] = -1.0, 1.0
    A[2, 3], A[3, 2] = -2.0, 2.0
    A[0, 3] = 0.25
    dist, _ = representability_distance(A)
    assert dist == pytest.approx(0.25, abs=1e-12)


def test_the_closed_form_matches_a_brute_force_minimisation():
    """The independent reference: a coordinate descent over
    `(alpha_i, omega_i)` that shares no line of code with the closed form."""
    A = CoupledOscillators(c=0.10).A_mat
    closed, pairing = representability_distance(A)

    def build(p):
        M = np.zeros((4, 4))
        for b, (i, j) in enumerate(pairing):
            M[i, i] = M[j, j] = p[2 * b]
            M[i, j], M[j, i] = -p[2 * b + 1], p[2 * b + 1]
        return M

    p, best = np.zeros(4), np.linalg.norm(A - np.zeros((4, 4)))
    for step in (0.5, 0.1, 0.02, 5e-3, 1e-3, 1e-4, 1e-5):
        improved = True
        while improved:
            improved = False
            for k in range(4):
                for s in (+1, -1):
                    q = p.copy()
                    q[k] += s * step
                    d = np.linalg.norm(A - build(q))
                    if d < best - 1e-15:
                        best, p, improved = d, q, True
    assert best == pytest.approx(closed, rel=1e-6)


def test_the_oscillator_is_far_in_the_identity_basis_and_exact_in_the_best_one():
    """26.0's load-bearing pair of numbers, at the configuration campaign B
    calls `delta = 0.000`."""
    A = CoupledOscillators(c=0.10).A_mat
    identity_basis, _ = representability_distance(A)
    assert identity_basis == pytest.approx(1.0654, abs=5e-4)
    assert identity_basis / np.linalg.norm(A) == pytest.approx(0.360, abs=5e-3)
    assert real_modal_distance(A) < 1e-12


def test_the_two_counterexamples_26_0_names():
    """Checked against the matrix the SYSTEM builds, not one rebuilt here --
    the object under test must be the one the campaigns actually trained on."""
    A = CoupledOscillators(c=0.10).A_mat
    assert (A[0, 2], A[2, 0]) == (1.0, -1.5), "pairing (1,2)(3,4) needs both zero"
    assert (A[0, 0], A[2, 2]) == (0.0, -0.10), "pairing (1,3)(2,4) needs a11 == a22"


# --- negative ---------------------------------------------------------------


def test_no_pairing_makes_the_oscillator_exactly_representable():
    """The claim 26.0 makes. Asserted over ALL pairings, not just the minimum,
    so a bug that returned the minimum of an incomplete list would show."""
    A = CoupledOscillators(c=0.10).A_mat
    for pairing in (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))):
        M = nearest_block_rotation_scaling(A, pairing)
        assert np.linalg.norm(A - M) > 1.0, f"{pairing} would be nearly exact"


def test_a_real_spectrum_has_no_exact_basis_either():
    """At c = 3.00 two eigenvalues are real, so no change of basis achieves the
    block form -- `real_modal_distance` must NOT return ~0 there. A version
    that ran the complex path unconditionally would."""
    A = CoupledOscillators(c=3.00).A_mat
    assert np.any(np.abs(np.linalg.eigvals(A).imag) < 1e-12)
    assert real_modal_distance(A) > 0.1


@pytest.mark.parametrize("shape", [(3, 3), (5, 5)])
def test_an_odd_order_is_refused(shape):
    """2x2 blocks cannot tile an odd dimension. Dropping a coordinate to make
    it fit would silently answer a different question."""
    with pytest.raises(ValueError, match="even"):
        representability_distance(np.zeros(shape))


def test_a_non_square_matrix_is_refused():
    with pytest.raises(ValueError, match="square"):
        representability_distance(np.zeros((2, 4)))
