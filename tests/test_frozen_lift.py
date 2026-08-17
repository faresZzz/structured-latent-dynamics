"""PROTOCOL 28.2's oracle arm: `lift="frozen"`, psi = T and pi = T^-1 fixed.

The contract this file pins, stated once so the assertions below read as
consequences rather than as choices:

- `T` is a CHANGE OF BASIS, so `pi . psi = Id` exactly and `psi` is linear
  (no bias). An affine `psi` would make `T A T^-1` stop meaning what 26.0
  measured.
- `T` and `T^-1` carry ZERO TRAINABLE parameters. That is the whole point of
  28.2's suspended budget matching: the oracle arm must differ from
  `on-learned` by the learned lift pair and by NOTHING else.
- A basis that cannot be inverted is refused, not repaired. A silently
  singular `T` would give the arm a misspecification while the section claims
  it has none -- the exact failure 26.0 measured in the identity basis.
- The basis actually delivers what 26.0 published: `T A T^-1` is block-diagonal
  rotation-scaling with blocks `(-0.05, +-1.5187)` and `(-0.05, +-1.0902)`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sld.predictors.base import PredictorState
from sld.predictors.structured import MLPKPredictor, StructuredPredictor
from sld.systems.coupled_oscillators import CoupledOscillators
from sld.systems.representability import (
    real_modal_basis,
    real_modal_distance,
    representability_distance,
)

D_Z, R, D_U, DELTA = 4, 2, 1, 0.05


@pytest.fixture(scope="module")
def A_and_T() -> tuple[np.ndarray, np.ndarray]:
    A = CoupledOscillators(c=0.10).A_mat
    return A, real_modal_basis(A)


# -- the basis itself -------------------------------------------------------


def test_basis_reproduces_26_0_published_numbers(A_and_T):
    """26.0's table, recomputed here rather than trusted."""
    A, _ = A_and_T
    identity, _ = representability_distance(A)
    assert identity == pytest.approx(1.0654, abs=5e-4)
    assert float(np.linalg.norm(A)) == pytest.approx(2.9614, abs=5e-4)
    assert identity / float(np.linalg.norm(A)) == pytest.approx(0.36, abs=5e-3)
    assert real_modal_distance(A) < 1e-12


def test_T_block_diagonalises_A_into_26_0_blocks(A_and_T):
    A, T = A_and_T
    M = T @ A @ np.linalg.inv(T)
    assert representability_distance(M)[0] < 1e-12
    # the two (alpha, omega) pairs 26.0 names, read off the block diagonal
    assert M[0, 0] == pytest.approx(-0.05, abs=1e-9)
    assert M[2, 2] == pytest.approx(-0.05, abs=1e-9)
    assert abs(M[0, 1]) == pytest.approx(1.5187, abs=5e-4)
    assert abs(M[2, 3]) == pytest.approx(1.0902, abs=5e-4)
    # off-block entries vanish: that IS the property the identity basis lacks
    assert np.abs(M[:2, 2:]).max() < 1e-12
    assert np.abs(M[2:, :2]).max() < 1e-12


def test_basis_is_well_conditioned_far_from_the_refusal_threshold(A_and_T):
    _, T = A_and_T
    assert float(np.linalg.cond(T)) < 10.0


def test_real_eigenvalues_are_refused_not_approximated():
    """26.0: no real basis puts such an A in block rotation-scaling form."""
    A = np.diag([-1.0, -2.0, -3.0, -4.0])
    with pytest.raises(ValueError, match="every eigenvalue to be complex"):
        real_modal_basis(A)


# -- the predictor arm ------------------------------------------------------


def _frozen(T: np.ndarray) -> MLPKPredictor:
    return MLPKPredictor(
        d_z=D_Z, r=R, d_u=D_U, delta=DELTA, width=16,
        lift="frozen", psi_matrix=torch.as_tensor(T, dtype=torch.float32),
    )


def test_pi_inverts_psi_exactly(A_and_T):
    _, T = A_and_T
    m = _frozen(T)
    z = torch.randn(32, D_Z)
    assert torch.allclose(m.project(m.lift(z)), z, atol=1e-5)


def test_psi_is_linear_not_affine(A_and_T):
    """psi(0) = 0, and psi(a+b) = psi(a) + psi(b). A bias would break both."""
    _, T = A_and_T
    m = _frozen(T)
    assert torch.allclose(m.lift(torch.zeros(1, D_Z)), torch.zeros(1, 2 * R), atol=1e-7)
    a, b = torch.randn(8, D_Z), torch.randn(8, D_Z)
    assert torch.allclose(m.lift(a + b), m.lift(a) + m.lift(b), atol=1e-5)


def test_lift_pair_carries_zero_trainable_parameters(A_and_T):
    """28.2: the arm differs from `on-learned` by the lift pair and nothing else."""
    _, T = A_and_T
    m = _frozen(T)
    assert sum(p.numel() for p in m.psi.parameters() if p.requires_grad) == 0
    assert sum(p.numel() for p in m.pi.parameters() if p.requires_grad) == 0
    # and they are still THERE, as non-trainable buffers of 2 * d_z^2 entries
    assert sum(p.numel() for p in m.psi.parameters()) == D_Z * 2 * R
    assert sum(p.numel() for p in m.pi.parameters()) == D_Z * 2 * R


def test_the_basis_receives_no_gradient(A_and_T):
    """A basis that drifts under training is not a GIVEN basis."""
    _, T = A_and_T
    m = _frozen(T)
    before = m.psi.weight.detach().clone()
    z = torch.randn(16, D_Z)
    u = torch.randn(16, D_U)
    h_next, _ = m.step(m.lift(z), u, PredictorState())
    m.project(h_next).pow(2).mean().backward()
    assert m.psi.weight.grad is None
    torch.testing.assert_close(m.psi.weight, before)
    # the head, by contrast, MUST have learned something to learn
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.head.parameters())


def test_operator_is_still_learnable(A_and_T):
    """The basis is given; the operator is not. Handing over both would be a
    different experiment and would answer a different question."""
    _, T = A_and_T
    m = _frozen(T)
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert any(n.startswith("head.") for n in trainable)
    assert not any(n.startswith(("psi.", "pi.")) for n in trainable)


# -- the refusals -----------------------------------------------------------


def test_frozen_without_a_matrix_is_refused():
    with pytest.raises(ValueError, match="requires psi_matrix"):
        MLPKPredictor(d_z=D_Z, r=R, d_u=D_U, delta=DELTA, width=16, lift="frozen")


def test_a_matrix_without_frozen_is_refused(A_and_T):
    """Accepting and ignoring it would let a config believe it froze a basis."""
    _, T = A_and_T
    for lift in ("mlp", "identity"):
        with pytest.raises(ValueError, match="only meaningful with lift='frozen'"):
            MLPKPredictor(
                d_z=D_Z, r=R, d_u=D_U, delta=DELTA, width=16, lift=lift,
                psi_matrix=torch.as_tensor(T, dtype=torch.float32),
            )


def test_singular_basis_is_refused(A_and_T):
    T = np.eye(4)
    T[3, 3] = 1e-12  # invertible in principle, useless in practice
    with pytest.raises(ValueError, match="numerically singular"):
        _frozen(T)


def test_wrong_shape_is_refused():
    with pytest.raises(ValueError, match="psi_matrix must have shape"):
        _frozen(np.eye(3))


def test_non_finite_basis_is_refused():
    T = np.eye(4)
    T[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _frozen(T)


def test_frozen_requires_2r_equals_d_z(A_and_T):
    _, T = A_and_T
    with pytest.raises(ValueError, match="requires 2r == d_z"):
        MLPKPredictor(
            d_z=D_Z, r=3, d_u=D_U, delta=DELTA, width=16, lift="frozen",
            psi_matrix=torch.as_tensor(T, dtype=torch.float32),
        )


def test_unknown_lift_is_still_a_refusal():
    """The third level must not have turned the validator into a pass-through."""
    with pytest.raises(ValueError, match="lift must be one of"):
        MLPKPredictor(d_z=D_Z, r=R, d_u=D_U, delta=DELTA, width=16, lift="frozn")


def test_all_three_levels_are_reachable(A_and_T):
    _, T = A_and_T
    assert isinstance(_frozen(T), StructuredPredictor)
    for lift in ("mlp", "identity"):
        m = MLPKPredictor(d_z=D_Z, r=R, d_u=D_U, delta=DELTA, width=16, lift=lift)
        assert isinstance(m, StructuredPredictor)
