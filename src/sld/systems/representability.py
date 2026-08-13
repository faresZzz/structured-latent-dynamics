"""How far a generator is from the block-diagonal rotation-scaling class.

A ground-truth property of a system, not a measurement on a learned model, so
it lives here rather than under ``audit/`` (REGLES section 2). NumPy only: it
is closed-form arithmetic on a constant matrix.

Why it exists (PROTOCOL 26.0). A rotation-scaling block imposes
``[[alpha, -omega], [omega, alpha]]``. Section 23.7 already measured how far
the oscillators' SPECTRUM is from what such blocks can carry -- that is
``delta(c)``, and it is zero at ``c = 0.10``, where the two conjugate pairs are
exactly representable. But "exactly representable" holds **in a basis**, and
25.3's ``on-stable-nolift`` arm removes the learned change of basis by setting
``psi = pi = identity``. This module measures what that arm is then left
facing: the distance in the coordinates it is forced to work in.

The answer at ``c = 0.10`` is ``1.0654`` -- 36 % of ``‖A‖_F`` -- against
``1.97e-15`` in the best basis. The lift pair is not an overhead on the
structure; it is the structure's change of basis.
"""

from __future__ import annotations

import itertools

import numpy as np


def nearest_block_rotation_scaling(A: np.ndarray, pairing: tuple[tuple[int, int], ...]) -> np.ndarray:
    """The closest block-diagonal rotation-scaling matrix for a FIXED pairing.

    Least squares, in closed form: for a block on coordinates ``(i, j)`` the
    best ``alpha`` is ``(a_ii + a_jj) / 2`` and the best ``omega`` is
    ``(a_ji - a_ij) / 2``; every entry outside a block must be zero.
    """
    M = np.zeros_like(A, dtype=float)
    for i, j in pairing:
        alpha = (A[i, i] + A[j, j]) / 2.0
        omega = (A[j, i] - A[i, j]) / 2.0
        M[i, i] = M[j, j] = alpha
        M[i, j], M[j, i] = -omega, omega
    return M


def representability_distance(A: np.ndarray) -> tuple[float, tuple[tuple[int, int], ...]]:
    """Minimal Frobenius distance from ``A`` to the block-diagonal
    rotation-scaling class, minimised over pairings AND over each block's
    ``(alpha_i, omega_i)``.

    Args:
        A: square, even order, in the coordinates the model actually works in.

    Returns:
        ``(distance, pairing)`` -- the distance and the pairing achieving it.

    Raises:
        ValueError: ``A`` is not square, or its order is odd (2x2 blocks
            cannot tile an odd dimension, and silently dropping a coordinate
            would answer a different question).
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got shape {A.shape}")
    d = A.shape[0]
    if d % 2:
        raise ValueError(f"A's order must be even for 2x2 blocks, got {d}")

    best: tuple[float, tuple] | None = None
    for pairing in _pairings(d):
        M = nearest_block_rotation_scaling(A, pairing)
        # off-block entries of A must vanish; M already has them at zero, so
        # the Frobenius norm of the difference counts them in full
        dist = float(np.linalg.norm(A - M))
        if best is None or dist < best[0]:
            best = (dist, pairing)
    return best


def _pairings(d: int):
    """Every perfect matching of ``range(d)`` into unordered pairs."""
    items = list(range(d))
    if not items:
        yield ()
        return
    first, rest = items[0], items[1:]
    for k in range(len(rest)):
        pair = (first, rest[k])
        remaining = rest[:k] + rest[k + 1 :]
        for sub in _pairings_of(remaining):
            yield (pair, *sub)


def _pairings_of(items: list[int]):
    if not items:
        yield ()
        return
    first, rest = items[0], items[1:]
    for k in range(len(rest)):
        remaining = rest[:k] + rest[k + 1 :]
        for sub in _pairings_of(remaining):
            yield ((first, rest[k]), *sub)


def real_modal_distance(A: np.ndarray) -> float:
    """The same distance AFTER the change of basis a learned ``psi``/``pi``
    could realise -- i.e. in the real modal basis.

    Exactly zero (to machine precision) when every eigenvalue is complex: the
    real Jordan form of such an ``A`` IS block-diagonal rotation-scaling. When
    real eigenvalues are present no basis achieves it and the returned value is
    the distance in the best real modal basis found, which is positive.
    """
    A = np.asarray(A, dtype=float)
    lam, vec = np.linalg.eig(A)
    if not np.all(np.abs(lam.imag) > 1e-12):
        # real eigenvalues present: no exact real block form exists
        return representability_distance(A)[0]
    cols, used = [], set()
    for i in range(len(lam)):
        if i in used:
            continue
        j = int(np.argmin([abs(lam[k] - np.conj(lam[i])) if k != i else np.inf for k in range(len(lam))]))
        used |= {i, j}
        cols += [vec[:, i].real, vec[:, i].imag]
    P = np.column_stack(cols)
    M = np.linalg.solve(P, A @ P)
    return representability_distance(M)[0]


def demo() -> None:
    """Self-check: the two counterexamples PROTOCOL 26.0 names, by hand."""
    from sld.systems.coupled_oscillators import CoupledOscillators

    A = CoupledOscillators(c=0.10).A_mat
    assert A[0, 2] == 1.0 and A[2, 0] == -1.5, "pairing (1,2)(3,4) is refused by these"
    assert A[0, 0] == 0.0 and A[2, 2] == -0.10, "pairing (1,3)(2,4) is refused by a11 != a22"
    identity_basis, _ = representability_distance(A)
    best_basis = real_modal_distance(A)
    assert identity_basis > 1.0, identity_basis
    assert best_basis < 1e-12, best_basis
    print(f"identity basis {identity_basis:.4f}   best basis {best_basis:.2e}")


if __name__ == "__main__":
    demo()
