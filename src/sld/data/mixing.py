"""Observation mixing maps: a smooth bijection and coordinate projection."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _random_rotation(dim: int, rng: np.random.Generator) -> np.ndarray:
    """A random rotation in SO(dim) via QR of a standard Gaussian matrix.

    QR of a standard Gaussian gives an orthogonal Q with a sign ambiguity;
    fixing R's diagonal to positive removes that ambiguity, then a single
    column flip (if det = -1) lands exactly in SO(dim) rather than O(dim).
    """
    A = rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0
    return Q


class SmoothMixing:
    """Smooth, bijective observation mixing g(z) = R_2 @ tanh(a * (R_1 @ z) + b).

    R_1, R_2 are fixed random rotations in SO(dim) and a ~ U(a_range)
    (elementwise, all > 0), b ~ U(b_range) are fixed at construction from
    ``seed``. tanh is a smooth bijection onto (-1, 1)^dim, and R_1, R_2 are
    isometries, so g is a smooth bijection of R^dim onto R_2 @ (-1, 1)^dim.
    """

    def __init__(
        self,
        dim: int,
        seed: int,
        a_range: tuple[float, float] = (0.5, 1.5),
        b_range: tuple[float, float] = (-0.3, 0.3),
    ) -> None:
        self.dim = dim
        rng = np.random.default_rng(seed)
        self.R_1 = _random_rotation(dim, rng)
        self.R_2 = _random_rotation(dim, rng)
        self.a = rng.uniform(a_range[0], a_range[1], size=dim)
        self.b = rng.uniform(b_range[0], b_range[1], size=dim)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        # forward mixing: rotate, squash with tanh, rotate again
        w = np.tanh(self.a * (z @ self.R_1.T) + self.b)
        return w @ self.R_2.T

    def inverse(self, x: np.ndarray) -> np.ndarray:
        """Closed-form inverse; artanh clipped near +/-1 to stay finite.

        The inverse's Lipschitz constant blows up as |w| -> 1 (d/dw artanh(w)
        = 1/(1 - w^2)); this is a declared property of the map, not hidden
        by the clip — the clip only guards the exact +/-1 boundary that
        rounding could otherwise produce.
        """
        x = np.asarray(x, dtype=np.float64)
        w = np.clip(x @ self.R_2, -(1.0 - 1e-7), 1.0 - 1e-7)
        z = (np.arctanh(w) - self.b) / self.a
        return z @ self.R_1

    def lipschitz_bound(self) -> float:
        """max(a): rotations are isometries and |d tanh| <= 1, so ||J_g||_2 <= max_i a_i.

        J_g(z) = R_2 @ diag(a_i * sech^2(...)) @ R_1: a product of two
        orthogonal (spectral norm 1) matrices and a diagonal matrix with
        entries in (0, a_i]. The spectral norm of the product is the
        diagonal's, bounded by max_i a_i, everywhere.
        """
        return float(np.max(self.a))


class PartialObservation:
    """Coordinate projection: __call__(x) = x[..., indices]."""

    def __init__(self, indices: Sequence[int]) -> None:
        self.indices = tuple(indices)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x)[..., list(self.indices)]
