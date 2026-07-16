"""Two damped, spring-coupled masses on a line — the repo's linear reference.

Convention: state x = (q1, q2, q1_dot, q2_dot), positions measured from each
mass's own equilibrium. Mass 1 is anchored to a wall by spring k1, mass 2 to
the opposite wall by spring k2, and the two masses are coupled by spring
k_c. Both masses carry viscous damping c. The control force u acts on mass 1
only. Because every term is linear in (x, u), this system is the repo's
analytically tractable reference: its linearization IS the dynamics, and its
controllability Gramian can be computed in closed form rather than estimated.
"""

from __future__ import annotations

import numpy as np

from sld.systems.base import DynamicalSystem


class CoupledOscillators(DynamicalSystem):
    """Two damped masses coupled by a spring, force input on mass 1 only.

    Equations of motion:
        m1 * q1_ddot = -k1*q1 - k_c*(q1 - q2) - c*q1_dot + u
        m2 * q2_ddot = -k2*q2 - k_c*(q2 - q1) - c*q2_dot

    In state-space form dx/dt = A_mat @ x + B_mat @ u with
    x = (q1, q2, q1_dot, q2_dot):
        A_mat = [[0, 0, 1, 0],
                 [0, 0, 0, 1],
                 [-(k1+k_c)/m1, k_c/m1, -c/m1, 0],
                 [k_c/m2, -(k2+k_c)/m2, 0, -c/m2]]
        B_mat = [[0], [0], [1/m1], [0]]

    Both A_mat and B_mat are constant, built once in ``__init__``.
    """

    state_dim = 4
    action_dim = 1

    def __init__(
        self,
        m1: float = 1.0,
        m2: float = 1.0,
        k1: float = 1.0,
        k_c: float = 0.5,
        k2: float = 1.5,
        c: float = 0.1,
        u_max: float = 2.0,
    ) -> None:
        self.m1 = m1
        self.m2 = m2
        self.k1 = k1
        self.k_c = k_c
        self.k2 = k2
        self.c = c
        self.u_max = u_max
        # partial sensor sees both positions, not the velocities
        self.partial_indices = (0, 1)

        # constant state-space matrices — this system is exactly linear
        self.A_mat = np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [-(k1 + k_c) / m1, k_c / m1, -c / m1, 0.0],
                [k_c / m2, -(k2 + k_c) / m2, 0.0, -c / m2],
            ],
            dtype=np.float64,
        )
        self.B_mat = np.array([[0.0], [0.0], [1.0 / m1], [0.0]], dtype=np.float64)

    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        # linear state-space dynamics: dx/dt = A_mat @ x + B_mat @ u
        return self.A_mat @ x + self.B_mat @ u

    def analytic_modes(self) -> np.ndarray:
        """Eigenvalues of A_mat — exact since the full system is linear."""
        return np.linalg.eigvals(self.A_mat)

    def exact_gramian(
        self, T: float, dt: float, R_u: np.ndarray | None = None
    ) -> np.ndarray:
        """Finite-horizon discrete-time reachability Gramian at step dt.

        Convention (matches the audit module): discretize (A_mat, B_mat) by
        zero-order hold at step dt to get (A_bar, B_bar), then accumulate the
        discrete-time Gramian over N = round(T/dt) steps:
            W_c = sum_{k=0}^{N-1} A_bar^k @ B_bar @ R_u^-1 @ B_bar.T @ (A_bar^T)^k

        A_bar is built via eigendecomposition (A_mat = V diag(lam) V^-1,
        A_bar = real(V @ diag(exp(lam*dt)) @ V^-1)) rather than scipy's expm,
        since scipy is not a dependency of this repo. This assumes A_mat is
        diagonalizable with distinct eigenvalues, true for generic spring
        constants; the reconstruction is asserted below as a guard.
        B_bar = A_mat^-1 @ (A_bar - I) @ B_mat, valid because A_mat is
        invertible here (both wall springs k1, k2 > 0 keep it free of a zero
        eigenvalue; also asserted below).
        """
        if R_u is None:
            R_u = np.eye(self.action_dim)
        R_u_inv = np.linalg.inv(R_u)

        # eigendecomposition of A_mat, used for the exact ZOH discretization
        lam, V = np.linalg.eig(self.A_mat)
        V_inv = np.linalg.inv(V)
        reconstruction_error = np.linalg.norm(V @ np.diag(lam) @ V_inv - self.A_mat)
        assert reconstruction_error < 1e-10, (
            "A_mat is not diagonalizable as assumed by exact_gramian"
        )
        A_bar = np.real(V @ np.diag(np.exp(lam * dt)) @ V_inv)

        assert np.min(np.abs(lam)) > 1e-12, (
            "A_mat has a near-zero eigenvalue; ZOH formula for B_bar breaks down"
        )
        A_mat_inv = np.linalg.inv(self.A_mat)
        B_bar = A_mat_inv @ (A_bar - np.eye(self.state_dim)) @ self.B_mat

        N = round(T / dt)
        W_c = np.zeros((self.state_dim, self.state_dim), dtype=np.float64)
        M_k = B_bar.copy()  # M_k = A_bar^k @ B_bar, starts at k = 0
        for _ in range(N):
            W_c += M_k @ R_u_inv @ M_k.T
            M_k = A_bar @ M_k
        # numerical-hygiene rule: Gramians are always symmetrized
        return (W_c + W_c.T) / 2.0

    def sample_ic(self, rng: np.random.Generator, regime: str = "default") -> np.ndarray:
        if regime == "default":
            return rng.uniform(-1.0, 1.0, size=self.state_dim)
        elif regime == "wide":
            return rng.uniform(-3.0, 3.0, size=self.state_dim)
        else:
            raise ValueError(f"unknown IC regime: {regime!r}")
