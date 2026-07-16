"""Contract for ground-truth dynamical systems.

Every system is a continuous-time ODE dx/dt = f(x, u) integrated with
fixed-step RK4 (zero-order hold on u within a step). All arrays are NumPy
float64 — ground truth never runs on a learned-model backend (REGLES §5:
the audit needs an exact reference, not a fast one).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class DynamicalSystem(ABC):
    """Base contract: f, RK4 integration, IC sampling, observation, linearization.

    Subclasses define the vector field ``f`` and declare ``state_dim``,
    ``action_dim``, ``u_max`` and ``partial_indices``. Conventions (angle
    origins, parameter meanings) are documented per system.
    """

    state_dim: int
    action_dim: int
    u_max: float
    # indices kept by observe(x, "partial") — each system declares which
    # coordinates a partial sensor sees (typically positions, not velocities)
    partial_indices: tuple[int, ...]

    @abstractmethod
    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Vector field dx/dt at state x (state_dim,) under action u (action_dim,)."""

    def integrate(self, x0: np.ndarray, u_seq: np.ndarray, dt: float) -> np.ndarray:
        """Roll the ODE forward with fixed-step RK4, zero-order hold on u.

        x0: (state_dim,); u_seq: (T, action_dim) — for autonomous systems
        (action_dim == 0) pass shape (T, 0). Returns (T + 1, state_dim)
        including x0. dt is declared per system config, never global.
        """
        x0 = np.asarray(x0, dtype=np.float64)
        u_seq = np.asarray(u_seq, dtype=np.float64).reshape(len(u_seq), self.action_dim)
        traj = np.empty((len(u_seq) + 1, self.state_dim), dtype=np.float64)
        traj[0] = x0
        x = x0
        for t, u in enumerate(u_seq):
            k1 = self.f(x, u)
            k2 = self.f(x + 0.5 * dt * k1, u)
            k3 = self.f(x + 0.5 * dt * k2, u)
            k4 = self.f(x + dt * k3, u)
            x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            traj[t + 1] = x
        return traj

    @abstractmethod
    def sample_ic(self, rng: np.random.Generator, regime: str = "default") -> np.ndarray:
        """Sample an initial condition from a named per-system distribution.

        Each system documents its supported regime names (e.g. "default",
        "wide", "on_attractor") and raises ValueError on unknown ones.
        """

    def observe(self, x: np.ndarray, mode: str = "full") -> np.ndarray:
        """Project a state (or trajectory, leading dims kept) onto the sensor.

        "full" is the identity; "partial" keeps ``partial_indices`` only.
        """
        if mode == "full":
            return x
        if mode == "partial":
            return x[..., list(self.partial_indices)]
        raise ValueError(f"unknown observation mode: {mode!r}")

    def linearize(
        self, x_eq: np.ndarray, u_eq: np.ndarray, eps: float = 1e-6
    ) -> tuple[np.ndarray, np.ndarray]:
        """Continuous-time (A, B) at (x_eq, u_eq) by central finite differences.

        Central differences are exact (to rounding) on linear systems, O(eps^2)
        otherwise. Systems with closed-form linearizations expose them as
        separate analytic methods so tests can compare the two.
        """
        x_eq = np.asarray(x_eq, dtype=np.float64)
        u_eq = np.asarray(u_eq, dtype=np.float64).reshape(self.action_dim)
        A = np.empty((self.state_dim, self.state_dim), dtype=np.float64)
        for j in range(self.state_dim):
            dx = np.zeros(self.state_dim)
            dx[j] = eps
            A[:, j] = (self.f(x_eq + dx, u_eq) - self.f(x_eq - dx, u_eq)) / (2.0 * eps)
        B = np.empty((self.state_dim, self.action_dim), dtype=np.float64)
        for j in range(self.action_dim):
            du = np.zeros(self.action_dim)
            du[j] = eps
            B[:, j] = (self.f(x_eq, u_eq + du) - self.f(x_eq, u_eq - du)) / (2.0 * eps)
        return A, B
