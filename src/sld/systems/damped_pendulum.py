"""Damped pendulum with torque input, a minimal but genuinely nonlinear system.

Convention: theta = 0 is hanging straight down (the stable equilibrium),
theta = pi is the inverted (unstable) equilibrium. State x = (theta, theta_dot).
"""

from __future__ import annotations

import numpy as np

from sld.systems.base import DynamicalSystem


class DampedPendulum(DynamicalSystem):
    """Torque-driven damped pendulum.

    theta = 0: hanging down (stable). theta = pi: inverted (unstable).
    State x = (theta, theta_dot), action u = (torque,).

    Equation of motion (see ``f``):
        theta_ddot = -omega0^2 sin(theta) - 2 zeta omega0 theta_dot + u / (m L^2)
    with omega0 = sqrt(g / L) the undamped natural frequency and zeta the
    (linear, viscous) damping ratio.
    """

    state_dim = 2
    action_dim = 1

    def __init__(
        self,
        m: float = 1.0,
        L: float = 1.0,
        g: float = 9.81,
        zeta: float = 0.1,
        u_max: float = 2.0,
    ) -> None:
        self.m = m
        self.L = L
        self.g = g
        self.zeta = zeta
        self.u_max = u_max
        self.omega0 = np.sqrt(g / L)
        # partial sensor sees the angle only, not the angular velocity
        self.partial_indices = (0,)

    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        theta, theta_dot = x[0], x[1]
        # nonlinear pendulum dynamics: gravity restoring torque, linear viscous
        # damping, external torque scaled by the moment of inertia m*L^2
        theta_ddot = (
            -(self.omega0**2) * np.sin(theta)
            - 2.0 * self.zeta * self.omega0 * theta_dot
            + u[0] / (self.m * self.L**2)
        )
        return np.array([theta_dot, theta_ddot], dtype=np.float64)

    def analytic_eigenvalues(self) -> np.ndarray:
        """Eigenvalues of the linearization at (theta, theta_dot) = (0, 0).

        Characteristic polynomial: lambda^2 + 2*zeta*omega0*lambda + omega0^2 = 0
        Roots: lambda = -zeta*omega0 +/- omega0*sqrt(zeta^2 - 1)
        which is complex (underdamped) for zeta < 1 and real for zeta >= 1.
        """
        coeffs = [1.0, 2.0 * self.zeta * self.omega0, self.omega0**2]
        return np.roots(coeffs).astype(np.complex128)

    def energy(self, x: np.ndarray) -> np.ndarray:
        """Mechanical energy E = kinetic + gravitational potential.

        E = 0.5 * m * L^2 * theta_dot^2 + m * g * L * (1 - cos(theta))
        Vectorized over any leading batch/time dimensions of x (..., 2).
        """
        theta, theta_dot = x[..., 0], x[..., 1]
        return 0.5 * self.m * self.L**2 * theta_dot**2 + self.m * self.g * self.L * (
            1.0 - np.cos(theta)
        )

    def sample_ic(self, rng: np.random.Generator, regime: str = "default") -> np.ndarray:
        if regime == "default":
            theta = rng.uniform(-0.5, 0.5)
            theta_dot = rng.uniform(-0.5, 0.5)
        elif regime == "wide":
            theta = rng.uniform(-np.pi, np.pi)
            theta_dot = rng.uniform(-2.0 * self.omega0, 2.0 * self.omega0)
        else:
            raise ValueError(f"unknown IC regime: {regime!r}")
        return np.array([theta, theta_dot], dtype=np.float64)
