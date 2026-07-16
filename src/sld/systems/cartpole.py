"""Cart-pole with a point-mass pole, force input on the cart.

CONVENTION: theta = 0 is UPRIGHT (the unstable equilibrium), theta measured
from the vertical, so the pole hanging at rest is theta = pi. This is the
opposite convention from ``DampedPendulum`` (where 0 is the stable, hanging
equilibrium) — the two systems are not interchangeable in angle origin.
"""

from __future__ import annotations

import numpy as np

from sld.systems.base import DynamicalSystem


class CartPole(DynamicalSystem):
    """Cart of mass m_c on a frictionless track, point-mass pole at the tip.

    State x = (p, p_dot, theta, theta_dot), p the cart position, theta the
    pole angle from vertical (0 = upright, unstable). Action u = (F,), a
    continuous force applied to the cart.

    Equations of motion (Euler-Lagrange for a cart + point-mass pole on a
    massless rod of length l_pole):
        (m_c + m_p)*p_ddot + m_p*l_pole*theta_ddot*cos(theta)
            - m_p*l_pole*theta_dot^2*sin(theta) = F
        l_pole*theta_ddot + p_ddot*cos(theta) - g*sin(theta) = 0
    These are two linear equations in (p_ddot, theta_ddot); solving them
    jointly (see ``f``) gives the closed-form accelerations
        theta_ddot = ((m_c+m_p)*g*sin(theta) - cos(theta)*(F + m_p*l_pole*theta_dot^2*sin(theta)))
                     / (l_pole*(m_c + m_p*sin(theta)^2))
        p_ddot     = (F + m_p*l_pole*theta_dot^2*sin(theta) - m_p*g*sin(theta)*cos(theta))
                     / (m_c + m_p*sin(theta)^2)
    """

    state_dim = 4
    action_dim = 1

    def __init__(
        self,
        m_c: float = 1.0,
        m_p: float = 0.1,
        l_pole: float = 0.5,
        g: float = 9.81,
        u_max: float = 10.0,
    ) -> None:
        self.m_c = m_c
        self.m_p = m_p
        self.l_pole = l_pole
        self.g = g
        self.u_max = u_max
        # partial sensor sees cart position and pole angle, not velocities
        self.partial_indices = (0, 2)

    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        p_dot, theta, theta_dot = x[1], x[2], x[3]
        F = u[0]
        m_c, m_p, l_pole, g = self.m_c, self.m_p, self.l_pole, self.g
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        # shared denominator from solving the coupled Euler-Lagrange equations
        denom = m_c + m_p * sin_theta**2
        theta_ddot = (
            (m_c + m_p) * g * sin_theta
            - cos_theta * (F + m_p * l_pole * theta_dot**2 * sin_theta)
        ) / (l_pole * denom)
        p_ddot = (
            F + m_p * l_pole * theta_dot**2 * sin_theta - m_p * g * sin_theta * cos_theta
        ) / denom
        return np.array([p_dot, p_ddot, theta_dot, theta_ddot], dtype=np.float64)

    def linearize_upright(self) -> tuple[np.ndarray, np.ndarray]:
        """Closed-form linearization at x = 0, F = 0 (first-order expansion).

        Near theta = 0: sin(theta) ~ theta, cos(theta) ~ 1, and the
        theta_dot^2*sin(theta) terms are second order and drop out, giving:
            A = [[0, 1, 0, 0],
                 [0, 0, -m_p*g/m_c, 0],
                 [0, 0, 0, 1],
                 [0, 0, (m_c+m_p)*g/(l_pole*m_c), 0]]
            B = [[0], [1/m_c], [0], [-1/(l_pole*m_c)]]
        NOTE the sign of B[3]: pushing the cart right (+F) makes the upright
        pole fall backwards (negative theta_ddot contribution) in this
        theta-from-vertical convention.
        """
        m_c, m_p, l_pole, g = self.m_c, self.m_p, self.l_pole, self.g
        A = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -m_p * g / m_c, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, (m_c + m_p) * g / (l_pole * m_c), 0.0],
            ],
            dtype=np.float64,
        )
        B = np.array([[0.0], [1.0 / m_c], [0.0], [-1.0 / (l_pole * m_c)]], dtype=np.float64)
        return A, B

    def energy(self, x: np.ndarray) -> np.ndarray:
        """Total mechanical energy, pole pivot (cart height) as reference.

        E = 0.5*(m_c+m_p)*p_dot^2 + m_p*l_pole*p_dot*theta_dot*cos(theta)
            + 0.5*m_p*l_pole^2*theta_dot^2 + m_p*g*l_pole*cos(theta)
        Vectorized over any leading batch/time dimensions of x (..., 4).
        """
        p_dot, theta, theta_dot = x[..., 1], x[..., 2], x[..., 3]
        m_c, m_p, l_pole, g = self.m_c, self.m_p, self.l_pole, self.g
        return (
            0.5 * (m_c + m_p) * p_dot**2
            + m_p * l_pole * p_dot * theta_dot * np.cos(theta)
            + 0.5 * m_p * l_pole**2 * theta_dot**2
            + m_p * g * l_pole * np.cos(theta)
        )

    def sample_ic(self, rng: np.random.Generator, regime: str = "default") -> np.ndarray:
        if regime == "default":
            p = rng.uniform(-0.5, 0.5)
            p_dot = rng.uniform(-0.5, 0.5)
            theta = rng.uniform(-0.2, 0.2)
            theta_dot = rng.uniform(-0.5, 0.5)
        elif regime == "hanging":
            p = rng.uniform(-0.5, 0.5)
            p_dot = rng.uniform(-0.2, 0.2)
            theta = np.pi + rng.uniform(-0.3, 0.3)
            theta_dot = rng.uniform(-0.2, 0.2)
        elif regime == "wide":
            p = rng.uniform(-1.0, 1.0)
            p_dot = rng.uniform(-1.0, 1.0)
            theta = rng.uniform(-np.pi, np.pi)
            theta_dot = rng.uniform(-2.0, 2.0)
        else:
            raise ValueError(f"unknown IC regime: {regime!r}")
        return np.array([p, p_dot, theta, theta_dot], dtype=np.float64)
