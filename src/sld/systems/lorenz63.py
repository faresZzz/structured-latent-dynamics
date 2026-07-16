"""Lorenz (1963) system — the repo's adversarial chaotic reference.

Autonomous (no control input): dx/dt = f(x). Chaotic for the classic
parameters (sigma=10, rho=28, beta=8/3), used across this repo as the system
where a latent model's structure is stress-tested against sensitivity to
initial conditions rather than against control difficulty.

Reference: E. N. Lorenz, "Deterministic Nonperiodic Flow", J. Atmos. Sci.,
1963. The largest Lyapunov exponent for the classic parameters is a
well-known literature constant, ``LAMBDA_MAX_REF``, used here only as a
sanity anchor for the Benettin-style estimator in the test suite.
"""

from __future__ import annotations

import numpy as np

from sld.systems.base import DynamicalSystem


class Lorenz63(DynamicalSystem):
    """The Lorenz 1963 chaotic convection system, no control input.

    Equations of motion:
        x_dot = sigma * (y - x)
        y_dot = x * (rho - z) - y
        z_dot = x * y - beta * z

    State x = (x, y, z). Autonomous: action_dim = 0, u_max = 0.0; ``f``
    ignores u and ``integrate`` must be called with u_seq of shape (T, 0).
    """

    state_dim = 3
    action_dim = 0
    # literature reference value for the largest Lyapunov exponent (Lorenz 1963,
    # classic parameters); used as a sanity anchor, not an exact target
    LAMBDA_MAX_REF = 0.9056

    def __init__(
        self,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8.0 / 3.0,
        u_max: float = 0.0,
    ) -> None:
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.u_max = u_max
        # partial sensor sees the x coordinate only
        self.partial_indices = (0,)

    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        # autonomous system: u is ignored (action_dim == 0)
        x, y, z = x[0], x[1], x[2]
        x_dot = self.sigma * (y - x)
        y_dot = x * (self.rho - z) - y
        z_dot = x * y - self.beta * z
        return np.array([x_dot, y_dot, z_dot], dtype=np.float64)

    def sample_ic(self, rng: np.random.Generator, regime: str = "default") -> np.ndarray:
        """Sample an initial condition.

        "on_attractor" (also what "default" maps to): jitter a point near the
        attractor and burn in for 30 time units at dt=0.01 so the returned
        state actually lies on the attractor — a raw box-sampled point is
        NOT on the attractor and would bias any statistic computed from it.
        "box": an independent, generously-sized box around the attractor,
        with no burn-in (useful for stress-testing far-from-attractor
        behavior, not for measuring on-attractor statistics).
        """
        if regime in ("on_attractor", "default"):
            x0 = np.array([1.0, 1.0, 25.0]) + rng.uniform(-5.0, 5.0, size=3)
            dt_burn_in = 0.01
            n_burn_in = round(30.0 / dt_burn_in)
            u_seq = np.zeros((n_burn_in, 0))
            traj = self.integrate(x0, u_seq, dt_burn_in)
            return traj[-1]
        elif regime == "box":
            x_coord = rng.uniform(-20.0, 20.0)
            y_coord = rng.uniform(-25.0, 25.0)
            z_coord = rng.uniform(5.0, 45.0)
            return np.array([x_coord, y_coord, z_coord], dtype=np.float64)
        else:
            raise ValueError(f"unknown IC regime: {regime!r}")
