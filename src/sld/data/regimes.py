"""Data-collection regimes: the LeJEPA-assumption axis of the project.

Each regime exposes ``sample_episode(system, T, dt, rng) -> (x0, u_seq)``
with x0 of shape (state_dim,) and u_seq of shape (T, action_dim), both
float64, every action clipped to [-u_max, u_max].

CONTRACT: u_seq must replay identically —
``system.integrate(x0, u_seq, dt)`` reproduces exactly the trajectory the
regime generated. RK4 is deterministic, and closed-loop regimes are
simulated one RK4 step at a time via the same ``system.integrate`` call
used for replay, so open-loop replay of the recorded u_seq recovers the
recorded closed-loop trajectory exactly.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from sld.systems.base import DynamicalSystem


def _closed_loop_episode(
    system: DynamicalSystem,
    x0: np.ndarray,
    policy: Callable[[np.ndarray, np.random.Generator], np.ndarray],
    T: int,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate a closed-loop episode step by step, recording actions.

    policy(x, rng) -> u (action_dim,), already clipped. Each step is one
    call to system.integrate over a single-step u, so replaying the
    recorded u_seq with system.integrate(x0, u_seq, dt) is the same
    computation and reproduces the trajectory exactly.
    """
    u_seq = np.empty((T, system.action_dim), dtype=np.float64)
    x = np.asarray(x0, dtype=np.float64)
    for t in range(T):
        u = policy(x, rng)
        u_seq[t] = u
        x = system.integrate(x, u[None, :], dt)[-1]
    return u_seq


def _ou_action_sequence(
    system: DynamicalSystem,
    T: int,
    dt: float,
    rng: np.random.Generator,
    theta_ou: float,
    sigma_stationary: float,
) -> np.ndarray:
    """Exact-transition-kernel OU action sequence, open-loop, isotropic.

    u_0 ~ N(0, sigma_stationary^2 I); for t >= 1,
    u_t = u_{t-1} * exp(-theta_ou*dt)
          + sigma_stationary * sqrt(1 - exp(-2*theta_ou*dt)) * xi_{t-1},
    xi ~ N(0, I_action_dim). This is the exact OU transition kernel written
    directly in terms of the stationary std, not an Euler-Maruyama
    approximation. Clipped to [-u_max, u_max] after generation.
    """
    action_dim = system.action_dim
    decay = np.exp(-theta_ou * dt)
    diffusion = sigma_stationary * np.sqrt(1.0 - np.exp(-2.0 * theta_ou * dt))
    u_seq = np.empty((T, action_dim), dtype=np.float64)
    u_seq[0] = sigma_stationary * rng.standard_normal(action_dim)
    for t in range(1, T):
        u_seq[t] = u_seq[t - 1] * decay + diffusion * rng.standard_normal(action_dim)
    return np.clip(u_seq, -system.u_max, system.u_max)


class OURegime:
    """Isotropic Ornstein-Uhlenbeck exploration noise, wide initial conditions.

    See ``_ou_action_sequence`` for the exact discretization. Actions are
    open-loop (no state feedback). ``sigma_stationary`` defaults to
    ``system.u_max / 3.0`` at sample time, chosen so the +/-u_max clip is
    rare (roughly a 3-sigma event).
    """

    def __init__(
        self,
        theta_ou: float = 1.0,
        sigma_stationary: float | None = None,
        ic_regime: str = "wide",
    ) -> None:
        self.theta_ou = theta_ou
        self.sigma_stationary = sigma_stationary
        self.ic_regime = ic_regime

    def sample_episode(
        self, system: DynamicalSystem, T: int, dt: float, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        x0 = system.sample_ic(rng, self.ic_regime)
        sigma_stationary = (
            system.u_max / 3.0 if self.sigma_stationary is None else self.sigma_stationary
        )
        u_seq = _ou_action_sequence(system, T, dt, rng, self.theta_ou, sigma_stationary)
        return x0, u_seq


class NarrowPolicyRegime:
    """Energy-shaping controller plus small noise: narrow, policy-concentrated data.

    u_t = clip(k_e * (E_target - energy(x_t)) * x_t[vel_index] + sigma * xi_t, +/- u_max)

    The energy-shaping form pumps energy into the system in phase with the
    velocity while below E_target (a standard swing-up primitive) and lets
    it bleed off otherwise. It is a declared stand-in for "narrow policy
    data", shaped specifically for pendulum-like systems exposing
    ``energy(x)`` — not a general controller. For the pendulum's upright
    target, E_target = 2*m*g*L (potential energy at theta = pi).
    """

    def __init__(
        self,
        E_target: float,
        k_e: float = 1.0,
        sigma: float = 0.05,
        ic_regime: str = "default",
        vel_index: int = 1,
    ) -> None:
        self.E_target = E_target
        self.k_e = k_e
        self.sigma = sigma
        self.ic_regime = ic_regime
        self.vel_index = vel_index

    def sample_episode(
        self, system: DynamicalSystem, T: int, dt: float, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(system, "energy"):
            raise AttributeError("NarrowPolicyRegime requires system.energy(x)")
        x0 = system.sample_ic(rng, self.ic_regime)

        def policy(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
            # energy-shaping torque: proportional to the energy deficit,
            # applied in phase with the velocity, plus small exploration noise
            energy_error = self.E_target - system.energy(x)
            u = self.k_e * energy_error * x[self.vel_index] + self.sigma * rng.standard_normal(
                system.action_dim
            )
            return np.clip(u, -system.u_max, system.u_max)

        u_seq = _closed_loop_episode(system, x0, policy, T, dt, rng)
        return x0, u_seq


class AnisotropicRegime:
    """Anisotropic initial-condition covariance, isotropic OU actions.

    x0 = ic_scales * rng.standard_normal(state_dim) (elementwise): a
    diagonal-covariance IC distribution with declared per-dimension scales.
    Actions reuse the same exact-OU kernel as OURegime, which is isotropic
    over action_dim. With 1-D actions (as in DampedPendulum) the anisotropy
    therefore lives entirely in the ICs, not in the action distribution —
    a declared limitation, not a general anisotropic-action regime.
    """

    def __init__(
        self,
        ic_scales: tuple[float, ...],
        theta_ou: float = 1.0,
        sigma_stationary: float | None = None,
    ) -> None:
        self.ic_scales = np.asarray(ic_scales, dtype=np.float64)
        self.theta_ou = theta_ou
        self.sigma_stationary = sigma_stationary

    def sample_episode(
        self, system: DynamicalSystem, T: int, dt: float, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        x0 = self.ic_scales * rng.standard_normal(system.state_dim)
        sigma_stationary = (
            system.u_max / 3.0 if self.sigma_stationary is None else self.sigma_stationary
        )
        u_seq = _ou_action_sequence(system, T, dt, rng, self.theta_ou, sigma_stationary)
        return x0, u_seq


class ActionConditionedRegime:
    """State-feedback exploration: actions statistically dependent on the state.

    u_t = clip(K_gain @ x_t + sigma * xi_t, +/- u_max). Of the four regimes
    this breaks the "actions independent of state" isotropy assumption most
    directly, since u_t is a deterministic (up to noise) linear function of
    x_t. K_gain has shape (action_dim, state_dim), declared at construction.
    """

    def __init__(
        self,
        K_gain: np.ndarray,
        sigma: float = 0.3,
        ic_regime: str = "wide",
    ) -> None:
        self.K_gain = np.asarray(K_gain, dtype=np.float64)
        self.sigma = sigma
        self.ic_regime = ic_regime

    def sample_episode(
        self, system: DynamicalSystem, T: int, dt: float, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        x0 = system.sample_ic(rng, self.ic_regime)

        def policy(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
            u = self.K_gain @ x + self.sigma * rng.standard_normal(system.action_dim)
            return np.clip(u, -system.u_max, system.u_max)

        u_seq = _closed_loop_episode(system, x0, policy, T, dt, rng)
        return x0, u_seq
