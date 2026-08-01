"""Tests for ``sld.systems.base.DynamicalSystem.integrate`` (the shared RK4
stepper every system subclasses) -- independent of any one system.

REGLES section 7.1: the old order test (``test_damped_pendulum.py::
test_rk4_fourth_order_convergence``) compares RK4 at two step sizes against
a REFERENCE computed with a FINER run of the SAME code path. That proves
4th-order self-convergence -- any consistent one-step method of order 4
(even one with wrong coefficients, as long as they are still order-4
accurate) would pass it. It does not prove the classical RK4 COEFFICIENTS
(1,2,2,1)/6 are the ones actually implemented. This file adds an
independent, hand-derived oracle that pins the exact coefficients.
"""

from __future__ import annotations

import numpy as np

from sld.systems.base import DynamicalSystem


class _AffineSystem(DynamicalSystem):
    """Minimal DynamicalSystem: x' = a*x + b*u, scalar state and action."""

    state_dim = 1
    action_dim = 1
    u_max = 10.0
    partial_indices = (0,)

    def __init__(self, a: float, b: float) -> None:
        self.a = a
        self.b = b

    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.array([self.a * x[0] + self.b * u[0]])

    def sample_ic(self, rng: np.random.Generator, regime: str = "default") -> np.ndarray:
        raise NotImplementedError  # unused by this test


def _hand_rk4_step(a: float, b: float, u: float, x0: float, dt: float) -> float:
    """Exact closed-form RK4 update for x' = a*x + b*u, u constant (ZOH) over
    the step -- worked out by hand, not by running the code under test.

    Write g = b*u (constant forcing), z = a*dt. Since f is linear in x, each
    RK4 stage is proportional to k1 = a*x0 + g:
        k1 = a*x0 + g
        k2 = a*(x0 + dt/2*k1) + g = k1 + (z/2)*k1        = k1*(1 + z/2)
        k3 = a*(x0 + dt/2*k2) + g = k1 + (z/2)*k2        = k1*(1 + z/2 + z^2/4)
        k4 = a*(x0 + dt*k3)   + g = k1 + z*k3            = k1*(1 + z + z^2/2 + z^3/4)
    x1 = x0 + dt/6*(k1 + 2*k2 + 2*k3 + k4)
       = x0 + dt/6 * k1 * [1 + 2(1+z/2) + 2(1+z/2+z^2/4) + (1+z+z^2/2+z^3/4)]
       = x0 + dt/6 * k1 * [6 + 3z + z^2 + z^3/4]
       = x0 + dt*k1*(1 + z/2 + z^2/6 + z^3/24)
    Substituting k1 = a*x0 + g and collecting the x0 and g terms separately
    (dt*a = z):
       x1 = x0*(1 + z + z^2/2 + z^3/6 + z^4/24) + dt*g*(1 + z/2 + z^2/6 + z^3/24)
          = R(z)*x0 + dt*g*S(z)
    R(z) = 1+z+z^2/2!+z^3/3!+z^4/4! is the classical RK4 stability polynomial
    (truncated exp(z) to 4th order) -- the standard textbook check that RK4
    reproduces the first five Taylor terms of e^z on a linear ODE. S(z) is
    the analogous polynomial for the constant-forcing term, and R(z) =
    1 + z*S(z) (substitute and check: 1 + z + z^2/2 + z^3/6 + z^4/24, matches).
    """
    z = a * dt
    g = b * u
    R = 1.0 + z + z**2 / 2.0 + z**3 / 6.0 + z**4 / 24.0
    S = 1.0 + z / 2.0 + z**2 / 6.0 + z**3 / 24.0
    return R * x0 + dt * g * S


def _hand_euler_step(a: float, b: float, u: float, x0: float, dt: float) -> float:
    """Explicit (forward) Euler: x1 = x0 + dt*f(x0,u) = x0*(1+z) + dt*g."""
    z = a * dt
    g = b * u
    return x0 * (1.0 + z) + dt * g


def _hand_heun_step(a: float, b: float, u: float, x0: float, dt: float) -> float:
    """Heun / explicit RK2: k1=f(x0), k2=f(x0+dt*k1), x1=x0+dt/2*(k1+k2).
    For linear f, k2 = k1*(1+z), so x1 = x0 + dt*k1*(1+z/2)
    = x0*(1+z+z^2/2) + dt*g*(1+z/2) -- matches exp(z) to 3rd order only
    (missing the RK4 z^3/6 and z^4/24 terms), so it disagrees with the RK4
    oracle above at any nonzero z.
    """
    z = a * dt
    g = b * u
    return x0 * (1.0 + z + z**2 / 2.0) + dt * g * (1.0 + z / 2.0)


# a*dt = 0.2: nonzero, moderate -- large enough that RK4 vs Euler/Heun
# disagree by orders of magnitude above tol, small enough that float64
# products of a handful of low-degree polynomial terms stay exact to ~1e-15
_A, _B, _U, _X0, _DT = 0.8, -1.5, 0.6, 1.7, 0.25
# float64 rounding after ~10 arithmetic ops (4 RK4 stages + combination) on
# O(1) operands: bound comfortably by the classic ~1e-15 * a-few unit-in-last-
# place accumulation, so 1e-12 has 3 orders of margin, not a loose guess
_TOL = 1e-12


def test_rk4_matches_hand_derived_affine_oracle() -> None:
    """Positive: one RK4 step of an affine system (nonzero B, nonzero u)
    against the exact hand-derived closed form above."""
    system = _AffineSystem(_A, _B)
    traj = system.integrate(np.array([_X0]), np.array([[_U]]), _DT)
    x1_numeric = traj[-1, 0]
    x1_oracle = _hand_rk4_step(_A, _B, _U, _X0, _DT)
    assert abs(x1_numeric - x1_oracle) < _TOL


def test_rk4_oracle_rejects_euler_and_heun_steps() -> None:
    """Negative: the SAME oracle/tolerance that accepts the true RK4 step
    must reject an explicit-Euler step and a Heun/RK2 step -- proving the
    oracle actually discriminates the coefficients, not just the order of
    accuracy or the general shape of the update."""
    x1_euler = _hand_euler_step(_A, _B, _U, _X0, _DT)
    x1_heun = _hand_heun_step(_A, _B, _U, _X0, _DT)
    x1_oracle = _hand_rk4_step(_A, _B, _U, _X0, _DT)
    assert abs(x1_euler - x1_oracle) > _TOL
    assert abs(x1_heun - x1_oracle) > _TOL

    # and the actual integrator (which implements RK4, not Euler/Heun) must
    # likewise disagree with those two wrong methods at this tolerance
    system = _AffineSystem(_A, _B)
    x1_numeric = system.integrate(np.array([_X0]), np.array([[_U]]), _DT)[-1, 0]
    assert abs(x1_numeric - x1_euler) > _TOL
    assert abs(x1_numeric - x1_heun) > _TOL
