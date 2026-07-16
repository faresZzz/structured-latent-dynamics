import numpy as np

from sld.systems.damped_pendulum import DampedPendulum


def test_small_angle_matches_analytic_linear_solution():
    # At amplitude 1e-3 the nonlinear correction to sin(theta) is
    # O(theta0^3) ~ 1e-9, far below the 1e-6 tolerance below, so the
    # analytic solution of the LINEARIZED equation is a valid reference.
    zeta = 0.1
    pendulum = DampedPendulum(zeta=zeta)
    theta0, theta_dot0 = 1e-3, 0.0
    dt = 1e-3
    n_steps = 10_000  # 10 s
    u_seq = np.zeros((n_steps, 1))
    traj = pendulum.integrate(np.array([theta0, theta_dot0]), u_seq, dt)

    omega0 = pendulum.omega0
    omega_d = omega0 * np.sqrt(1.0 - zeta**2)
    t = np.arange(n_steps + 1) * dt
    theta_analytic = np.exp(-zeta * omega0 * t) * (
        theta0 * np.cos(omega_d * t)
        + ((theta_dot0 + zeta * omega0 * theta0) / omega_d) * np.sin(omega_d * t)
    )

    max_abs_error = np.max(np.abs(traj[:, 0] - theta_analytic))
    print(f"[test1] max_abs_error = {max_abs_error:.3e}")
    assert max_abs_error < 1e-6


def test_energy_conserved_at_zero_damping():
    pendulum = DampedPendulum(zeta=0.0)
    x0 = np.array([2.0, 0.0])
    dt = 1e-3
    n_steps = 10_000  # 10 s
    u_seq = np.zeros((n_steps, 1))
    traj = pendulum.integrate(x0, u_seq, dt)

    energy = pendulum.energy(traj)
    max_relative_drift = np.max(np.abs(energy - energy[0]) / energy[0])
    print(f"[test2] max_relative_drift = {max_relative_drift:.3e}")
    assert max_relative_drift < 1e-8


def test_rk4_fourth_order_convergence():
    pendulum = DampedPendulum(zeta=0.1)
    x0 = np.array([2.0, 0.5])
    duration = 5.0
    dt_coarse = 0.02
    dt_fine = 0.01
    dt_ref = dt_coarse / 64.0  # shared reference grid for both step sizes

    def final_state(dt: float) -> np.ndarray:
        n_steps = round(duration / dt)
        u_seq = np.zeros((n_steps, 1))
        return pendulum.integrate(x0, u_seq, dt)[-1]

    x_ref = final_state(dt_ref)
    x_coarse = final_state(dt_coarse)
    x_fine = final_state(dt_fine)

    error_coarse = np.linalg.norm(x_coarse - x_ref)
    error_fine = np.linalg.norm(x_fine - x_ref)
    ratio = error_coarse / error_fine
    print(f"[test3] error(dt=0.02) = {error_coarse:.3e}, error(dt=0.01) = {error_fine:.3e}, ratio = {ratio:.3f}")
    assert 14.0 <= ratio <= 18.0


def test_analytic_eigenvalues_match_finite_difference_linearization():
    pendulum = DampedPendulum(zeta=0.1)
    A, _B = pendulum.linearize(np.zeros(2), np.zeros(1))
    eig_numeric = np.sort_complex(np.linalg.eigvals(A))
    eig_analytic = np.sort_complex(pendulum.analytic_eigenvalues())
    np.testing.assert_allclose(eig_numeric, eig_analytic, atol=1e-6)


def test_sample_ic_deterministic_and_regime_bounds():
    pendulum = DampedPendulum()

    for regime in ("default", "wide"):
        rng_a = np.random.default_rng(0)
        rng_b = np.random.default_rng(0)
        x_a = pendulum.sample_ic(rng_a, regime=regime)
        x_b = pendulum.sample_ic(rng_b, regime=regime)
        np.testing.assert_array_equal(x_a, x_b)

    rng = np.random.default_rng(1)
    for _ in range(1000):
        theta, theta_dot = pendulum.sample_ic(rng, regime="wide")
        assert -np.pi <= theta <= np.pi
        assert -2.0 * pendulum.omega0 <= theta_dot <= 2.0 * pendulum.omega0

    try:
        pendulum.sample_ic(rng, regime="bogus")
        assert False, "expected ValueError for unknown regime"
    except ValueError:
        pass


def test_partial_observation_keeps_angle_only():
    pendulum = DampedPendulum()
    rng = np.random.default_rng(2)
    x = rng.standard_normal((4, 3, 2))
    observation = pendulum.observe(x, "partial")
    assert observation.shape == (4, 3, 1)
    np.testing.assert_array_equal(observation, x[..., :1])
